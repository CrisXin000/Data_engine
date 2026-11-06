"""Grounded referring-expression generation (§2.1 paper: Qwen3-VL).

[engineering] Open-source CPU-friendly stand-in:
  - structured / grounded: detect boxes (YOLO-World) → crop stats (color, position, scale)
    → English expression + structured constraints. Does NOT load Qwen3-VL.
  - qwen3_vl (config alias): optionally try a tiny caption model; on miss/OOM fall back to
    structured and record fallback in meta. Still NOT Qwen3-VL.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image as PILImage

from wde.discover.tagger import DiscoveredQuery, ReferringGenerator


# Named colors for conservative HSV/RGB nearest-neighbor labeling.
_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "red": (180, 40, 40),
    "orange": (220, 140, 40),
    "yellow": (220, 200, 50),
    "green": (50, 150, 60),
    "cyan": (40, 170, 180),
    "blue": (40, 80, 180),
    "purple": (120, 50, 160),
    "pink": (210, 100, 150),
    "brown": (120, 80, 40),
    "black": (25, 25, 25),
    "white": (230, 230, 230),
    "gray": (128, 128, 128),
}

# Categories unlikely in COCO-ish scenes — used for absent/negative hints.
_ABSENT_POOL = [
    ("purple hexagon", "hexagon"),
    ("yellow bicycle", "bicycle"),
    ("pink giraffe", "giraffe"),
    ("orange skateboard", "skateboard"),
    ("cyan fire hydrant", "fire hydrant"),
]


@dataclass
class _Det:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    color: str = "unknown"
    color_hist: dict[str, Any] = field(default_factory=dict)
    h_pos: str = "center"  # left | center | right
    v_pos: str = "middle"  # upper | middle | lower
    area: float = 0.0
    size_rank: str = "unknown"  # largest | smallest | mid | only


def empty_constraints(
    *,
    target: Optional[str] = None,
    reference: Optional[str] = None,
    absent: bool = False,
) -> dict[str, Any]:
    return {
        "attributes": {},
        "states": {},
        "actions": {},
        "relations": [],
        "part_scope": None,
        "target": target,
        "reference": reference,
        "absent": absent,
    }


def _estimate_color(crop: PILImage.Image) -> tuple[str, dict[str, Any]]:
    """Dominant named color from a downsampled crop. Conservative; no invented attributes."""
    small = crop.convert("RGB").resize((32, 32), PILImage.BILINEAR)
    pixels = list(small.getdata())
    if not pixels:
        return "unknown", {"note": "empty_crop"}

    # Drop near-gray low-sat pixels from "chromatic" vote; still keep gray/black/white bins.
    counts: dict[str, float] = {k: 0.0 for k in _COLOR_RGB}
    for r, g, b in pixels:
        best, best_d = "gray", 1e18
        for name, (cr, cg, cb) in _COLOR_RGB.items():
            d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if d < best_d:
                best_d = d
                best = name
        counts[best] += 1.0

    total = sum(counts.values()) or 1.0
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top_name, top_n = ranked[0]
    frac = top_n / total
    # Require a clear majority; otherwise unknown (do not invent).
    if frac < 0.22:
        return "unknown", {
            "top": [{"name": n, "frac": round(c / total, 3)} for n, c in ranked[:3]],
            "note": "no_clear_majority",
        }
    return top_name, {
        "top": [{"name": n, "frac": round(c / total, 3)} for n, c in ranked[:3]],
        "dominant_frac": round(frac, 3),
    }


def _position(cx: float, cy: float, w: int, h: int) -> tuple[str, str]:
    hx = cx / max(w, 1)
    hy = cy / max(h, 1)
    if hx < 0.33:
        h_pos = "left"
    elif hx > 0.67:
        h_pos = "right"
    else:
        h_pos = "center"
    if hy < 0.33:
        v_pos = "upper"
    elif hy > 0.67:
        v_pos = "lower"
    else:
        v_pos = "middle"
    return h_pos, v_pos


def _assign_size_ranks(dets: list[_Det]) -> None:
    if not dets:
        return
    if len(dets) == 1:
        dets[0].size_rank = "only"
        return
    ordered = sorted(dets, key=lambda d: -d.area)
    ordered[0].size_rank = "largest"
    ordered[-1].size_rank = "smallest"
    for d in ordered[1:-1]:
        d.size_rank = "mid"


def _resolve_yolo_weights(weights: str = "yolov8s-worldv2.pt") -> str:
    w = Path(weights)
    if w.is_file():
        return str(w)
    try:
        from wde.config import project_root

        cand = project_root() / "data" / "models" / w.name
        if cand.is_file():
            return str(cand)
        cwd_cand = Path.cwd() / w.name
        if cwd_cand.is_file():
            return str(cwd_cand.resolve())
    except Exception:
        pass
    return weights


def _detect_instances(
    image_path: Path,
    categories: list[str],
    *,
    conf: float = 0.25,
    device: str = "cpu",
) -> tuple[list[_Det], dict[str, Any]]:
    """Run YOLO-World once for grounding boxes; unload immediately after."""
    info: dict[str, Any] = {"detector": "yolo_world", "mock": False}
    labels = [c.strip() for c in categories if c and str(c).strip()]
    if not labels:
        labels = ["object"]
    # Deduplicate preserving order
    labels = list(dict.fromkeys(labels))

    model = None
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(_resolve_yolo_weights())
        model.set_classes(labels)
        results = model.predict(str(image_path), conf=conf, device=device, verbose=False)
        dets: list[_Det] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.cpu().tolist() if r.boxes.cls is not None else [0] * len(xyxy)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
                i = int(cls_id)
                lab = labels[i] if 0 <= i < len(labels) else str(i)
                dets.append(
                    _Det(
                        label=lab,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(score),
                        area=max(0.0, float(x2 - x1) * float(y2 - y1)),
                    )
                )
        info["n_detections"] = len(dets)
        info["set_classes"] = labels
        return dets, info
    except Exception as exc:
        info["detector_error"] = str(exc)
        info["n_detections"] = 0
        return [], info
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()


def _enrich_dets(image_path: Path, dets: list[_Det], width: int, height: int) -> None:
    with PILImage.open(image_path) as im:
        im = im.convert("RGB")
        for d in dets:
            x1 = max(0, int(d.x1))
            y1 = max(0, int(d.y1))
            x2 = min(width, int(d.x2))
            y2 = min(height, int(d.y2))
            if x2 <= x1 or y2 <= y1:
                d.color = "unknown"
                d.color_hist = {"note": "degenerate_box"}
            else:
                crop = im.crop((x1, y1, x2, y2))
                d.color, d.color_hist = _estimate_color(crop)
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            d.h_pos, d.v_pos = _position(cx, cy, width, height)
            d.area = max(0.0, (d.x2 - d.x1) * (d.y2 - d.y1))
    _assign_size_ranks(dets)


def _try_blip_caption(image_path: Path, box: Optional[tuple[float, float, float, float]] = None) -> Optional[str]:
    """Optional tiny caption. Returns None on ImportError / OOM / any failure. Never required."""
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore
    except Exception:
        return None
    model = None
    processor = None
    try:
        # Prefer small BLIP; skip download thrash if RAM looks desperate
        try:
            import psutil  # type: ignore

            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
            if avail_mb < 1200:
                return None
        except Exception:
            pass

        model_id = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        model.eval()
        with PILImage.open(image_path) as im:
            im = im.convert("RGB")
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im.width, x2), min(im.height, y2)
                if x2 > x1 and y2 > y1:
                    im = im.crop((x1, y1, x2, y2))
            inputs = processor(images=im, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20)
            text = processor.decode(out[0], skip_special_tokens=True).strip()
            return text or None
    except Exception:
        return None
    finally:
        try:
            del model
            del processor
        except Exception:
            pass
        gc.collect()


def _box_evidence(d: _Det) -> dict[str, Any]:
    return {
        "box_xyxy": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
        "score": round(d.score, 4),
        "color_hist": d.color_hist,
        "area": round(d.area, 1),
        "h_pos": d.h_pos,
        "v_pos": d.v_pos,
        "size_rank": d.size_rank,
    }


def _expression_for_target(d: _Det, others: list[_Det]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Build one grounded positive referring expression + constraints + evidence extras."""
    cons = empty_constraints(target=d.label, reference=None, absent=False)
    attrs: dict[str, Any] = {}
    parts: list[str] = ["the"]

    if d.color and d.color != "unknown":
        attrs["color"] = d.color
        parts.append(d.color)

    parts.append(d.label)

    # Spatial attribute (prefer horizontal)
    spatial = None
    if d.h_pos in ("left", "right"):
        spatial = d.h_pos
        attrs["horizontal_position"] = d.h_pos
        parts.extend(["on", "the", d.h_pos])
    elif d.v_pos in ("upper", "lower"):
        spatial = d.v_pos
        attrs["vertical_position"] = d.v_pos
        region = "upper region" if d.v_pos == "upper" else "lower region"
        parts.extend(["in", "the", *region.split()])

    if d.size_rank in ("largest", "smallest"):
        attrs["size_rank"] = d.size_rank

    cons["attributes"] = attrs
    cons["states"] = {}  # unknown — do not invent
    cons["actions"] = {}
    cons["part_scope"] = None

    # Relation only when a second visible referent of a different label exists
    ref = None
    for o in others:
        if o.label != d.label:
            ref = o
            break
    if ref is not None and spatial is None:
        # Prefer "near X" when we didn't already pin left/right
        cons["relations"] = [{"type": "near", "reference": ref.label, "role": "reference"}]
        cons["reference"] = ref.label
        text = f"the {d.color + ' ' if d.color != 'unknown' else ''}{d.label} near the {ref.label}".replace(
            "  ", " "
        ).strip()
        # tidy double spaces from empty color
        while "  " in text:
            text = text.replace("  ", " ")
        return text, cons, {"reference_box": _box_evidence(ref)}

    if ref is not None and spatial is not None:
        # Keep spatial expression; still record a soft near-relation if close enough
        cons["relations"] = []
        # optional: record that a reference exists but expression used position
        cons["reference"] = None

    # Size-based alternate when multiple same-label instances
    same = [o for o in [d] + others if o.label == d.label]
    if len(same) >= 2 and d.size_rank == "largest":
        text = f"the largest {d.label}"
        cons["attributes"] = {"size_rank": "largest"}
        if d.color != "unknown":
            cons["attributes"]["color"] = d.color
            text = f"the largest {d.color} {d.label}"
        cons["relations"] = []
        return text, cons, {}

    text = " ".join(parts)
    return text, cons, {}


def _leftmost_expression(d: _Det) -> tuple[str, dict[str, Any]]:
    cons = empty_constraints(target=d.label, absent=False)
    attrs: dict[str, Any] = {"horizontal_position": "leftmost"}
    if d.color != "unknown":
        attrs["color"] = d.color
    cons["attributes"] = attrs
    if d.color != "unknown":
        text = f"the leftmost {d.color} {d.label}"
    else:
        text = f"the leftmost {d.label}"
    return text, cons


def _pick_absent(categories: list[str], detected_labels: set[str]) -> tuple[str, str]:
    present = {c.lower() for c in categories} | {d.lower() for d in detected_labels}
    for phrase, noun in _ABSENT_POOL:
        # avoid suggesting something that is actually present
        if noun.lower() in present or phrase.split()[-1] in present:
            continue
        if any(noun.lower() in p for p in present):
            continue
        return phrase, noun
    return "purple hexagon", "hexagon"


class StructuredReferringGenerator(ReferringGenerator):
    """CPU-friendly grounded referring generator (open-source stand-in for Qwen3-VL)."""

    name = "structured"

    def __init__(
        self,
        *,
        enable_caption: bool = False,
        conf: float = 0.25,
        device: str = "cpu",
        max_positive: int = 4,
        backend_name: Optional[str] = None,
    ) -> None:
        self.enable_caption = enable_caption
        self.conf = conf
        self.device = device
        self.max_positive = max_positive
        self.backend_name = backend_name or self.name

    def generate(
        self,
        image_path: Path,
        categories: list[str],
        width: int,
        height: int,
    ) -> list[DiscoveredQuery]:
        dets, det_info = _detect_instances(
            image_path, categories, conf=self.conf, device=self.device
        )
        if dets:
            _enrich_dets(image_path, dets, width, height)

        caption: Optional[str] = None
        caption_note: Optional[str] = None
        if self.enable_caption:
            caption = _try_blip_caption(image_path)
            caption_note = "blip_caption" if caption else "blip_skipped_or_failed"

        out: list[DiscoveredQuery] = []
        used_texts: set[str] = set()

        def _base_meta(**extra: Any) -> dict[str, Any]:
            m: dict[str, Any] = {
                "mock": False,
                "model": "structured_grounded_v1",
                "backend": self.backend_name,
                "stands_in_for": "Qwen3-VL",
                "note": (
                    "Open-source grounded stand-in (YOLO-World boxes + crop color/position). "
                    "NOT Qwen3-VL."
                ),
                "detector": det_info,
            }
            if caption is not None:
                m["optional_caption"] = caption
            if caption_note:
                m["caption_status"] = caption_note
            m.update(extra)
            return m

        # Positive referring expressions from detections
        if dets:
            # Prefer diverse targets: sort by score, then pick unique labels first
            by_score = sorted(dets, key=lambda d: -d.score)
            picked: list[_Det] = []
            seen_labels: set[str] = set()
            for d in by_score:
                if d.label not in seen_labels:
                    picked.append(d)
                    seen_labels.add(d.label)
                if len(picked) >= self.max_positive:
                    break
            # Fill with more instances (e.g. second person) if room
            for d in by_score:
                if len(picked) >= self.max_positive:
                    break
                if d not in picked:
                    picked.append(d)

            for d in picked:
                others = [o for o in dets if o is not d]
                text, cons, extra_ev = _expression_for_target(d, others)
                if text in used_texts:
                    continue
                used_texts.add(text)
                evidence = {"target": _box_evidence(d), **extra_ev}
                out.append(
                    DiscoveredQuery(
                        text=text,
                        query_type="referring",
                        meta=_base_meta(
                            base_category=d.label,
                            constraints=cons,
                            evidence=evidence,
                            role="target",
                        ),
                    )
                )

            # Extra relational / leftmost when multiple instances of same class
            from collections import Counter

            counts = Counter(d.label for d in dets)
            multi_label = next((lab for lab, n in counts.items() if n >= 2), None)
            if multi_label and len(out) < self.max_positive + 1:
                same = [d for d in dets if d.label == multi_label]
                leftmost = min(same, key=lambda d: (d.x1 + d.x2) / 2.0)
                text, cons = _leftmost_expression(leftmost)
                if text not in used_texts:
                    used_texts.add(text)
                    out.append(
                        DiscoveredQuery(
                            text=text,
                            query_type="referring",
                            meta=_base_meta(
                                base_category=multi_label,
                                constraints=cons,
                                evidence={"target": _box_evidence(leftmost)},
                                role="target",
                            ),
                        )
                    )
        else:
            # No boxes — still emit conservative category-anchored referring (position unknown)
            for lab in categories[:2]:
                cons = empty_constraints(target=lab, absent=False)
                cons["attributes"] = {"color": "unknown", "note": "no_detection_box"}
                text = f"the {lab} near the center"
                if text in used_texts:
                    continue
                used_texts.add(text)
                out.append(
                    DiscoveredQuery(
                        text=text,
                        query_type="referring",
                        meta=_base_meta(
                            base_category=lab,
                            constraints=cons,
                            evidence={"note": "ungrounded_fallback_no_box"},
                            role="target",
                            mock=False,
                            weak_ungrounded=True,
                        ),
                    )
                )

        # At least one absent / negative referring query
        phrase, noun = _pick_absent(categories, {d.label for d in dets})
        cons_neg = empty_constraints(target=noun, absent=True)
        cons_neg["attributes"] = {"color": phrase.split()[0] if " " in phrase else "unknown"}
        cons_neg["states"] = {}
        cons_neg["actions"] = {}
        cons_neg["relations"] = []
        cons_neg["part_scope"] = None
        abs_text = f"the {phrase} that does not exist"
        out.append(
            DiscoveredQuery(
                text=abs_text,
                query_type="referring",
                meta=_base_meta(
                    base_category=noun,
                    constraints=cons_neg,
                    evidence={"note": "synthetic_absent_not_satisfied_by_image"},
                    is_negative_hint=True,
                    role="absent",
                ),
            )
        )

        return out


class GroundedReferringGenerator(StructuredReferringGenerator):
    """Alias name for StructuredReferringGenerator."""

    name = "grounded"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("backend_name", "grounded")
        super().__init__(**kwargs)


class Qwen3VLReferringAlias(ReferringGenerator):
    """Config alias `qwen3_vl`: try optional light caption, else structured fallback.

    Honest: this does NOT run Qwen3-VL. Meta always records stands_in_for and fallback.
    """

    name = "qwen3_vl"

    def __init__(self, *, enable_caption: bool = True, **kwargs: Any) -> None:
        self.enable_caption = enable_caption
        self._kwargs = kwargs

    def generate(
        self,
        image_path: Path,
        categories: list[str],
        width: int,
        height: int,
    ) -> list[DiscoveredQuery]:
        caption = None
        caption_status = "not_attempted"
        if self.enable_caption:
            caption = _try_blip_caption(image_path)
            caption_status = "ok" if caption else "failed_or_oom_or_unavailable"

        inner = StructuredReferringGenerator(
            enable_caption=False,
            backend_name="qwen3_vl_alias→structured",
            **self._kwargs,
        )
        refs = inner.generate(image_path, categories, width, height)
        for q in refs:
            q.meta["backend"] = "qwen3_vl_alias→structured"
            q.meta["stands_in_for"] = "Qwen3-VL"
            q.meta["mock"] = False
            q.meta["fallback"] = {
                "requested": "qwen3_vl",
                "resolved": "structured",
                "reason": (
                    "Qwen3-VL not wired; used YOLO-World+crop-stats structured generator"
                    + (f"; optional BLIP caption={caption!r}" if caption else f"; caption={caption_status}")
                ),
            }
            q.meta["caption_status"] = caption_status
            if caption:
                q.meta["optional_caption"] = caption
            note = q.meta.get("note", "")
            q.meta["note"] = (
                f"{note} | qwen3_vl is a config alias — open-source structured stand-in, NOT Qwen3-VL."
            ).strip(" |")
        return refs
