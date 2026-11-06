"""VLM / rule verification (§2.2 / §2.3).

Paper: Qwen3-VL verifies query–instance correspondence with full-image context,
deduplicates, and resolves conflicts.

Engineering substitutes (honest, not Qwen3-VL):
  - rule: IoU dedup + score threshold
  - clip_semantic / vlm_json: OpenAI CLIP ViT-B/32 crop–text similarity + IoU
    safety-net (Florence-2 attempted first for vlm_json but transformers 5.x
    currently breaks Florence-2 remote code; auto-falls back to CLIP)
"""

from __future__ import annotations

import gc
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from PIL import Image as PILImage

from wde.propose.experts import Proposal
from wde.utils import iou_xyxy

ImageLike = Union[str, Path, PILImage.Image]

PROMPT_VERSION = "wde-clip-semantic-v1"
FLORENCE_PROMPT_VERSION = "wde-florence2-region-v1"


@dataclass
class VerifyDecision:
    proposal_index: int
    decision: str  # accept | reject | uncertain | duplicate
    score: float
    reason: str
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    name: str

    @abstractmethod
    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
        image: Optional[ImageLike] = None,
    ) -> list[VerifyDecision]:
        ...

    def release(self) -> None:
        """Drop heavy weights to free RAM (no-op by default)."""
        return None


def _open_image(image: Optional[ImageLike]) -> Optional[PILImage.Image]:
    if image is None:
        return None
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    return PILImage.open(Path(image)).convert("RGB")


def _apply_iou_dedup(
    proposals: Sequence[Proposal],
    decisions: list[VerifyDecision],
    *,
    iou_dedup: float,
) -> list[VerifyDecision]:
    """Safety-net: among accept decisions, mark lower-score high-IoU as duplicate."""
    accept_idxs = [d.proposal_index for d in decisions if d.decision == "accept"]
    accept_idxs.sort(key=lambda i: proposals[i].score, reverse=True)
    kept: list[int] = []
    dup_of: dict[int, int] = {}
    for idx in accept_idxs:
        hit = None
        for k in kept:
            if iou_xyxy(proposals[idx].as_xyxy(), proposals[k].as_xyxy()) >= iou_dedup:
                hit = k
                break
        if hit is not None:
            dup_of[idx] = hit
        else:
            kept.append(idx)

    out: list[VerifyDecision] = []
    for d in decisions:
        if d.proposal_index in dup_of:
            out.append(
                VerifyDecision(
                    proposal_index=d.proposal_index,
                    decision="duplicate",
                    score=d.score,
                    reason=(
                        f"IoU>={iou_dedup} with accepted proposal[{dup_of[d.proposal_index]}] "
                        f"(post-VLM safety net); prior={d.decision}"
                    ),
                    backend=d.backend,
                    raw={
                        **d.raw,
                        "duplicate_of": dup_of[d.proposal_index],
                        "prior_decision": d.decision,
                        "iou_dedup_threshold": iou_dedup,
                    },
                )
            )
        else:
            out.append(d)
    return out


def release_expert_models(experts: Sequence[Any]) -> None:
    """Unload detector weights so a VLM/CLIP verifier can fit in RAM."""
    for ex in experts:
        for attr in ("_model", "_processor", "_torch"):
            if hasattr(ex, attr):
                try:
                    setattr(ex, attr, None)
                except Exception:
                    pass
    gc.collect()
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class RuleVerifier(Verifier):
    """IoU NMS-style dedup across experts + score threshold accept."""

    name = "rule"

    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
        image: Optional[ImageLike] = None,
    ) -> list[VerifyDecision]:
        del image  # rule path ignores full-image context
        if not proposals:
            return []

        order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
        kept: list[int] = []
        decisions: list[VerifyDecision | None] = [None] * len(proposals)

        for idx in order:
            p = proposals[idx]
            if p.score < accept_min_score:
                decisions[idx] = VerifyDecision(
                    proposal_index=idx,
                    decision="reject",
                    score=p.score,
                    reason=f"score {p.score:.3f} < {accept_min_score}",
                    backend=self.name,
                    raw={"mock": False, "mock_vlm": False, "stands_in_for": "Qwen3-VL"},
                )
                continue
            dup_of = None
            for k in kept:
                if iou_xyxy(p.as_xyxy(), proposals[k].as_xyxy()) >= iou_dedup:
                    dup_of = k
                    break
            if dup_of is not None:
                decisions[idx] = VerifyDecision(
                    proposal_index=idx,
                    decision="duplicate",
                    score=p.score,
                    reason=f"IoU>={iou_dedup} with proposal[{dup_of}] ({proposals[dup_of].expert_name})",
                    backend=self.name,
                    raw={"duplicate_of": dup_of, "mock": False, "stands_in_for": "Qwen3-VL"},
                )
            else:
                kept.append(idx)
                decisions[idx] = VerifyDecision(
                    proposal_index=idx,
                    decision="accept",
                    score=p.score,
                    reason="passed score + IoU dedup (rule stand-in for Qwen3-VL)",
                    backend=self.name,
                    raw={
                        "mock": False,
                        "note": "rule geometry/score — NOT Qwen3-VL semantic verification",
                        "stands_in_for": "Qwen3-VL",
                        "query": query_text,
                        "query_type": query_type,
                    },
                )

        return [d for d in decisions if d is not None]


class ClipSemanticVerifier(Verifier):
    """Weak semantic verifier: CLIP crop↔query similarity + IoU dedup.

    Honest substitute for paper Qwen3-VL — weaker, no generative reasoning.
    Uses full-image CLIP score as context signal alongside the padded crop.
    """

    name = "clip_semantic"
    DEFAULT_WEIGHTS = "weights/clip/ViT-B-32.pt"
    DEFAULT_MODEL_NAME = "openai/CLIP-ViT-B/32"

    def __init__(
        self,
        *,
        weights: Optional[str] = None,
        device: str = "cpu",
        accept_min: float = 0.22,
        reject_max: float = 0.18,
        pad_ratio: float = 0.12,
        fallback_reason: Optional[str] = None,
        stands_in_for: str = "Qwen3-VL",
    ) -> None:
        self.weights = weights
        self.device = device
        self.accept_min = accept_min
        self.reject_max = reject_max
        self.pad_ratio = pad_ratio
        self.fallback_reason = fallback_reason
        self.stands_in_for = stands_in_for
        self._model: Any = None
        self._preprocess: Any = None
        self._clip_mod: Any = None
        self._torch: Any = None
        self._resolved_weights: Optional[str] = None

    def _resolve_weights(self) -> str:
        if self._resolved_weights:
            return self._resolved_weights
        from wde.config import project_root

        candidates: list[Path] = []
        if self.weights:
            candidates.append(Path(self.weights))
        candidates.append(project_root() / self.DEFAULT_WEIGHTS)
        candidates.append(Path.cwd() / self.DEFAULT_WEIGHTS)
        for c in candidates:
            if c.is_file():
                self._resolved_weights = str(c.resolve())
                return self._resolved_weights
        # Let openai-clip download ViT-B/32
        self._resolved_weights = "ViT-B/32"
        return self._resolved_weights

    def _load(self) -> None:
        if self._model is not None:
            return
        import clip as openai_clip
        import torch

        w = self._resolve_weights()
        model, preprocess = openai_clip.load(w, device=self.device, jit=False)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._clip_mod = openai_clip
        self._torch = torch

    def release(self) -> None:
        self._model = None
        self._preprocess = None
        self._clip_mod = None
        self._torch = None
        gc.collect()
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _crop(self, im: PILImage.Image, box: tuple[float, float, float, float]) -> PILImage.Image:
        w, h = im.size
        x1, y1, x2, y2 = box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x, pad_y = bw * self.pad_ratio, bh * self.pad_ratio
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return im
        return im.crop((cx1, cy1, cx2, cy2))

    def _cosine(self, pil_img: PILImage.Image, text: str) -> float:
        self._load()
        torch = self._torch
        clip = self._clip_mod
        image_input = self._preprocess(pil_img).unsqueeze(0).to(self.device)
        # CLIP context length 77; keep prompts short
        text_prompt = text if len(text) < 60 else text[:57] + "..."
        tokens = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = float((image_features @ text_features.T)[0, 0].item())
        return sim

    @staticmethod
    def _query_prompt(query_text: str, query_type: str) -> tuple[str, str]:
        """Return (clip_text, note). Referring uses noun heuristic — weaker path."""
        q = (query_text or "").strip()
        if query_type == "category":
            return f"a photo of a {q}", "category"
        # Referring: strip stopwords, keep last content tokens
        stop = {
            "the", "a", "an", "on", "in", "at", "of", "to", "for", "and", "or", "near",
            "left", "right", "upper", "lower", "largest", "smallest", "region", "center",
            "coloured", "colored", "that", "does", "not", "exist", "with", "from", "by",
        }
        tokens = re.findall(r"[a-z0-9_]+", q.lower())
        content = [t for t in tokens if t not in stop and not t.isdigit()]
        noun = content[-1] if content else q
        return f"a photo of a {noun}", f"referring_degraded_to_noun:{noun}"

    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
        image: Optional[ImageLike] = None,
    ) -> list[VerifyDecision]:
        if not proposals:
            return []

        im = _open_image(image)
        if im is None:
            # Without image we cannot do semantic check — mark uncertain, not accept
            return [
                VerifyDecision(
                    proposal_index=i,
                    decision="uncertain",
                    score=p.score,
                    reason="clip_semantic requires full image; none provided",
                    backend=self.name,
                    raw={
                        "mock": False,
                        "model": self.DEFAULT_MODEL_NAME,
                        "stands_in_for": self.stands_in_for,
                        "prompt_version": PROMPT_VERSION,
                        "error": "missing_image",
                    },
                )
                for i, p in enumerate(proposals)
            ]

        prompt, prompt_note = self._query_prompt(query_text, query_type)
        sim_full = self._cosine(im, prompt)

        tentative: list[VerifyDecision] = []
        for idx, p in enumerate(proposals):
            # Soft detector-score gate (keep low-score junk out before CLIP)
            if p.score < accept_min_score * 0.5:
                tentative.append(
                    VerifyDecision(
                        proposal_index=idx,
                        decision="reject",
                        score=float(p.score),
                        reason=f"detector score {p.score:.3f} far below {accept_min_score}",
                        backend=self.name,
                        raw=self._base_raw(prompt, prompt_note, query_text, query_type, sim_full),
                    )
                )
                continue

            crop = self._crop(im, p.as_xyxy())
            sim_crop = self._cosine(crop, prompt)
            # Context: crop should be at least competitive with the whole image
            context_ok = sim_crop >= (sim_full - 0.03)

            raw = self._base_raw(prompt, prompt_note, query_text, query_type, sim_full)
            raw.update(
                {
                    "sim_crop": round(sim_crop, 4),
                    "sim_full": round(sim_full, 4),
                    "context_ok": context_ok,
                    "accept_min": self.accept_min,
                    "reject_max": self.reject_max,
                    "box": list(p.as_xyxy()),
                    "model_output_summary": (
                        f"cosine(crop, '{prompt}')={sim_crop:.4f}; "
                        f"cosine(full, '{prompt}')={sim_full:.4f}"
                    ),
                }
            )

            if sim_crop < self.reject_max:
                decision = "reject"
                reason = (
                    f"CLIP sim_crop={sim_crop:.3f} < reject_max={self.reject_max}; "
                    f"sim_full={sim_full:.3f}"
                )
            elif sim_crop < self.accept_min:
                # Mid band → uncertain (never treat unknown as accept)
                decision = "uncertain"
                reason = (
                    f"CLIP sim_crop={sim_crop:.3f} in uncertain band "
                    f"[{self.reject_max}, {self.accept_min}); not treating as accept; "
                    f"sim_full={sim_full:.3f}, context_ok={context_ok}"
                )
            elif not context_ok and sim_crop < self.accept_min + 0.02:
                decision = "uncertain"
                reason = (
                    f"CLIP sim_crop={sim_crop:.3f} barely above accept but "
                    f"below full-image context (sim_full={sim_full:.3f})"
                )
            else:
                decision = "accept"
                reason = (
                    f"CLIP sim_crop={sim_crop:.3f} >= {self.accept_min} "
                    f"(sim_full={sim_full:.3f}, context_ok={context_ok})"
                )

            tentative.append(
                VerifyDecision(
                    proposal_index=idx,
                    decision=decision,
                    score=float(sim_crop),
                    reason=reason,
                    backend=self.name,
                    raw=raw,
                )
            )

        return _apply_iou_dedup(proposals, tentative, iou_dedup=iou_dedup)

    def _base_raw(
        self,
        prompt: str,
        prompt_note: str,
        query_text: str,
        query_type: str,
        sim_full: float,
    ) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "mock": False,
            "model": self.DEFAULT_MODEL_NAME,
            "weights": self._resolve_weights(),
            "backend": "openai-clip",
            "stands_in_for": self.stands_in_for,
            "prompt": prompt,
            "prompt_version": PROMPT_VERSION,
            "prompt_note": prompt_note,
            "query": query_text,
            "query_type": query_type,
            "device": self.device,
            "note": (
                "WEAK semantic stand-in for Qwen3-VL (CLIP crop–text similarity). "
                "Not a generative VLM; referring queries degraded to noun prompts."
            ),
        }
        if self.fallback_reason:
            raw["fallback_from"] = "florence2"
            raw["fallback_reason"] = self.fallback_reason
        return raw


class Florence2Verifier(Verifier):
    """Attempt Florence-2-base region/caption verification (paper-style stand-in).

    On current transformers 5.x the Florence-2 remote code fails to load; callers
    should catch and fall back to ClipSemanticVerifier. Kept for honest attempt + docs.
    """

    name = "florence2"
    DEFAULT_MODEL_ID = "microsoft/Florence-2-base"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        stands_in_for: str = "Qwen3-VL",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.stands_in_for = stands_in_for
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def release(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None
        gc.collect()

    def _region_to_category(self, im: PILImage.Image, box: tuple[float, float, float, float]) -> str:
        """Run Florence <REGION_TO_CATEGORY> if available; else caption the crop."""
        self._load()
        torch = self._torch
        w, h = im.size
        x1, y1, x2, y2 = box
        # Florence location tokens are quantized to 0–999
        locs = [
            int(max(0, min(999, round(x1 / w * 999)))),
            int(max(0, min(999, round(y1 / h * 999)))),
            int(max(0, min(999, round(x2 / w * 999)))),
            int(max(0, min(999, round(y2 / h * 999)))),
        ]
        loc_str = "".join(f"<loc_{n}>" for n in locs)
        task = "<REGION_TO_CATEGORY>"
        prompt = task + loc_str
        inputs = self._processor(text=prompt, images=im, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=32,
                num_beams=1,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text, task=task, image_size=(w, h)
        )
        # parsed typically {task: "category_name"}
        if isinstance(parsed, dict):
            val = parsed.get(task) or next(iter(parsed.values()), "")
            return str(val).strip().lower()
        return str(parsed).strip().lower()

    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
        image: Optional[ImageLike] = None,
    ) -> list[VerifyDecision]:
        im = _open_image(image)
        if im is None:
            raise RuntimeError("Florence2Verifier requires full image context")

        self._load()
        q = (query_text or "").strip().lower()
        # Referring: compare against core noun
        if query_type != "category":
            tokens = re.findall(r"[a-z0-9_]+", q)
            stop = {"the", "a", "an", "on", "in", "at", "of", "to", "for", "and", "or", "near", "left", "right"}
            content = [t for t in tokens if t not in stop]
            q_match = content[-1] if content else q
        else:
            q_match = q

        tentative: list[VerifyDecision] = []
        for idx, p in enumerate(proposals):
            if p.score < accept_min_score * 0.5:
                tentative.append(
                    VerifyDecision(
                        proposal_index=idx,
                        decision="reject",
                        score=float(p.score),
                        reason=f"detector score too low ({p.score:.3f})",
                        backend=self.name,
                        raw={"mock": False, "model": self.model_id, "stands_in_for": self.stands_in_for},
                    )
                )
                continue
            try:
                cat = self._region_to_category(im, p.as_xyxy())
            except Exception as exc:
                tentative.append(
                    VerifyDecision(
                        proposal_index=idx,
                        decision="uncertain",
                        score=float(p.score),
                        reason=f"Florence inference failed: {type(exc).__name__}: {exc}",
                        backend=self.name,
                        raw={
                            "mock": False,
                            "model": self.model_id,
                            "stands_in_for": self.stands_in_for,
                            "prompt_version": FLORENCE_PROMPT_VERSION,
                            "error": str(exc),
                        },
                    )
                )
                continue

            cat_l = cat.lower()
            raw = {
                "mock": False,
                "model": self.model_id,
                "backend": "transformers.Florence-2",
                "stands_in_for": self.stands_in_for,
                "prompt_version": FLORENCE_PROMPT_VERSION,
                "prompt": "<REGION_TO_CATEGORY>+locs",
                "model_output_summary": cat[:200],
                "query": query_text,
                "query_type": query_type,
                "predicted_category": cat,
            }
            if not cat or cat in {"", "none", "unknown", "n/a"}:
                decision, reason = "uncertain", f"Florence returned empty/unknown category: {cat!r}"
            elif q_match in cat_l or cat_l in q_match:
                decision, reason = "accept", f"Florence category {cat!r} matches query {q_match!r}"
            else:
                decision, reason = "reject", f"Florence category {cat!r} != query {q_match!r}"

            tentative.append(
                VerifyDecision(
                    proposal_index=idx,
                    decision=decision,
                    score=float(p.score),
                    reason=reason,
                    backend=self.name,
                    raw=raw,
                )
            )

        return _apply_iou_dedup(proposals, tentative, iou_dedup=iou_dedup)


class VLMJSONVerifier(Verifier):
    """Paper-facing backend name `vlm_json`.

    Tries Florence-2-base first; on load/API failure falls back to CLIP semantic.
    Never pretends to be Qwen3-VL.
    """

    name = "vlm_json"

    def __init__(
        self,
        *,
        prefer: str = "florence2",
        clip_kwargs: Optional[dict[str, Any]] = None,
        florence_kwargs: Optional[dict[str, Any]] = None,
        fallback_to_rule: bool = False,
    ) -> None:
        self.prefer = prefer
        self.clip_kwargs = clip_kwargs or {}
        self.florence_kwargs = florence_kwargs or {}
        self.fallback_to_rule = fallback_to_rule
        self._impl: Optional[Verifier] = None
        self._init_error: Optional[str] = None
        self._fallback_chain: list[str] = []

    def _ensure(self) -> Verifier:
        if self._impl is not None:
            return self._impl

        errors: list[str] = []
        if self.prefer in {"florence2", "florence", "vlm", "vlm_json", "auto"}:
            try:
                fv = Florence2Verifier(**self.florence_kwargs)
                fv._load()  # fail fast
                self._impl = fv
                self._fallback_chain.append("florence2")
                return self._impl
            except Exception as exc:
                msg = f"Florence-2 unavailable ({type(exc).__name__}: {exc})"
                errors.append(msg)
                self._init_error = msg

        try:
            cv = ClipSemanticVerifier(
                fallback_reason=self._init_error or "; ".join(errors) or None,
                **self.clip_kwargs,
            )
            cv._load()
            self._impl = cv
            self._fallback_chain.append("clip_semantic")
            return self._impl
        except Exception as exc:
            errors.append(f"CLIP unavailable ({type(exc).__name__}: {exc})")
            self._init_error = "; ".join(errors)
            if self.fallback_to_rule:
                self._impl = RuleVerifier()
                self._fallback_chain.append("rule")
                return self._impl
            raise RuntimeError(
                "VLM verifier failed to load Florence-2 and CLIP. "
                f"Details: {self._init_error}"
            ) from exc

    def release(self) -> None:
        if self._impl is not None:
            self._impl.release()
        self._impl = None

    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
        image: Optional[ImageLike] = None,
    ) -> list[VerifyDecision]:
        impl = self._ensure()
        decisions = impl.verify(
            query_text,
            query_type,
            proposals,
            iou_dedup=iou_dedup,
            accept_min_score=accept_min_score,
            image=image,
        )
        # Stamp outer backend name + fallback chain on every raw blob
        for d in decisions:
            d.backend = self.name if impl.name != "rule" else f"{self.name}->rule"
            d.raw = {
                **d.raw,
                "vlm_json_impl": impl.name,
                "fallback_chain": list(self._fallback_chain),
                "stands_in_for": d.raw.get("stands_in_for", "Qwen3-VL"),
                "mock": False,
            }
            if self._init_error and impl.name != "florence2":
                d.raw.setdefault("fallback_reason", self._init_error)
        return decisions


def build_verifier(name: str, cfg: Optional[dict[str, Any]] = None) -> Verifier:
    """Build verifier by backend name.

    Aliases:
      rule
      clip_semantic | clip
      florence2 | florence
      vlm_json | vlm   → VLMJSONVerifier (Florence then CLIP)
    """
    cfg = cfg or {}
    key = (name or "rule").lower().strip()
    clip_kwargs = {
        "weights": cfg.get("clip_weights"),
        "device": cfg.get("device", "cpu"),
        "accept_min": float(cfg.get("clip_accept_min", cfg.get("accept_min", 0.22))),
        "reject_max": float(cfg.get("clip_reject_max", cfg.get("reject_max", 0.18))),
        "pad_ratio": float(cfg.get("clip_pad_ratio", 0.12)),
    }
    # Drop None weights so ClipSemanticVerifier uses default path
    if clip_kwargs["weights"] is None:
        clip_kwargs.pop("weights")

    florence_kwargs = {
        "model_id": cfg.get("florence_model", Florence2Verifier.DEFAULT_MODEL_ID),
        "device": cfg.get("device", "cpu"),
    }
    fallback_to_rule = bool(cfg.get("fallback_to_rule", False))

    if key == "rule":
        return RuleVerifier()
    if key in {"clip_semantic", "clip"}:
        return ClipSemanticVerifier(**clip_kwargs)
    if key in {"florence2", "florence"}:
        # Direct Florence — no auto CLIP fallback (caller may wrap)
        return Florence2Verifier(**florence_kwargs)
    if key in {"vlm_json", "vlm"}:
        return VLMJSONVerifier(
            prefer=cfg.get("vlm_prefer", "florence2"),
            clip_kwargs=clip_kwargs,
            florence_kwargs=florence_kwargs,
            fallback_to_rule=fallback_to_rule,
        )
    raise ValueError(f"Unknown verify backend: {name!r}")
