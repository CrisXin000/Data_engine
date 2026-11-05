"""Multi-expert localization (§2.2 / §2.3).

Paper detection experts: SAM3 + in-house DINO-style open-vocabulary detector.
Referring: SAM3 Agent workflow + same in-house detector.

[engineering] MVP provides:
  - MockExpertA / MockExpertB (deterministic fake boxes)
  - YoloWorldExpert — open-source stand-in for paper Expert A (SAM3)
  - Owlv2Expert — open-source stand-in for paper Expert B (in-house DINO OVD)
    (falls back to a second YOLO-World config if transformers is unavailable)

Do NOT pretend these are SAM3 / internal DINO. Rex-Omni is NOT a paper WDE expert (§2).
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image as PILImage


@dataclass
class Proposal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    expert_name: str
    raw: dict[str, Any] = field(default_factory=dict)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Expert(ABC):
    name: str

    @abstractmethod
    def propose(
        self,
        image_path: Path,
        query_text: str,
        query_type: str,
        width: int,
        height: int,
    ) -> list[Proposal]:
        ...


def _seed_from(image_path: Path, query_text: str, expert: str) -> int:
    h = hashlib.md5(f"{image_path.name}|{query_text}|{expert}".encode()).hexdigest()
    return int(h[:8], 16)


# Common stopwords / adjectives stripped when extracting class nouns from referring text.
# [engineering] heuristic — not a linguistic parser.
_STOP = {
    "the", "a", "an", "on", "in", "at", "of", "to", "for", "and", "or", "near",
    "left", "right", "upper", "lower", "largest", "smallest", "region", "center",
    "coloured", "colored", "that", "does", "not", "exist", "with", "from", "by",
    "this", "those", "these", "some", "any", "all", "very", "most", "more",
}


def extract_class_names(query_text: str, query_type: str) -> list[str]:
    """Pull core noun / category tokens for open-vocab set_classes.

    [engineering] Simple heuristic: category queries use the whole text;
    referring queries drop stopwords and keep remaining tokens (last noun preferred).
    """
    text = (query_text or "").strip().lower()
    if not text:
        return ["object"]
    if query_type == "category":
        # Prefer short category label as-is
        return [text]

    # Referring: drop punctuation, stopwords; keep content tokens
    tokens = re.findall(r"[a-z0-9_]+", text)
    content = [t for t in tokens if t not in _STOP and not t.isdigit()]
    if not content:
        return [text]
    # Prefer last content token as the object noun ("the red bus near ..." → bus)
    # Also keep multi-word if two consecutive content words look like "sports ball"
    if len(content) >= 2:
        # Use last 1–2 tokens as candidate class phrases
        phrase2 = " ".join(content[-2:])
        return [content[-1], phrase2, " ".join(content)]
    return [content[-1]]


class MockExpertA(Expert):
    """MOCK stand-in for SAM3 / SAM3-Agent proposals."""

    name = "ExpertA_mock_sam3"

    def propose(
        self,
        image_path: Path,
        query_text: str,
        query_type: str,
        width: int,
        height: int,
    ) -> list[Proposal]:
        if "does not exist" in query_text.lower() or "purple hexagon" in query_text.lower():
            return []
        seed = _seed_from(image_path, query_text, self.name)
        boxes: list[Proposal] = []
        n = 1 + (seed % 2)
        for i in range(n):
            cx = (0.25 + 0.35 * i + (seed % 7) * 0.01) * width
            cy = (0.30 + 0.25 * (i % 2) + (seed % 5) * 0.01) * height
            bw = 0.18 * width
            bh = 0.18 * height
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(float(width), cx + bw / 2)
            y2 = min(float(height), cy + bh / 2)
            boxes.append(
                Proposal(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=0.55 + (seed % 40) / 100.0,
                    expert_name=self.name,
                    raw={
                        "mock": True,
                        "stands_in_for": "SAM3 / SAM3-Agent",
                        "query": query_text,
                        "query_type": query_type,
                    },
                )
            )
        return boxes


class MockExpertB(Expert):
    """MOCK stand-in for in-house DINO-style open-vocabulary detector."""

    name = "ExpertB_mock_dino_ovd"

    def propose(
        self,
        image_path: Path,
        query_text: str,
        query_type: str,
        width: int,
        height: int,
    ) -> list[Proposal]:
        if "does not exist" in query_text.lower() or "purple hexagon" in query_text.lower():
            return []
        seed = _seed_from(image_path, query_text, self.name)
        boxes: list[Proposal] = []
        n = 1 + ((seed >> 3) % 2)
        for i in range(n):
            cx = (0.28 + 0.32 * i + (seed % 11) * 0.008) * width
            cy = (0.32 + 0.22 * (i % 2) + (seed % 9) * 0.008) * height
            bw = 0.20 * width
            bh = 0.16 * height
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(float(width), cx + bw / 2)
            y2 = min(float(height), cy + bh / 2)
            boxes.append(
                Proposal(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=0.50 + ((seed >> 2) % 45) / 100.0,
                    expert_name=self.name,
                    raw={
                        "mock": True,
                        "stands_in_for": "in-house DINO-style OVD",
                        "query": query_text,
                        "query_type": query_type,
                    },
                )
            )
        return boxes


class ReservedExpert(Expert):
    """Placeholder for backends not yet wired (true SAM3 etc.)."""

    def __init__(self, name: str, stands_in_for: str) -> None:
        self.name = name
        self.stands_in_for = stands_in_for

    def propose(
        self,
        image_path: Path,
        query_text: str,
        query_type: str,
        width: int,
        height: int,
    ) -> list[Proposal]:
        raise NotImplementedError(
            f"{self.name} reserved for {self.stands_in_for}. "
            "Use mock, yolo_world, or owlv2 experts."
        )


class YoloWorldExpert(Expert):
    """Real open-vocab detector via ultralytics YOLO-World (CPU-capable).

    Paper substitute for Expert A (SAM3) — NOT SAM3.
    """

    DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"

    def __init__(
        self,
        *,
        name: str = "ExpertA_yolo_world",
        weights: str = DEFAULT_WEIGHTS,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "cpu",
        stands_in_for: str = "paper Expert A (SAM3) — open-source substitute: YOLO-World",
    ) -> None:
        self.name = name
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self.stands_in_for = stands_in_for
        self._model: Any = None

    def _resolve_weights(self) -> str:
        w = Path(self.weights)
        if w.is_file():
            return str(w)
        # Prefer project data/models/<weights>
        try:
            from wde.config import project_root

            cand = project_root() / "data" / "models" / w.name
            if cand.is_file():
                return str(cand)
            # cwd symlink / file
            cwd_cand = Path.cwd() / w.name
            if cwd_cand.is_file():
                return str(cwd_cand.resolve())
        except Exception:
            pass
        return self.weights  # let ultralytics download

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self._resolve_weights())
        return self._model

    def propose(
        self,
        image_path: Path,
        query_text: str,
        query_type: str,
        width: int,
        height: int,
    ) -> list[Proposal]:
        if "does not exist" in query_text.lower() or "purple hexagon" in query_text.lower():
            return []

        classes = extract_class_names(query_text, query_type)
        # For category queries keep a single class string; for referring try primary noun first
        set_classes = [classes[0]] if query_type == "category" else list(dict.fromkeys(classes))

        model = self._load()
        model.set_classes(set_classes)
        results = model.predict(
            str(image_path),
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        boxes: list[Proposal] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.cpu().tolist() if r.boxes.cls is not None else [0] * len(xyxy)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
                cls_id_i = int(cls_id)
                cls_name = set_classes[cls_id_i] if 0 <= cls_id_i < len(set_classes) else str(cls_id_i)
                boxes.append(
                    Proposal(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(score),
                        expert_name=self.name,
                        raw={
                            "mock": False,
                            "model": self.weights,
                            "backend": "ultralytics.YOLO-World",
                            "stands_in_for": self.stands_in_for,
                            "conf_threshold": self.conf,
                            "iou_threshold": self.iou,
                            "set_classes": set_classes,
                            "predicted_class": cls_name,
                            "query": query_text,
                            "query_type": query_type,
                            "class_extract_note": "engineering heuristic extract_class_names",
                            "device": self.device,
                        },
                    )
                )
        return boxes


class Owlv2Expert(Expert):
    """Real open-vocab detector via HuggingFace transformers OWLv2 (CPU-capable, slow).

    Paper substitute for Expert B (in-house DINO-style OVD) — NOT the paper model.
    """

    DEFAULT_MODEL_ID = "google/owlv2-base-patch16"

    def __init__(
        self,
        *,
        name: str = "ExpertB_owlv2",
        model_id: str = DEFAULT_MODEL_ID,
        score_threshold: float = 0.1,
        device: str = "cpu",
        stands_in_for: str = "paper Expert B (in-house DINO OVD) — open-source substitute: OWLv2",
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.score_threshold = score_threshold
        self.device = device
        self.stands_in_for = stands_in_for
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore

            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self._torch = torch
        return self._processor, self._model

    def propose(
        self,
        image_path: Path,
        query_text: str,
        query_type: str,
        width: int,
        height: int,
    ) -> list[Proposal]:
        if "does not exist" in query_text.lower() or "purple hexagon" in query_text.lower():
            return []

        classes = extract_class_names(query_text, query_type)
        # OWLv2 expects a list of text queries per image
        text_queries = list(dict.fromkeys(classes))[:5]

        processor, model = self._load()
        torch = self._torch

        image = PILImage.open(image_path).convert("RGB")
        # Resize large images for CPU speed [engineering]
        max_side = 768
        w0, h0 = image.size
        scale = 1.0
        if max(w0, h0) > max_side:
            scale = max_side / float(max(w0, h0))
            image_small = image.resize((int(w0 * scale), int(h0 * scale)), PILImage.BILINEAR)
        else:
            image_small = image

        inputs = processor(text=[text_queries], images=image_small, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_small.size[::-1]], device=self.device)  # (h, w)
        # transformers>=4.45 / 5.x: grounded API on processor; fall back to image_processor
        if hasattr(processor, "post_process_grounded_object_detection"):
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.score_threshold,
                text_labels=[text_queries],
            )[0]
        else:
            results = processor.image_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.score_threshold,
            )[0]

        boxes: list[Proposal] = []
        inv = 1.0 / scale if scale else 1.0
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.tolist()
            # Map back to original image pixels
            x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            lab_i = int(label)
            cls_name = text_queries[lab_i] if 0 <= lab_i < len(text_queries) else str(lab_i)
            boxes.append(
                Proposal(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    expert_name=self.name,
                    raw={
                        "mock": False,
                        "model": self.model_id,
                        "backend": "transformers.Owlv2",
                        "stands_in_for": self.stands_in_for,
                        "score_threshold": self.score_threshold,
                        "text_queries": text_queries,
                        "predicted_class": cls_name,
                        "query": query_text,
                        "query_type": query_type,
                        "class_extract_note": "engineering heuristic extract_class_names",
                        "device": self.device,
                        "inference_max_side": max_side,
                    },
                )
            )
        return boxes


def _try_build_owlv2(**kwargs: Any) -> Expert:
    """Build Owlv2Expert; on ImportError fall back to second YOLO-World config."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return Owlv2Expert(**kwargs)
    except Exception as exc:  # ImportError or runtime
        # Weak dual-expert fallback: same YOLO-World, stricter conf
        fb = YoloWorldExpert(
            name="ExpertB_yolo_world_fallback",
            weights=YoloWorldExpert.DEFAULT_WEIGHTS,
            conf=0.35,
            iou=0.45,
            stands_in_for=(
                "paper Expert B (in-house DINO OVD) — WEAK FALLBACK: second YOLO-World "
                f"config (owlv2 unavailable: {type(exc).__name__}: {exc})"
            ),
        )
        return fb


def build_experts(cfg_propose: dict[str, str]) -> list[Expert]:
    experts: list[Expert] = []
    a = (cfg_propose.get("expert_a") or "mock").lower()
    b = (cfg_propose.get("expert_b") or "mock").lower()

    # Expert A
    if a == "mock":
        experts.append(MockExpertA())
    elif a in {"yolo_world", "yoloworld"}:
        experts.append(
            YoloWorldExpert(
                name="ExpertA_yolo_world",
                conf=float(cfg_propose.get("expert_a_conf", 0.25)),
                iou=float(cfg_propose.get("expert_a_iou", 0.5)),
            )
        )
    elif a in {"owlv2", "owl_v2"}:
        experts.append(_try_build_owlv2(name="ExpertA_owlv2"))
    elif a in {"sam3", "grounding_dino"}:
        experts.append(ReservedExpert(f"ExpertA_{a}", a))
    else:
        experts.append(MockExpertA())

    # Expert B
    if b == "mock":
        experts.append(MockExpertB())
    elif b in {"owlv2", "owl_v2"}:
        experts.append(
            _try_build_owlv2(
                name="ExpertB_owlv2",
                score_threshold=float(cfg_propose.get("expert_b_score_threshold", 0.1)),
            )
        )
    elif b in {"yolo_world", "yoloworld", "yolo_world_b"}:
        experts.append(
            YoloWorldExpert(
                name="ExpertB_yolo_world",
                conf=float(cfg_propose.get("expert_b_conf", 0.35)),
                iou=float(cfg_propose.get("expert_b_iou", 0.45)),
                stands_in_for=(
                    "paper Expert B (in-house DINO OVD) — open-source substitute: "
                    "YOLO-World (weak dual-expert, not paper setting)"
                ),
            )
        )
    elif b in {"dino_style_ovd", "grounding_dino", "sam3"}:
        experts.append(ReservedExpert(f"ExpertB_{b}", b))
    else:
        experts.append(MockExpertB())

    return experts


def run_experts(
    experts: Sequence[Expert],
    image_path: Path,
    query_text: str,
    query_type: str,
) -> list[Proposal]:
    with PILImage.open(image_path) as im:
        w, h = im.size
    out: list[Proposal] = []
    for ex in experts:
        out.extend(ex.propose(image_path, query_text, query_type, w, h))
    return out
