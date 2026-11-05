"""Semantic discovery (§2.1).

Paper: Tokenize Anything for category discovery; Qwen3-VL for referring expressions.
MVP: mock tagger + rule/template referring generator; optional YOLO-World if available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from PIL import Image as PILImage


QueryType = Literal["category", "referring"]


@dataclass
class DiscoveredQuery:
    text: str
    query_type: QueryType
    meta: dict[str, Any] = field(default_factory=dict)


class CategoryTagger(ABC):
    name: str

    @abstractmethod
    def tag(self, image_path: Path, width: int, height: int) -> list[DiscoveredQuery]:
        ...


class MockCategoryTagger(CategoryTagger):
    """Honest mock: reads sidecar colors / returns synthetic categories from image meta."""

    name = "mock_category_tagger"

    def __init__(self, default_labels: Optional[list[str]] = None) -> None:
        self.default_labels = default_labels or ["red rectangle", "blue circle", "green triangle"]

    def tag(self, image_path: Path, width: int, height: int) -> list[DiscoveredQuery]:
        # Prefer gold sidecar if present
        sidecar = image_path.with_suffix(".labels.json")
        labels = list(self.default_labels)
        if sidecar.exists():
            import json

            data = json.loads(sidecar.read_text(encoding="utf-8"))
            labels = data.get("categories") or labels
        # Heuristic: filename stem hints
        stem = image_path.stem.lower()
        if "single" in stem:
            labels = labels[:1]
        return [
            DiscoveredQuery(
                text=lab,
                query_type="category",
                meta={"backend": self.name, "note": "MOCK — not Tokenize Anything"},
            )
            for lab in labels
        ]


_DEFAULT_YOLO_WORLD_CLASSES = [
    "person", "bus", "car", "truck", "bicycle", "motorcycle", "dog", "cat",
    "chair", "bottle", "cup", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "bird", "horse", "sheep", "cow", "airplane", "train", "boat",
    "traffic light", "fire hydrant", "stop sign", "bench", "teddy bear",
]


def _read_sidecar_categories(image_path: Path) -> Optional[list[str]]:
    sidecar = image_path.with_suffix(".labels.json")
    if not sidecar.exists():
        # also allow stem.labels.json next to foo.jpg → foo.labels.json already covered;
        # try foo.jpg.labels.json style
        alt = Path(str(image_path) + ".labels.json")
        if alt.exists():
            sidecar = alt
        else:
            return None
    import json

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    cats = data.get("categories")
    return list(cats) if cats else None


class YOLOWorldCategoryTagger(CategoryTagger):
    """Real open-vocab category tagger via ultralytics YOLO-World (CPU ok if installed).

    Prefers sidecar ``*.labels.json`` categories when present (gold / curated),
    otherwise runs YOLO-World with a COCO-ish vocabulary.
    """

    name = "yolo_world"

    def __init__(self, class_names: Optional[list[str]] = None) -> None:
        self.class_names = class_names or list(_DEFAULT_YOLO_WORLD_CLASSES)
        self._model = None
        self.weights = "yolov8s-worldv2.pt"

    def _resolve_weights(self) -> str:
        from pathlib import Path as P
        w = P(self.weights)
        if w.is_file():
            return str(w)
        try:
            from wde.config import project_root
            cand = project_root() / "data" / "models" / w.name
            if cand.is_file():
                return str(cand)
        except Exception:
            pass
        return self.weights

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self._resolve_weights())
            self._model.set_classes(self.class_names)
        return self._model

    def tag(self, image_path: Path, width: int, height: int) -> list[DiscoveredQuery]:
        sidecar = _read_sidecar_categories(image_path)
        if sidecar:
            return [
                DiscoveredQuery(
                    text=n,
                    query_type="category",
                    meta={
                        "backend": "sidecar_labels",
                        "note": "curated labels.json — YOLO-World tagging skipped",
                    },
                )
                for n in sidecar
            ]
        try:
            model = self._load()
            # Refresh classes in case shared model mutated by experts
            model.set_classes(self.class_names)
            results = model.predict(str(image_path), conf=0.25, device="cpu", verbose=False)
            names: set[str] = set()
            for r in results:
                if r.boxes is None:
                    continue
                for cls_id in r.boxes.cls.tolist():
                    i = int(cls_id)
                    names.add(self.class_names[i] if i < len(self.class_names) else str(i))
            if not names:
                return MockCategoryTagger().tag(image_path, width, height)
            return [
                DiscoveredQuery(
                    text=n,
                    query_type="category",
                    meta={"backend": self.name, "model": self.weights, "mock": False},
                )
                for n in sorted(names)
            ]
        except Exception as exc:
            # Fall back to mock — do not pretend YOLO ran
            qs = MockCategoryTagger().tag(image_path, width, height)
            for q in qs:
                q.meta["yolo_world_error"] = str(exc)
                q.meta["backend"] = "mock_fallback"
            return qs


class ReferringGenerator(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        image_path: Path,
        categories: list[str],
        width: int,
        height: int,
    ) -> list[DiscoveredQuery]:
        ...


class MockReferringGenerator(ReferringGenerator):
    """Rule/template mock referring expressions. [engineering] Paper uses Qwen3-VL."""

    name = "mock_referring"

    TEMPLATES = [
        "the {color} {shape} on the left",
        "the largest {label}",
        "the {label} near the center",
        "{label} in the upper region",
    ]

    def generate(
        self,
        image_path: Path,
        categories: list[str],
        width: int,
        height: int,
    ) -> list[DiscoveredQuery]:
        out: list[DiscoveredQuery] = []
        for i, lab in enumerate(categories[:3]):
            parts = lab.split()
            color = parts[0] if len(parts) > 1 else "colored"
            shape = parts[-1] if parts else "object"
            tmpl = self.TEMPLATES[i % len(self.TEMPLATES)]
            text = tmpl.format(color=color, shape=shape, label=lab)
            out.append(
                DiscoveredQuery(
                    text=text,
                    query_type="referring",
                    meta={
                        "backend": self.name,
                        "note": "MOCK template — Qwen3-VL interface reserved",
                        "base_category": lab,
                    },
                )
            )
        # one simple negative-style template (still positive binding in M1 gold)
        if categories:
            out.append(
                DiscoveredQuery(
                    text=f"the purple hexagon that does not exist",
                    query_type="referring",
                    meta={
                        "backend": self.name,
                        "is_negative_hint": True,
                        "note": "MOCK absent-target referring query",
                    },
                )
            )
        return out


class Qwen3VLReferringInterface(ReferringGenerator):
    """Reserved interface for Qwen3-VL referring generation — not implemented."""

    name = "qwen3_vl"

    def generate(
        self,
        image_path: Path,
        categories: list[str],
        width: int,
        height: int,
    ) -> list[DiscoveredQuery]:
        raise NotImplementedError(
            "Qwen3-VL referring generator is reserved. Set referring_generator=mock for MVP."
        )


def build_category_tagger(name: str) -> CategoryTagger:
    if name == "yolo_world":
        return YOLOWorldCategoryTagger()
    return MockCategoryTagger()


def build_referring_generator(name: str) -> ReferringGenerator:
    if name == "qwen3_vl":
        return Qwen3VLReferringInterface()
    return MockReferringGenerator()


def discover_queries(
    image_path: Path,
    *,
    category_tagger: CategoryTagger,
    referring_generator: ReferringGenerator,
) -> list[DiscoveredQuery]:
    with PILImage.open(image_path) as im:
        w, h = im.size
    cats = category_tagger.tag(image_path, w, h)
    cat_texts = [c.text for c in cats]
    refs = referring_generator.generate(image_path, cat_texts, w, h)
    return cats + refs
