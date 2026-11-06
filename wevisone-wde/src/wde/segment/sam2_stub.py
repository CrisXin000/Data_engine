"""SAM2 mask annotation (§2.4).

``backend=stub`` writes no file (M1 default, kept).
Non-stub backends dispatch to ``wde.segment.backend`` (ultralytics SAM2-Tiny
stand-in, MobileSAM, or box_fill). This is not the official Meta sam2 package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MaskResult:
    kind: str = "mask"
    path: Optional[str] = None
    source: str = "stub"
    meta: dict[str, Any] | None = None


def segment_instance(
    *,
    image_path: str,
    box_xyxy: tuple[float, float, float, float],
    backend: str = "stub",
    output_path: Optional[str] = None,
    instance_uid: Optional[str] = None,
    query_text: Optional[str] = None,
    query_id: Optional[int] = None,
    extra_meta: Optional[dict[str, Any]] = None,
    weights: Optional[str] = None,
    device: str = "cpu",
    imgsz: int = 512,
) -> MaskResult:
    name = (backend or "stub").strip().lower()
    if name in {"stub", ""}:
        return MaskResult(
            kind="mask",
            path=None,
            source="stub",
            meta={
                "note": "M1 stub — SAM2 not run",
                "box": list(box_xyxy),
                "image": image_path,
                "instance_uid": instance_uid,
                "query_text": query_text,
                "query_id": query_id,
                "mock": True,
            },
        )
    from wde.segment.backend import segment_instance as _real

    return _real(
        image_path=image_path,
        box_xyxy=box_xyxy,
        backend=name,
        output_path=output_path,
        instance_uid=instance_uid,
        query_text=query_text,
        query_id=query_id,
        extra_meta=extra_meta,
        weights=weights,
        device=device,
        imgsz=imgsz,
    )
