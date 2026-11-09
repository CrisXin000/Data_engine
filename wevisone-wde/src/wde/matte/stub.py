"""Matting stub / dispatcher (§2.6). Stub is the M1 default (skip / no file)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AlphaResult:
    kind: str = "alpha"
    path: Optional[str] = None
    source: str = "stub"
    meta: dict[str, Any] | None = None


def matte_instance(
    *,
    image_path: str,
    box_xyxy: tuple[float, float, float, float],
    backend: str = "stub",
    mask_path: Optional[str] = None,
    output_path: Optional[str] = None,
    instance_uid: Optional[str] = None,
    query_text: Optional[str] = None,
    query_id: Optional[int] = None,
    extra_meta: Optional[dict[str, Any]] = None,
    weights: Optional[str] = None,
    device: str = "cpu",
    imgsz: int = 1024,
    crop_pad_ratio: float = 0.15,
    bin_threshold: float = 0.5,
    iou_min: float = 0.5,
    coverage_min: float = 0.6,
    area_ratio_min: float = 0.5,
    area_ratio_max: float = 2.0,
    fallback: str = "feather",
    feather_radius: float = 3.0,
    feather_band: float = 4.0,
) -> AlphaResult:
    """Produce a full-image continuous alpha (or a stub no-op).

    Non-stub backends dispatch to ``wde.matte.backend``. This is an
    open-source stand-in for paper BiRefNet + MatAnyone2, not those models
    in full and not a trained paper matting annotator.
    """
    name = (backend or "stub").strip().lower()
    if name in {"stub", ""}:
        return AlphaResult(
            kind="alpha",
            path=None,
            source="stub",
            meta={
                "note": "M1 stub — matting not run",
                "box": list(box_xyxy),
                "image": image_path,
                "mask_path": mask_path,
                "instance_uid": instance_uid,
                "query_text": query_text,
                "query_id": query_id,
                "mock": True,
            },
        )
    from wde.matte.backend import matte_instance as _real

    return _real(
        image_path=image_path,
        box_xyxy=box_xyxy,
        backend=name,
        mask_path=mask_path,
        output_path=output_path,
        instance_uid=instance_uid,
        query_text=query_text,
        query_id=query_id,
        extra_meta=extra_meta,
        weights=weights,
        device=device,
        imgsz=imgsz,
        crop_pad_ratio=crop_pad_ratio,
        bin_threshold=bin_threshold,
        iou_min=iou_min,
        coverage_min=coverage_min,
        area_ratio_min=area_ratio_min,
        area_ratio_max=area_ratio_max,
        fallback=fallback,
        feather_radius=feather_radius,
        feather_band=feather_band,
    )
