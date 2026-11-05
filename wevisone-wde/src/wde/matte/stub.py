"""Matting stub (§2.6). Skipped in M1 by default."""

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
) -> AlphaResult:
    if backend != "stub":
        raise NotImplementedError(
            f"matte backend={backend} reserved (BiRefNet/MatAnyone2). Use stub."
        )
    return AlphaResult(
        kind="alpha",
        path=None,
        source="stub",
        meta={
            "note": "M1 stub — matting not run",
            "box": list(box_xyxy),
            "image": image_path,
        },
    )
