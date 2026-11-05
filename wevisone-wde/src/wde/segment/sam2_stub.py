"""SAM2 mask annotation stub (§2.4). Skipped in M1 by default."""

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
) -> MaskResult:
    if backend != "stub":
        raise NotImplementedError(f"segment backend={backend} reserved (SAM2). Use stub.")
    return MaskResult(
        kind="mask",
        path=None,
        source="stub",
        meta={
            "note": "M1 stub — SAM2 not run",
            "box": list(box_xyxy),
            "image": image_path,
        },
    )
