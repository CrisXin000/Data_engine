"""Box-prompt binary masks — open-source stand-in for paper SAM2.

Paper uses SAM2. This module does NOT load the official Meta ``sam2`` package
or the full SAM2 family (sam2_s/b/l, video memory, etc.).

Preferred path: Ultralytics **SAM2-Tiny** (``sam2_t.pt``) via ``ultralytics.SAM``,
which *is* a SAM2 checkpoint, run on CPU with a reduced ``imgsz``.

Fallback chain when ``backend=sam2``:
  1. ultralytics SAM2-Tiny
  2. ultralytics MobileSAM (weaker; still prompted segmentation)
  3. ``box_fill`` (fill the prompt box with 1s) — much weaker than SAM2; never silent
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage

from wde.config import project_root
from wde.segment.sam2_stub import MaskResult

STANDS_IN_FOR = "SAM2"
SAM2_T_NAME = "sam2_t.pt"
MOBILE_SAM_NAME = "mobile_sam.pt"
ULTRALYTICS_ASSET = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"

_SESSION: dict[str, Any] | None = None


def _free_ram() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_segment_model() -> None:
    """Drop cached SAM weights so later stages (or the next run) can reuse RAM."""
    global _SESSION
    if _SESSION is not None:
        _SESSION["model"] = None
        _SESSION = None
    _free_ram()


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in box)
    x1i = int(max(0, min(round(x1), width)))
    y1i = int(max(0, min(round(y1), height)))
    x2i = int(max(0, min(round(x2), width)))
    y2i = int(max(0, min(round(y2), height)))
    if x2i <= x1i:
        x2i = min(width, x1i + 1)
    if y2i <= y1i:
        y2i = min(height, y1i + 1)
    return x1i, y1i, x2i, y2i


def _save_mask_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = (arr > 0.5).astype(np.uint8) * 255
    elif arr.max() <= 1:
        arr = (arr > 0).astype(np.uint8) * 255
    PILImage.fromarray(arr, mode="L").save(path)


def _box_fill_array(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = _clip_box(box, width, height)
    mask[y1:y2, x1:x2] = 255
    return mask


def _default_weights_dir() -> Path:
    return project_root() / "weights" / "segment"


def _resolve_weights(explicit: Optional[str], filename: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (project_root() / p).resolve()
        return p
    return _default_weights_dir() / filename


def _maybe_download(dest: Path, filename: str) -> Path:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = ULTRALYTICS_ASSET.format(name=filename)
    try:
        import urllib.request

        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as exc:
        if dest.is_file():
            return dest
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc
    return dest


def _normalize_backend(backend: str) -> str:
    name = (backend or "stub").strip().lower()
    aliases = {
        "sam2_t": "sam2",
        "ultralytics_sam2": "sam2",
        "ultralytics_sam2_t": "sam2",
        "sam": "sam2",
        "mobilesam": "mobile_sam",
        "mobile-sam": "mobile_sam",
        "boxfill": "box_fill",
        "box-fill": "box_fill",
    }
    return aliases.get(name, name)


def _ensure_model(kind: str, weights: Path, device: str):
    """Load or reuse a cached ultralytics SAM model. ``kind`` is sam2|mobile_sam."""
    global _SESSION
    key = (kind, str(weights.resolve()), device)
    if _SESSION is not None and _SESSION.get("key") == key and _SESSION.get("model") is not None:
        return _SESSION["model"]
    release_segment_model()
    from ultralytics import SAM

    model = SAM(str(weights))
    _SESSION = {"key": key, "model": model, "kind": kind, "weights": str(weights), "device": device}
    return model


def _predict_mask(
    model: Any,
    image_path: str,
    box: tuple[int, int, int, int],
    *,
    device: str,
    imgsz: int,
) -> Optional[np.ndarray]:
    results = model.predict(
        image_path,
        bboxes=[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]],
        device=device,
        imgsz=int(imgsz),
        verbose=False,
        retina_masks=True,
    )
    if not results:
        return None
    r = results[0]
    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        return None
    arr = r.masks.data[0].detach().cpu().numpy()
    return arr


def _resize_to(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    if arr.shape[0] == height and arr.shape[1] == width:
        if arr.dtype == np.uint8 and arr.max() > 1:
            return arr
        return (arr > 0.5).astype(np.uint8) * 255
    bin_im = PILImage.fromarray((arr > 0.5).astype(np.uint8) * 255, mode="L")
    return np.array(bin_im.resize((width, height), resample=PILImage.NEAREST))


def _try_ultralytics(
    *,
    kind: str,
    filename: str,
    image_path: str,
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    weights: Optional[str],
    device: str,
    imgsz: int,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    info: dict[str, Any] = {"attempt": kind, "filename": filename}
    try:
        wpath = _resolve_weights(weights if kind == "sam2" else None if weights and kind != "mobile_sam" else weights, filename)
        # If caller passed sam2_t weights but we are on mobile_sam fallback, ignore that path.
        if kind == "mobile_sam":
            # Prefer explicit only when it looks like mobile_sam
            if weights and "mobile" in Path(weights).name.lower():
                wpath = _resolve_weights(weights, filename)
            else:
                wpath = _resolve_weights(None, filename)
        if not wpath.is_file():
            wpath = _maybe_download(wpath, filename)
        info["weights"] = str(wpath)
        model = _ensure_model(kind, wpath, device)
        clipped = _clip_box(box, width, height)
        raw = _predict_mask(model, image_path, clipped, device=device, imgsz=imgsz)
        if raw is None:
            info["error"] = "empty_mask"
            return None, info
        arr = _resize_to(raw, width, height)
        try:
            import ultralytics

            info["package"] = f"ultralytics=={ultralytics.__version__}"
        except Exception:
            info["package"] = "ultralytics"
        info["is_sam2"] = bool(getattr(model, "is_sam2", kind == "sam2"))
        info["ok"] = True
        return arr, info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        release_segment_model()
        return None, info


def segment_instance(
    *,
    image_path: str,
    box_xyxy: tuple[float, float, float, float],
    backend: str = "sam2",
    output_path: Optional[str] = None,
    instance_uid: Optional[str] = None,
    query_text: Optional[str] = None,
    query_id: Optional[int] = None,
    extra_meta: Optional[dict[str, Any]] = None,
    weights: Optional[str] = None,
    device: str = "cpu",
    imgsz: int = 512,
) -> MaskResult:
    """Write a full-image binary mask (foreground=255) for one verified box.

    ``backend``:
      - ``sam2``: ultralytics SAM2-Tiny, then MobileSAM, then box_fill
      - ``mobile_sam``: MobileSAM then box_fill
      - ``box_fill``: rectangle fill only (weaker than SAM2; labelled as such)
    """
    backend = _normalize_backend(backend)
    with PILImage.open(image_path) as im:
        width, height = im.size

    if not output_path:
        uid = instance_uid or "anon"
        output_path = str(
            project_root() / "data" / "runs" / "_anon_masks" / f"{Path(image_path).stem}_{uid}.png"
        )
    out = Path(output_path)

    attempts: list[dict[str, Any]] = []
    arr: Optional[np.ndarray] = None
    used = backend
    model_name = ""
    note = ""
    weaker = False

    if backend == "sam2":
        arr, info = _try_ultralytics(
            kind="sam2",
            filename=SAM2_T_NAME,
            image_path=image_path,
            box=box_xyxy,
            width=width,
            height=height,
            weights=weights,
            device=device,
            imgsz=imgsz,
        )
        attempts.append(info)
        if arr is not None:
            used = "ultralytics_sam2_t"
            model_name = info.get("weights") or f"ultralytics/{SAM2_T_NAME}"
            note = (
                "Ultralytics SAM2-Tiny via ultralytics.SAM on CPU "
                f"(imgsz={imgsz}). This is a SAM2-Tiny checkpoint, not the official "
                "Meta sam2 package or the full SAM2 family."
            )
        else:
            arr, info2 = _try_ultralytics(
                kind="mobile_sam",
                filename=MOBILE_SAM_NAME,
                image_path=image_path,
                box=box_xyxy,
                width=width,
                height=height,
                weights=None,
                device=device,
                imgsz=imgsz,
            )
            attempts.append(info2)
            if arr is not None:
                used = "mobile_sam"
                model_name = info2.get("weights") or f"ultralytics/{MOBILE_SAM_NAME}"
                weaker = True
                note = (
                    "Fell back to Ultralytics MobileSAM (not SAM2). "
                    f"SAM2-Tiny failed: {info.get('error')}"
                )

    elif backend == "mobile_sam":
        arr, info = _try_ultralytics(
            kind="mobile_sam",
            filename=MOBILE_SAM_NAME,
            image_path=image_path,
            box=box_xyxy,
            width=width,
            height=height,
            weights=weights,
            device=device,
            imgsz=imgsz,
        )
        attempts.append(info)
        if arr is not None:
            used = "mobile_sam"
            model_name = info.get("weights") or f"ultralytics/{MOBILE_SAM_NAME}"
            weaker = True
            note = (
                "Ultralytics MobileSAM on CPU — prompted segmentation stand-in for "
                f"paper SAM2 (imgsz={imgsz}); weaker than SAM2."
            )

    elif backend != "box_fill":
        raise NotImplementedError(
            f"segment backend={backend!r} unknown. Use stub|sam2|mobile_sam|box_fill."
        )

    if arr is None:
        arr = _box_fill_array(width, height, box_xyxy)
        if backend == "box_fill":
            used = "box_fill"
            model_name = "box_fill"
            weaker = True
            note = (
                "Weak geometric fallback: every pixel inside the prompt box is 255. "
                "Not a learned segmenter; much weaker than SAM2."
            )
        else:
            used = "box_fill"
            model_name = "box_fill"
            weaker = True
            last_err = attempts[-1].get("error") if attempts else "no attempt"
            note = (
                "Weak geometric fallback (box_fill): SAM/MobileSAM unavailable "
                f"({last_err}). Fill prompt box with 255; much weaker than SAM2."
            )

    _save_mask_png(arr, out)
    fg = int((arr > 127).sum())
    meta: dict[str, Any] = {
        "mock": False,
        "model": model_name,
        "backend": backend,
        "resolved_backend": used,
        "stands_in_for": STANDS_IN_FOR,
        "note": note,
        "weaker_than_sam2": weaker,
        "box": list(box_xyxy),
        "image": image_path,
        "instance_uid": instance_uid,
        "query_text": query_text,
        "query_id": query_id,
        "imgsz": imgsz,
        "device": device,
        "mask_size": [width, height],
        "foreground_pixels": fg,
        "foreground_value": 255,
        "attempts": attempts,
    }
    if extra_meta:
        meta.update(extra_meta)

    return MaskResult(kind="mask", path=str(out.resolve()), source=used, meta=meta)
