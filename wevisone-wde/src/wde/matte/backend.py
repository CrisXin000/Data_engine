"""Continuous alpha matting — simplified open-source stand-in for paper §2.6.

Paper: on an instance mask S, estimate α^B (BiRefNet) and α^M (MatAnyone2) on a
crop, then keep a prediction only if it passes quality gates; otherwise fall
back to S. This module does **not** run MatAnyone2 (no video-style memory /
propagation) and does **not** train the paper's dedicated matting annotator.

CPU MVP:
  1. Prefer ONNX **BiRefNet-General-Lite** (Swin-Tiny backbone,
     ``BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx``) on a padded crop.
     That is a real BiRefNet-family weight, but the *lite/general* checkpoint,
     not BiRefNet-matting and not paired with MatAnyone2.
  2. Gate vs S with a configurable subset of paper-style checks: IoU(B,S),
     coverage (|B∩S|/|S|), area ratio (|B|/|S|).
  3. On model or gate failure: soften S (distance-transform / Gaussian feather)
     or copy S, and record ``alpha_source`` / ``fallback_reason``.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter

from wde.config import project_root
from wde.matte.stub import AlphaResult

STANDS_IN_FOR = "BiRefNet/MatAnyone2"
ONNX_NAME = "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
ONNX_URLS = (
    f"https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/{ONNX_NAME}",
    f"https://github.com/danielgatis/rembg/releases/download/v0.0.0/{ONNX_NAME}",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_ONNX_SIZE = 1024

_SESSION: dict[str, Any] | None = None


def _free_ram() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_matte_model() -> None:
    """Drop cached ONNX session so later stages (or the next run) can reuse RAM."""
    global _SESSION
    if _SESSION is not None:
        _SESSION["session"] = None
        _SESSION = None
    _free_ram()


def _normalize_backend(backend: str) -> str:
    name = (backend or "stub").strip().lower()
    aliases = {
        "birefnet": "birefnet_lite",
        "birefnet-lite": "birefnet_lite",
        "birefnet_general_lite": "birefnet_lite",
        "birefnet-general-lite": "birefnet_lite",
        # Config reserved name: run the lite stand-in; MatAnyone2 is still NOT run.
        "birefnet_matanyone": "birefnet_lite",
        "birefnet+matanyone": "birefnet_lite",
        "birefnet_matanyone2": "birefnet_lite",
        "soften": "feather",
        "gaussian": "feather",
        "copy": "copy_s",
        "copy_s": "copy_s",
        "identity": "copy_s",
    }
    return aliases.get(name, name)


def _default_weights_dir() -> Path:
    return project_root() / "weights" / "matte"


def _resolve_weights(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (project_root() / p).resolve()
        return p
    return _default_weights_dir() / ONNX_NAME


def _maybe_download(dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 50_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in ONNX_URLS:
        try:
            import urllib.request

            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            if dest.is_file() and dest.stat().st_size > 50_000_000:
                return dest
            last_err = RuntimeError(f"downloaded file too small: {dest} from {url}")
        except Exception as exc:
            last_err = exc
            continue
    if dest.is_file() and dest.stat().st_size > 50_000_000:
        return dest
    raise RuntimeError(f"failed to download {ONNX_NAME}: {last_err}")


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


def _padded_crop_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    pad_ratio: float,
    mask: Optional[np.ndarray] = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in box)
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if xs.size:
            x1 = min(x1, float(xs.min()))
            x2 = max(x2, float(xs.max() + 1))
            y1 = min(y1, float(ys.min()))
            y2 = max(y2, float(ys.max() + 1))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * float(pad_ratio)
    py = bh * float(pad_ratio)
    return _clip_box((x1 - px, y1 - py, x2 + px, y2 + py), width, height)


def _box_fill_mask(width: int, height: int, box: tuple[float, float, float, float]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = _clip_box(box, width, height)
    mask[y1:y2, x1:x2] = 255
    return mask


def _load_binary_s(
    mask_path: Optional[str],
    width: int,
    height: int,
    box: tuple[float, float, float, float],
) -> tuple[np.ndarray, str]:
    """Return uint8 {0,255} S of shape (H, W) and a note about its origin."""
    if mask_path:
        p = Path(mask_path)
        if p.is_file():
            with PILImage.open(p) as im:
                arr = np.array(im.convert("L"))
            if arr.shape[0] != height or arr.shape[1] != width:
                arr = np.array(
                    PILImage.fromarray(arr, mode="L").resize((width, height), resample=PILImage.NEAREST)
                )
            return (arr > 127).astype(np.uint8) * 255, "sam_mask"
    return _box_fill_mask(width, height, box), "box_fill_missing_mask"


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    out = np.empty_like(logits, dtype=np.float32)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    ez = np.exp(logits[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _feather_s(s_bin: np.ndarray, *, radius: float, band: float) -> np.ndarray:
    """Continuous alpha from binary S: EDT ramp if scipy exists, else Gaussian blur."""
    binary = (s_bin > 127).astype(np.uint8)
    try:
        from scipy.ndimage import distance_transform_edt

        dist_in = distance_transform_edt(binary)
        dist_out = distance_transform_edt(1 - binary)
        half = max(float(band), 1.0)
        alpha = np.clip((dist_in - dist_out + half) / (2.0 * half), 0.0, 1.0).astype(np.float32)
        return alpha
    except Exception:
        pass
    im = PILImage.fromarray(binary * 255, mode="L")
    blurred = im.filter(ImageFilter.GaussianBlur(radius=max(float(radius), 0.5)))
    return np.array(blurred, dtype=np.float32) / 255.0


def _copy_s(s_bin: np.ndarray) -> np.ndarray:
    return (s_bin > 127).astype(np.float32)


def _metrics(alpha: np.ndarray, s_bin: np.ndarray, thresh: float) -> dict[str, float]:
    a = alpha >= float(thresh)
    s = s_bin > 127
    inter = float(np.logical_and(a, s).sum())
    union = float(np.logical_or(a, s).sum())
    s_area = float(s.sum())
    a_area = float(a.sum())
    iou = inter / union if union > 0 else 0.0
    coverage = inter / s_area if s_area > 0 else 0.0
    area_ratio = a_area / s_area if s_area > 0 else 0.0
    return {
        "iou_vs_s": iou,
        "coverage_vs_s": coverage,
        "area_ratio_vs_s": area_ratio,
        "alpha_fg_pixels": a_area,
        "s_fg_pixels": s_area,
    }


def _gates_ok(m: dict[str, float], iou_min: float, coverage_min: float, area_min: float, area_max: float) -> tuple[bool, dict[str, bool]]:
    checks = {
        "iou": m["iou_vs_s"] >= float(iou_min),
        "coverage": m["coverage_vs_s"] >= float(coverage_min),
        "area_ratio": float(area_min) <= m["area_ratio_vs_s"] <= float(area_max),
    }
    return all(checks.values()), checks


def _save_alpha_png(alpha01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.round(alpha01 * 255.0), 0, 255).astype(np.uint8)
    PILImage.fromarray(arr, mode="L").save(path)


def _ensure_onnx(weights: Path) -> tuple[Any, str]:
    global _SESSION
    key = str(weights.resolve())
    if _SESSION is not None and _SESSION.get("key") == key and _SESSION.get("session") is not None:
        return _SESSION["session"], _SESSION["input_name"]
    release_matte_model()
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(weights), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    _SESSION = {"key": key, "session": sess, "input_name": input_name}
    return sess, input_name


def _preprocess(crop_rgb: PILImage.Image, size: int) -> np.ndarray:
    resized = crop_rgb.convert("RGB").resize((size, size), resample=PILImage.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return chw


def _infer_birefnet_crop(
    crop_rgb: PILImage.Image,
    *,
    weights: Path,
    size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"weights": str(weights), "imgsz": size}
    sess, input_name = _ensure_onnx(weights)
    blob = _preprocess(crop_rgb, size)
    outs = sess.run(None, {input_name: blob})
    raw = np.asarray(outs[0], dtype=np.float32)
    # expected (1, 1, H, W) logits
    while raw.ndim > 2:
        raw = raw[0]
    alpha = _stable_sigmoid(raw)
    info["input_name"] = input_name
    info["logit_min"] = float(np.min(raw))
    info["logit_max"] = float(np.max(raw))
    return alpha, info


def matte_instance(
    *,
    image_path: str,
    box_xyxy: tuple[float, float, float, float],
    backend: str = "birefnet_lite",
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
    """Write a full-image continuous alpha PNG (0–255) for one verified instance."""
    backend = _normalize_backend(backend)
    with PILImage.open(image_path) as im:
        rgb = im.convert("RGB")
        width, height = rgb.size

    s_u8, s_origin = _load_binary_s(mask_path, width, height, box_xyxy)

    if not output_path:
        uid = instance_uid or "anon"
        output_path = str(
            project_root() / "data" / "runs" / "_anon_alphas" / f"{Path(image_path).stem}_{uid}.png"
        )
    out = Path(output_path)

    attempts: list[dict[str, Any]] = []
    pred_alpha: Optional[np.ndarray] = None
    pred_metrics: dict[str, float] = {}
    gate_detail: dict[str, bool] = {}
    used = backend
    model_name = ""
    note = ""
    fallback_used = False
    fallback_reason: Optional[str] = None
    alpha_source = backend

    want_model = backend == "birefnet_lite"
    if backend not in {"birefnet_lite", "feather", "copy_s"}:
        raise NotImplementedError(
            f"matte backend={backend!r} unknown. "
            "Use stub|birefnet_lite|birefnet|feather|copy_s "
            "(birefnet_matanyone is an alias of birefnet_lite; MatAnyone2 is not run)."
        )

    if want_model:
        info: dict[str, Any] = {"attempt": "birefnet_lite"}
        try:
            wpath = _resolve_weights(weights)
            if not wpath.is_file() or wpath.stat().st_size < 50_000_000:
                wpath = _maybe_download(wpath)
            info["weights"] = str(wpath)
            size = int(imgsz) if int(imgsz) > 0 else DEFAULT_ONNX_SIZE
            # Official lite ONNX is exported at 1024x1024; other sizes often fail.
            if size != DEFAULT_ONNX_SIZE:
                info["imgsz_note"] = (
                    f"requested imgsz={size}; official lite ONNX expects {DEFAULT_ONNX_SIZE}. "
                    "Will try the requested size and fall back if the session rejects it."
                )
            crop_box = _padded_crop_box(box_xyxy, width, height, crop_pad_ratio, mask=(s_u8 > 127))
            info["crop"] = list(crop_box)
            cx1, cy1, cx2, cy2 = crop_box
            crop = rgb.crop((cx1, cy1, cx2, cy2))
            raw_a, inf2 = _infer_birefnet_crop(crop, weights=wpath, size=size)
            info.update(inf2)
            crop_a = PILImage.fromarray(
                np.clip(np.round(raw_a * 255.0), 0, 255).astype(np.uint8), mode="L"
            ).resize((cx2 - cx1, cy2 - cy1), resample=PILImage.BILINEAR)
            full = np.zeros((height, width), dtype=np.float32)
            full[cy1:cy2, cx1:cx2] = np.array(crop_a, dtype=np.float32) / 255.0
            pred_alpha = full
            pred_metrics = _metrics(pred_alpha, s_u8, bin_threshold)
            ok, gate_detail = _gates_ok(
                pred_metrics, iou_min, coverage_min, area_ratio_min, area_ratio_max
            )
            info["metrics"] = pred_metrics
            info["gates"] = gate_detail
            info["ok"] = bool(ok)
            model_name = str(wpath)
            if ok:
                used = "birefnet_lite"
                alpha_source = "birefnet_lite"
                note = (
                    "ONNX BiRefNet-General-Lite (Swin-Tiny) on a padded instance crop. "
                    "This is a real BiRefNet-family checkpoint and an open-source stand-in "
                    f"for paper {STANDS_IN_FOR}. It is NOT BiRefNet-matting, NOT MatAnyone2, "
                    "and NOT the paper-trained matting annotator; there is no α^M branch."
                )
            else:
                failed = [k for k, v in gate_detail.items() if not v]
                fallback_reason = (
                    "gates_failed:"
                    + ",".join(failed)
                    + f" (iou={pred_metrics['iou_vs_s']:.3f},"
                    + f" coverage={pred_metrics['coverage_vs_s']:.3f},"
                    + f" area_ratio={pred_metrics['area_ratio_vs_s']:.3f})"
                )
                pred_alpha = None
            attempts.append(info)
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            info["ok"] = False
            attempts.append(info)
            fallback_reason = f"model_failed: {type(exc).__name__}: {exc}"
            try:
                release_matte_model()
            except Exception:
                pass

    if pred_alpha is None:
        fallback_used = True
        fb = (fallback or "feather").strip().lower()
        if backend == "copy_s" or fb in {"copy", "copy_s", "identity", "s"}:
            pred_alpha = _copy_s(s_u8)
            used = "copy_s"
            alpha_source = "copy_s"
            if not fallback_reason:
                fallback_reason = "explicit_copy_s" if backend == "copy_s" else "fallback_copy_s"
            note = (
                "Fallback alpha is a copy of the binary instance mask S "
                f"(origin={s_origin}). No neural matting applied. "
                f"Reason: {fallback_reason}"
            )
        else:
            pred_alpha = _feather_s(s_u8, radius=feather_radius, band=feather_band)
            used = "feather_s"
            alpha_source = "feather_s"
            if not fallback_reason:
                fallback_reason = "explicit_feather" if backend == "feather" else "fallback_feather_s"
            note = (
                "Fallback alpha is a softened copy of binary mask S "
                f"(distance-transform ramp or Gaussian; origin={s_origin}). "
                f"Not BiRefNet and not MatAnyone2. Reason: {fallback_reason}"
            )
        pred_metrics = _metrics(pred_alpha, s_u8, bin_threshold)
        if backend == "feather":
            fallback_used = True
            model_name = model_name or "feather_s"
        elif backend == "copy_s":
            model_name = model_name or "copy_s"

    if want_model and not fallback_used:
        model_name = model_name or ONNX_NAME

    _save_alpha_png(pred_alpha, out)
    final_metrics = pred_metrics or _metrics(pred_alpha, s_u8, bin_threshold)

    meta: dict[str, Any] = {
        "mock": False,
        "model": model_name or used,
        "backend": backend,
        "resolved_backend": used,
        "stands_in_for": STANDS_IN_FOR,
        "note": note,
        "paper_note": (
            "Paper §2.6 estimates α^B (BiRefNet) and α^M (MatAnyone2) on the crop "
            "and keeps a prediction only if it passes quality gates; else fall back to S. "
            "This MVP runs a single BiRefNet-General-Lite ONNX (α^B stand-in) plus gates "
            "against S. MatAnyone2 is not implemented; no paper matting annotator is trained."
        ),
        "fallback": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "alpha_source": alpha_source,
        "s_origin": s_origin,
        "mask_path": mask_path,
        "iou_vs_s": final_metrics.get("iou_vs_s"),
        "coverage_vs_s": final_metrics.get("coverage_vs_s"),
        "area_ratio_vs_s": final_metrics.get("area_ratio_vs_s"),
        "gates": {
            "iou_min": iou_min,
            "coverage_min": coverage_min,
            "area_ratio_min": area_ratio_min,
            "area_ratio_max": area_ratio_max,
            "bin_threshold": bin_threshold,
            "passed": gate_detail,
        },
        "box": list(box_xyxy),
        "image": image_path,
        "instance_uid": instance_uid,
        "query_text": query_text,
        "query_id": query_id,
        "imgsz": imgsz,
        "device": device,
        "crop_pad_ratio": crop_pad_ratio,
        "alpha_size": [width, height],
        "mean_alpha": float(pred_alpha.mean()) if pred_alpha is not None else None,
        "attempts": attempts,
        "weaker_than_paper": True,
    }
    if extra_meta:
        meta.update(extra_meta)

    return AlphaResult(kind="alpha", path=str(out.resolve()), source=used, meta=meta)
