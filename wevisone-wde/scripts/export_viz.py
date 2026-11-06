#!/usr/bin/env python3
"""Draw candidate / accept boxes from an export.json onto images → viz/."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


COLORS = {
    "ExpertA_mock_sam3": (220, 60, 60),
    "ExpertB_mock_dino_ovd": (60, 60, 220),
    "ExpertA_yolo_world": (40, 180, 60),
    "ExpertB_owlv2": (200, 120, 20),
    "ExpertB_yolo_world": (20, 160, 200),
    "ExpertB_yolo_world_fallback": (160, 80, 160),
    "accept": (0, 200, 0),
    "reject": (180, 40, 40),
    "duplicate": (160, 160, 40),
    "uncertain": (120, 80, 200),
}

def _checkerboard(w: int, h: int, tile: int = 12) -> np.ndarray:
    yy, xx = np.indices((h, w))
    field = ((xx // tile) + (yy // tile)) % 2
    light = np.array([230, 230, 230], dtype=np.uint8)
    dark = np.array([170, 170, 170], dtype=np.uint8)
    return np.where(field[..., None] == 0, light, dark)


def _compose_alpha(rgb: np.ndarray, alpha01: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = alpha01[..., None]
    out = rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--out-dir", default=None, type=Path)
    ap.add_argument("--decisions", default="all", choices=["all", "accept", "candidates"])
    args = ap.parse_args()

    data = json.loads(args.export.read_text(encoding="utf-8"))
    out_dir = args.out_dir or (args.export.parent / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = {i["id"]: i for i in data["image"]}
    queries = {q["id"]: q for q in data["query"]}
    cands_by_img: dict[int, list] = defaultdict(list)
    for c in data["candidate"]:
        q = queries.get(c["query_id"])
        if not q:
            continue
        cands_by_img[q["image_id"]].append((c, q))

    ver_by_cand = {v["candidate_id"]: v for v in data.get("verification", [])}
    queries_by_id = queries
    links = {l["id"]: l for l in data.get("instance_link", [])}
    masks_by_img: dict[int, list] = defaultdict(list)
    alphas_by_img: dict[int, list] = defaultdict(list)
    for m in data.get("mask_alpha", []):
        if not m.get("path"):
            continue
        link = links.get(m.get("instance_link_id"))
        if not link:
            continue
        q = queries_by_id.get(link.get("query_id"))
        if not q:
            continue
        kind = m.get("kind")
        if kind == "mask":
            masks_by_img[q["image_id"]].append((m, link, q))
        elif kind == "alpha":
            alphas_by_img[q["image_id"]].append((m, link, q))

    overlay_colors = [
        (0, 200, 80),
        (40, 140, 255),
        (255, 160, 0),
        (200, 60, 200),
        (80, 220, 220),
        (240, 80, 80),
    ]

    for img_id, items in cands_by_img.items():
        meta = images.get(img_id)
        if not meta:
            continue
        path = Path(meta["path"])
        if not path.exists():
            continue
        im = PILImage.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for c, q in items:
            v = ver_by_cand.get(c["id"])
            decision = v["decision"] if v else "cand"
            if args.decisions == "accept" and decision != "accept":
                continue
            color = COLORS.get(c["expert_name"], (255, 140, 0))
            if decision == "accept":
                outline = COLORS["accept"]
            elif decision == "duplicate":
                outline = COLORS["duplicate"]
            elif decision == "reject":
                outline = COLORS["reject"]
            elif decision == "uncertain":
                outline = COLORS["uncertain"]
            else:
                outline = color
            box = [c["x1"], c["y1"], c["x2"], c["y2"]]
            draw.rectangle(box, outline=outline, width=3)
            label = f"{c['expert_name'][:18]} {c['score']:.2f} [{decision}] {q['text'][:24]}"
            ty = max(0, box[1] - 12)
            draw.text((box[0] + 2, ty), label, fill=outline, font=font)

        out = out_dir / f"{path.stem}_boxes.png"
        im.save(out)
        print("wrote", out)

        overlay_items = masks_by_img.get(img_id, [])
        if overlay_items:
            base = im.convert("RGBA")
            tint = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(tint)
            for i, (m, link, q) in enumerate(overlay_items):
                mp = Path(m["path"])
                if not mp.exists():
                    continue
                mk = PILImage.open(mp).convert("L")
                if mk.size != base.size:
                    mk = mk.resize(base.size, resample=PILImage.NEAREST)
                color = overlay_colors[i % len(overlay_colors)]
                color_img = PILImage.new("RGBA", base.size, color + (110,))
                tint.paste(color_img, (0, 0), mask=mk)
                # outline the linked box
                tdraw.rectangle(
                    [link["x1"], link["y1"], link["x2"], link["y2"]],
                    outline=color + (220,),
                    width=2,
                )
                label = f"{q['text'][:28]} {m.get('source') or ''}"
                ty = max(0, link["y1"] - 12)
                tdraw.text((link["x1"] + 2, ty), label, fill=color + (255,), font=font)
            composed = PILImage.alpha_composite(base, tint).convert("RGB")
            mout = out_dir / f"{path.stem}_masks.png"
            composed.save(mout)
            print("wrote", mout)

        alpha_items = alphas_by_img.get(img_id, [])
        if alpha_items:
            rgb = np.array(PILImage.open(path).convert("RGB"))
            h, w = rgb.shape[:2]
            cb = _checkerboard(w, h)
            acc = cb.astype(np.float32)
            mask_lookup = {m["instance_link_id"]: m for m, _l, _q in masks_by_img.get(img_id, [])}
            tiles = []
            for i, (m, link, q) in enumerate(alpha_items):
                ap = Path(m["path"])
                if not ap.exists():
                    continue
                al = PILImage.open(ap).convert("L")
                if al.size != (w, h):
                    al = al.resize((w, h), resample=PILImage.BILINEAR)
                a01 = np.array(al, dtype=np.float32) / 255.0
                acc = rgb.astype(np.float32) * a01[..., None] + acc * (1.0 - a01[..., None])
                # crop compare tile: binary S | alpha | rgb*alpha
                x1, y1, x2, y2 = [int(round(v)) for v in (link["x1"], link["y1"], link["x2"], link["y2"])]
                pad = 8
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                if x2 <= x1 or y2 <= y1:
                    continue
                s_img = None
                sm = mask_lookup.get(m.get("instance_link_id"))
                if sm and sm.get("path") and Path(sm["path"]).exists():
                    s_img = PILImage.open(sm["path"]).convert("L")
                    if s_img.size != (w, h):
                        s_img = s_img.resize((w, h), resample=PILImage.NEAREST)
                crop_s = np.array(s_img.crop((x1, y1, x2, y2))) if s_img is not None else np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                crop_a = np.array(al.crop((x1, y1, x2, y2)))
                crop_rgb = rgb[y1:y2, x1:x2]
                crop_cb = _checkerboard(crop_rgb.shape[1], crop_rgb.shape[0])
                left = np.stack([crop_s, crop_s, crop_s], axis=-1)
                mid = np.stack([crop_a, crop_a, crop_a], axis=-1)
                right = _compose_alpha(crop_rgb, crop_a.astype(np.float32) / 255.0, crop_cb)
                # resize tiles to a common height
                th = 160
                def _rs(arr):
                    im = PILImage.fromarray(arr)
                    nh = th
                    nw = max(1, int(round(im.size[0] * (th / max(im.size[1], 1)))))
                    return np.array(im.resize((nw, nh), resample=PILImage.BILINEAR))
                gap = np.full((th, 6, 3), 40, dtype=np.uint8)
                row = np.concatenate([_rs(left), gap, _rs(mid), gap, _rs(right)], axis=1)
                caption = f"{q['text'][:28]} src={m.get('source')} fb={(m.get('meta') or {}).get('fallback')}"
                tiles.append((row, caption))
            acc_im = PILImage.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
            aout = out_dir / f"{path.stem}_alphas.png"
            acc_im.save(aout)
            print("wrote", aout)
            if tiles:
                # stack a few compare rows
                rows = []
                max_w = max(t[0].shape[1] for t in tiles[:6])
                font = None
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                for row, caption in tiles[:6]:
                    if row.shape[1] < max_w:
                        pad = np.full((row.shape[0], max_w - row.shape[1], 3), 30, dtype=np.uint8)
                        row = np.concatenate([row, pad], axis=1)
                    banner = PILImage.new("RGB", (max_w, 16), (30, 30, 30))
                    d = ImageDraw.Draw(banner)
                    d.text((4, 2), caption, fill=(240, 240, 240), font=font)
                    rows.append(np.array(banner))
                    rows.append(row)
                    rows.append(np.full((8, max_w, 3), 30, dtype=np.uint8))
                montage = np.concatenate(rows, axis=0)
                cout = out_dir / f"{path.stem}_alpha_vs_mask.png"
                PILImage.fromarray(montage).save(cout)
                print("wrote", cout)

    # masks on images that somehow have no candidates (unlikely)
    for img_id, overlay_items in masks_by_img.items():
        if img_id in cands_by_img:
            continue
        meta = images.get(img_id)
        if not meta:
            continue
        path = Path(meta["path"])
        if not path.exists() or not overlay_items:
            continue
        print("note: masks without candidates for", path)

    print("viz dir:", out_dir)


if __name__ == "__main__":
    main()
