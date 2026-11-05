#!/usr/bin/env python3
"""Draw candidate / accept boxes from an export.json onto images → viz/."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

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
}


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

    print("viz dir:", out_dir)


if __name__ == "__main__":
    main()
