"""M0 gold set: synthetic images + annotations (no copyrighted downloads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageDraw
from sqlalchemy.orm import Session

from wde.schema.models import Image, Query
from wde.store.db import get_session, init_db
from wde.utils import file_sha256


def _draw_scene(
    path: Path,
    size: tuple[int, int],
    shapes: list[dict[str, Any]],
    bg: tuple[int, int, int] = (240, 240, 245),
) -> None:
    im = PILImage.new("RGB", size, bg)
    draw = ImageDraw.Draw(im)
    for s in shapes:
        kind = s["kind"]
        color = tuple(s["color"])
        box = s["xyxy"]  # x1,y1,x2,y2
        if kind == "rectangle":
            draw.rectangle(box, fill=color, outline=(20, 20, 20), width=2)
        elif kind == "circle":
            draw.ellipse(box, fill=color, outline=(20, 20, 20), width=2)
        elif kind == "triangle":
            x1, y1, x2, y2 = box
            mid = ((x1 + x2) // 2, y1)
            draw.polygon([mid, (x1, y2), (x2, y2)], fill=color, outline=(20, 20, 20))
        elif kind == "square":
            draw.rectangle(box, fill=color, outline=(20, 20, 20), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def generate_gold(root: Path | None = None) -> dict[str, Any]:
    """Create 5 synthetic gold images covering single/multi/referring cases."""
    from wde.config import project_root

    root = root or project_root()
    img_dir = root / "data" / "gold" / "images"
    ann_dir = root / "data" / "gold" / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[dict[str, Any]] = [
        {
            "name": "single_red_rect",
            "size": (640, 480),
            "shapes": [
                {
                    "label": "red rectangle",
                    "kind": "rectangle",
                    "color": (220, 40, 40),
                    "xyxy": [180, 120, 420, 340],
                }
            ],
            "queries": [
                {"text": "red rectangle", "query_type": "category", "is_negative": False},
            ],
        },
        {
            "name": "multi_shapes",
            "size": (640, 480),
            "shapes": [
                {
                    "label": "red rectangle",
                    "kind": "rectangle",
                    "color": (220, 40, 40),
                    "xyxy": [40, 60, 200, 220],
                },
                {
                    "label": "blue circle",
                    "kind": "circle",
                    "color": (40, 90, 220),
                    "xyxy": [260, 80, 420, 240],
                },
                {
                    "label": "green triangle",
                    "kind": "triangle",
                    "color": (40, 180, 70),
                    "xyxy": [450, 200, 600, 400],
                },
            ],
            "queries": [
                {"text": "red rectangle", "query_type": "category", "is_negative": False},
                {"text": "blue circle", "query_type": "category", "is_negative": False},
                {"text": "green triangle", "query_type": "category", "is_negative": False},
                {
                    "text": "the blue circle near the center",
                    "query_type": "referring",
                    "is_negative": False,
                    "target_label": "blue circle",
                },
            ],
        },
        {
            "name": "referring_left_right",
            "size": (640, 480),
            "shapes": [
                {
                    "label": "yellow square",
                    "kind": "square",
                    "color": (230, 200, 40),
                    "xyxy": [50, 180, 180, 310],
                },
                {
                    "label": "yellow square",
                    "kind": "square",
                    "color": (230, 200, 40),
                    "xyxy": [420, 180, 550, 310],
                },
            ],
            "queries": [
                {"text": "yellow square", "query_type": "category", "is_negative": False},
                {
                    "text": "the yellow square on the left",
                    "query_type": "referring",
                    "is_negative": False,
                    "target_index": 0,
                },
            ],
        },
        {
            "name": "absent_target",
            "size": (640, 480),
            "shapes": [
                {
                    "label": "red rectangle",
                    "kind": "rectangle",
                    "color": (220, 40, 40),
                    "xyxy": [200, 150, 400, 320],
                }
            ],
            "queries": [
                {"text": "red rectangle", "query_type": "category", "is_negative": False},
                {
                    "text": "the purple hexagon that does not exist",
                    "query_type": "referring",
                    "is_negative": True,
                },
            ],
        },
        {
            "name": "dense_small",
            "size": (640, 480),
            "shapes": [
                {
                    "label": "blue circle",
                    "kind": "circle",
                    "color": (40, 90, 220),
                    "xyxy": [30 + i * 100, 40 + (i % 2) * 120, 90 + i * 100, 100 + (i % 2) * 120],
                }
                for i in range(5)
            ],
            "queries": [
                {"text": "blue circle", "query_type": "category", "is_negative": False},
                {
                    "text": "the largest blue circle",
                    "query_type": "referring",
                    "is_negative": False,
                },
            ],
        },
    ]

    index: list[dict[str, Any]] = []
    for sc in scenes:
        img_path = img_dir / f"{sc['name']}.png"
        _draw_scene(img_path, tuple(sc["size"]), sc["shapes"])
        # sidecar labels for mock tagger
        cats = sorted({s["label"] for s in sc["shapes"]})
        (img_path.with_suffix(".labels.json")).write_text(
            json.dumps({"categories": cats}, indent=2), encoding="utf-8"
        )
        ann = {
            "image": sc["name"] + ".png",
            "width": sc["size"][0],
            "height": sc["size"][1],
            "objects": [
                {"label": s["label"], "kind": s["kind"], "xyxy": s["xyxy"]} for s in sc["shapes"]
            ],
            "queries": sc["queries"],
            "note": "Synthetic gold — public domain / generated; no external copyrighted images.",
        }
        ann_path = ann_dir / f"{sc['name']}.json"
        ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({"image": img_path.name, "annotation": ann_path.name})

    (ann_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {"images": len(scenes), "img_dir": str(img_dir), "ann_dir": str(ann_dir)}


def ingest_gold(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure gold images exist, then register images + gold queries in DB."""
    root = Path(cfg["paths"]["root"])
    gen = generate_gold(root)
    engine = init_db(cfg["paths"]["db"])
    session: Session = get_session(engine)
    img_dir = Path(cfg["paths"]["gold_images"])
    ann_dir = Path(cfg["paths"]["gold_annotations"])
    data_version = cfg["data_version"]

    n_img = n_q = 0
    for ann_path in sorted(ann_dir.glob("*.json")):
        if ann_path.name == "index.json":
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        img_path = img_dir / ann["image"]
        if not img_path.exists():
            continue
        existing = session.query(Image).filter_by(path=str(img_path.resolve())).one_or_none()
        if existing:
            img = existing
        else:
            img = Image(
                path=str(img_path.resolve()),
                width=ann["width"],
                height=ann["height"],
                source="gold",
                sha256=file_sha256(img_path),
                meta={"gold_annotation": ann_path.name},
                data_version=data_version,
                run_id="gold-ingest",
            )
            session.add(img)
            session.flush()
            n_img += 1
        for q in ann.get("queries", []):
            # Avoid duplicate gold queries on re-ingest
            dup = (
                session.query(Query)
                .filter_by(image_id=img.id, text=q["text"], source="gold")
                .one_or_none()
            )
            if dup:
                continue
            session.add(
                Query(
                    image_id=img.id,
                    text=q["text"],
                    query_type=q["query_type"],
                    source="gold",
                    is_negative=bool(q.get("is_negative")),
                    meta={k: v for k, v in q.items() if k not in {"text", "query_type", "is_negative"}},
                    data_version=data_version,
                    run_id="gold-ingest",
                )
            )
            n_q += 1
    session.commit()
    session.close()
    return {"generated": gen, "new_images": n_img, "new_queries": n_q}
