#!/usr/bin/env python3
"""Write referring-query list + mock-vs-structured comparison from a run export."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mock-export", default=None, type=Path, help="Optional mock-referring export for diff")
    args = ap.parse_args()

    data = json.loads(args.export.read_text(encoding="utf-8"))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    images = {i["id"]: i for i in data.get("image", [])}
    refs = [q for q in data.get("query", []) if q.get("query_type") == "referring"]
    cats = [q for q in data.get("query", []) if q.get("query_type") == "category"]

    by_img: dict[int, list] = defaultdict(list)
    for q in refs:
        by_img[q["image_id"]].append(q)

    listing = []
    for img_id, qs in sorted(by_img.items()):
        img = images.get(img_id, {})
        path = img.get("path", "")
        stem = Path(path).name if path else str(img_id)
        items = []
        for q in qs:
            meta = q.get("meta") or {}
            items.append(
                {
                    "text": q["text"],
                    "is_negative": bool(q.get("is_negative") or meta.get("is_negative_hint")),
                    "base_category": meta.get("base_category"),
                    "backend": meta.get("backend"),
                    "model": meta.get("model"),
                    "mock": meta.get("mock"),
                    "stands_in_for": meta.get("stands_in_for"),
                    "constraints": meta.get("constraints"),
                    "evidence": meta.get("evidence"),
                    "fallback": meta.get("fallback"),
                }
            )
        listing.append({"image": stem, "image_id": img_id, "path": path, "referring": items})

    (out / "referring_queries.json").write_text(
        json.dumps({"run_id": data.get("run_id"), "images": listing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Referring query report (M2)",
        "",
        f"- run_id: `{data.get('run_id')}`",
        f"- category queries: {len(cats)}",
        f"- referring queries: {len(refs)}",
        f"- negative/absent referring: {sum(1 for q in refs if q.get('is_negative') or (q.get('meta') or {}).get('is_negative_hint'))}",
        "",
        "## Per-image referring expressions",
        "",
    ]
    for block in listing:
        lines.append(f"### {block['image']}")
        lines.append("")
        for item in block["referring"]:
            neg = " **[absent/negative]**" if item["is_negative"] else ""
            lines.append(f"- `{item['text']}`{neg}")
            cons = item.get("constraints") or {}
            attrs = cons.get("attributes") or {}
            rels = cons.get("relations") or []
            lines.append(
                f"  - target=`{cons.get('target')}` ref=`{cons.get('reference')}` "
                f"absent=`{cons.get('absent')}` attrs={attrs} relations={rels}"
            )
            lines.append(
                f"  - backend=`{item.get('backend')}` mock=`{item.get('mock')}` "
                f"stands_in_for=`{item.get('stands_in_for')}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Generator is an **open-source grounded stand-in** (YOLO-World boxes + crop color/position).",
            "- **Not** Qwen3-VL. Config name `qwen3_vl` is an alias that falls back to structured.",
            "- Propose/verify still use noun heuristics for referring; CLIP verify degrades to core noun.",
            "",
        ]
    )

    if args.mock_export and args.mock_export.exists():
        mock = json.loads(args.mock_export.read_text(encoding="utf-8"))
        mock_refs = [q for q in mock.get("query", []) if q.get("query_type") == "referring"]
        lines.append("## Diff vs mock templates")
        lines.append("")
        lines.append("| structured (grounded) | mock template |")
        lines.append("|----------------------|---------------|")
        # Pair by image stem order roughly
        mock_by_img: dict[str, list[str]] = defaultdict(list)
        mock_images = {i["id"]: i for i in mock.get("image", [])}
        for q in mock_refs:
            stem = Path(mock_images.get(q["image_id"], {}).get("path", "")).name
            mock_by_img[stem].append(q["text"])
        for block in listing:
            mtexts = mock_by_img.get(block["image"], [])
            for i, item in enumerate(block["referring"]):
                mt = mtexts[i] if i < len(mtexts) else "(n/a)"
                lines.append(f"| `{item['text']}` | `{mt}` |")
        lines.append("")

    (out / "REFERRING.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'referring_queries.json'} and {out / 'REFERRING.md'}")


if __name__ == "__main__":
    main()
