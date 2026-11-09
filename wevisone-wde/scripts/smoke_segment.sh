#!/usr/bin/env bash
# CPU smoke: real detectors + rule verify + ultralytics SAM2-Tiny masks on 1–2 real images.
# Does NOT download the official Meta sam2 package. Uses weights/segment/sam2_t.pt (~75MB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import torch
from ultralytics import SAM  # noqa: F401
print("deps ok: torch", torch.__version__, "cuda=", torch.cuda.is_available())
w = Path("weights/segment/sam2_t.pt")
print("SAM2-Tiny weights:", w, "exists=", w.is_file(), "size=", w.stat().st_size if w.is_file() else 0)
if not w.is_file():
    raise SystemExit("missing weights/segment/sam2_t.pt — download sam2_t.pt from ultralytics assets (not Meta sam2)")
PY

IMG_DIR="data/gold/images_real"
STAGING="data/gold/images_real_smoke_segment"
rm -rf "$STAGING"
mkdir -p "$STAGING"
copied=0
for name in bus.jpg dog.jpg zidane.jpg; do
  if [[ -f "$IMG_DIR/$name" ]]; then
    cp "$IMG_DIR/$name" "$STAGING/"
    stem="${name%.*}"
    [[ -f "$IMG_DIR/${stem}.labels.json" ]] && cp "$IMG_DIR/${stem}.labels.json" "$STAGING/"
    copied=$((copied + 1))
  fi
  [[ $copied -ge 2 ]] && break
done
if [[ $copied -lt 1 ]]; then
  echo "No real images under $IMG_DIR" >&2
  exit 1
fi
echo "images: $(ls "$STAGING")"

python -m wde.cli init-db
python -m wde.cli kg init || true

RUN_ID="$(python - <<PY
from pathlib import Path
from wde.pipeline.runner import run_pipeline
from wde.utils import new_run_id
import json
rid = new_run_id()
paths = sorted(Path("$STAGING").glob("*"))
paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
stats = run_pipeline(
    paths,
    run_id=rid,
    mode="real",
    verify_backend="rule",
    referring_generator="mock",
    segment_backend="sam2",
)
out = Path(f"data/runs/{rid}")
out.mkdir(parents=True, exist_ok=True)
(out / "smoke_stats.json").write_text(json.dumps(stats, indent=2))
(out / "smoke_tag.txt").write_text("segment_sam2\n")
print(stats["run_id"])
PY
)"
echo "RUN_ID=$RUN_ID"
python -m wde.cli export --run-id "$RUN_ID" --out "data/runs/$RUN_ID/export.json"
python scripts/export_viz.py --export "data/runs/$RUN_ID/export.json" --out-dir "data/runs/$RUN_ID/viz"

python - <<PY
import json
from collections import Counter
from pathlib import Path
from PIL import Image

rid = "$RUN_ID"
exp = json.loads(Path(f"data/runs/{rid}/export.json").read_text())
masks = [m for m in exp.get("mask_alpha", []) if m.get("kind") == "mask"]
print(f"instance_links={len(exp.get('instance_link', []))} masks={len(masks)}")
assert exp.get("instance_link"), "no accepted instances to segment"
assert masks, "expected mask_alpha rows"
sources = Counter(m.get("source") for m in masks)
print("mask sources", dict(sources))
missing = []
for m in masks:
    p = m.get("path")
    if not p or not Path(p).exists():
        missing.append(p)
        continue
    meta = m.get("meta") or {}
    assert meta.get("stands_in_for") == "SAM2", meta
    assert meta.get("mock") is False, meta
    assert meta.get("instance_uid"), meta
    link = next(l for l in exp["instance_link"] if l["id"] == m["instance_link_id"])
    q = next(qq for qq in exp["query"] if qq["id"] == link["query_id"])
    img = next(i for i in exp["image"] if i["id"] == q["image_id"])
    mk = Image.open(p)
    im = Image.open(img["path"])
    assert mk.size == im.size, (mk.size, im.size, p)
    assert mk.mode in {"L", "1"}, mk.mode
    print("mask", Path(p).name, "source=", m.get("source"), "query=", q["text"][:40], "size=", mk.size, "fg≈", (list(mk.getdata()).count(255) if mk.size[0]*mk.size[1] < 3_000_000 else "skip"))
if missing:
    raise SystemExit(f"missing mask files: {missing}")
# keep a pointer for reports
Path("data/runs/latest_segment_run.txt").write_text(rid + "\n")
print("OK smoke_segment", rid)
print("example mask:", masks[0]["path"])
PY

echo "OK smoke_segment RUN_ID=$RUN_ID"
