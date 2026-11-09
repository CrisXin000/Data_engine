#!/usr/bin/env bash
# CPU smoke: real detectors + rule verify + SAM2-Tiny masks + BiRefNet-lite alpha.
# Does NOT run MatAnyone2 or a paper-trained matting annotator.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import importlib.util
import torch
from ultralytics import SAM  # noqa: F401

print("deps ok: torch", torch.__version__, "cuda=", torch.cuda.is_available())
if importlib.util.find_spec("onnxruntime") is None:
    raise SystemExit("missing onnxruntime — pip install 'onnxruntime>=1.17' (see pyproject extra [matte])")
w = Path("weights/segment/sam2_t.pt")
print("SAM2-Tiny weights:", w, "exists=", w.is_file(), "size=", w.stat().st_size if w.is_file() else 0)
if not w.is_file():
    raise SystemExit("missing weights/segment/sam2_t.pt — needed because matting gates against S")
print("onnxruntime present")
PY

IMG_DIR="data/gold/images_real"
STAGING="data/gold/images_real_smoke_matte"
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
    matte_backend="birefnet_lite",
)
out = Path(f"data/runs/{rid}")
out.mkdir(parents=True, exist_ok=True)
(out / "smoke_stats.json").write_text(json.dumps(stats, indent=2))
(out / "smoke_tag.txt").write_text("matte_birefnet_lite\n")
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
alphas = [m for m in exp.get("mask_alpha", []) if m.get("kind") == "alpha"]
print(f"instance_links={len(exp.get('instance_link', []))} masks={len(masks)} alphas={len(alphas)}")
assert exp.get("instance_link"), "no accepted instances"
assert masks, "expected binary masks (run with --segment sam2)"
assert alphas, "expected alpha rows"
sources = Counter(m.get("source") for m in alphas)
fb = Counter(bool((m.get("meta") or {}).get("fallback")) for m in alphas)
print("alpha sources", dict(sources))
print("fallback counts", {str(k): v for k, v in fb.items()})
missing = []
for m in alphas:
    p = m.get("path")
    if not p or not Path(p).exists():
        missing.append(p)
        continue
    meta = m.get("meta") or {}
    assert meta.get("stands_in_for") == "BiRefNet/MatAnyone2", meta
    assert meta.get("mock") is False, meta
    assert meta.get("alpha_source"), meta
    assert "fallback" in meta, meta
    link = next(l for l in exp["instance_link"] if l["id"] == m["instance_link_id"])
    q = next(qq for qq in exp["query"] if qq["id"] == link["query_id"])
    img = next(i for i in exp["image"] if i["id"] == q["image_id"])
    al = Image.open(p)
    im = Image.open(img["path"])
    assert al.size == im.size, (al.size, im.size, p)
    assert al.mode in {"L", "1"}, al.mode
    print(
        "alpha", Path(p).name,
        "source=", m.get("source"),
        "fallback=", meta.get("fallback"),
        "reason=", (meta.get("fallback_reason") or "")[:80],
        "iou=", None if meta.get("iou_vs_s") is None else round(float(meta["iou_vs_s"]), 3),
        "query=", q["text"][:36],
        "size=", al.size,
    )
if missing:
    raise SystemExit(f"missing alpha files: {missing}")
Path("data/runs/latest_matte_run.txt").write_text(rid + "\n")
print("OK smoke_matte", rid)
print("example alpha:", alphas[0]["path"])
PY

echo "OK smoke_matte RUN_ID=$RUN_ID"
