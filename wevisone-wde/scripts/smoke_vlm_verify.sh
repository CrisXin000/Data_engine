#!/usr/bin/env bash
# CPU smoke: real detectors + VLM/CLIP semantic verify on 1–2 real images
# Compares against a rule-verify run on the same staging set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
import torch, clip  # noqa: F401
print("deps ok: torch", torch.__version__, "clip ok, cuda=", torch.cuda.is_available())
from pathlib import Path
w = Path("weights/clip/ViT-B-32.pt")
print("CLIP weights:", w, "exists=", w.is_file())
PY

IMG_DIR="data/gold/images_real"
STAGING="data/gold/images_real_smoke_vlm"
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

run_one() {
  local VERIFY="$1"
  local TAG="$2"
  python - <<PY
from pathlib import Path
from wde.pipeline.runner import run_pipeline
from wde.utils import new_run_id
import json
rid = new_run_id()
paths = sorted(Path("$STAGING").glob("*"))
paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
stats = run_pipeline(paths, run_id=rid, mode="real", verify_backend="$VERIFY")
out = Path(f"data/runs/{rid}")
out.mkdir(parents=True, exist_ok=True)
(out / "smoke_stats.json").write_text(json.dumps(stats, indent=2))
(out / "smoke_tag.txt").write_text("$TAG\n")
print(stats["run_id"])
PY
}

echo "=== rule baseline ==="
RULE_ID="$(run_one rule rule)"
echo "RULE_ID=$RULE_ID"
python -m wde.cli export --run-id "$RULE_ID" --out "data/runs/$RULE_ID/export.json"
python scripts/export_viz.py --export "data/runs/$RULE_ID/export.json" --out-dir "data/runs/$RULE_ID/viz"

echo "=== vlm_json (Florence→CLIP fallback) ==="
VLM_ID="$(run_one vlm_json vlm_json)"
echo "VLM_ID=$VLM_ID"
python -m wde.cli export --run-id "$VLM_ID" --out "data/runs/$VLM_ID/export.json"
python scripts/export_viz.py --export "data/runs/$VLM_ID/export.json" --out-dir "data/runs/$VLM_ID/viz"

COMPARE_DIR="data/runs/compare_rule_vs_vlm"
mkdir -p "$COMPARE_DIR"
echo "$RULE_ID" > "$COMPARE_DIR/rule_run_id.txt"
echo "$VLM_ID" > "$COMPARE_DIR/vlm_run_id.txt"

python scripts/write_vlm_comparison.py \
  --rule-id "$RULE_ID" \
  --vlm-id "$VLM_ID" \
  --out-dir "$COMPARE_DIR"

echo "OK smoke_vlm_verify rule=$RULE_ID vlm=$VLM_ID"
