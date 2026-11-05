#!/usr/bin/env bash
# CPU smoke: real YOLO-World (+ OWLv2) on 1–2 real sample images → export + viz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
import torch
from ultralytics import YOLO  # noqa: F401
print("deps ok: torch", torch.__version__, "cuda=", torch.cuda.is_available())
PY

IMG_DIR="data/gold/images_real"
# Prefer bus + dog (2 images) for faster smoke
STAGING="data/gold/images_real_smoke"
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
stats = run_pipeline(paths, run_id=rid, mode="real")
Path("data/runs").mkdir(parents=True, exist_ok=True)
Path(f"data/runs/{rid}").mkdir(parents=True, exist_ok=True)
Path(f"data/runs/{rid}/smoke_stats.json").write_text(json.dumps(stats, indent=2))
print(stats["run_id"])
PY
)"
echo "RUN_ID=$RUN_ID"
mkdir -p "data/runs/$RUN_ID"
python -m wde.cli export --run-id "$RUN_ID" --out "data/runs/$RUN_ID/export.json"
python scripts/export_viz.py --export "data/runs/$RUN_ID/export.json" --out-dir "data/runs/$RUN_ID/viz"

python - <<PY
import json
from pathlib import Path
from collections import Counter
exp = json.loads(Path("data/runs/$RUN_ID/export.json").read_text())
cands = exp["candidate"]
assert cands, "no candidates"
real = [c for c in cands if isinstance(c.get("raw_output"), dict) and c["raw_output"].get("mock") is False]
print(f"candidates={len(cands)} real_mock_false={len(real)}")
assert real, "expected raw_output.mock == false on real experts"
dec = Counter(v["decision"] for v in exp["verification"])
print("decisions", dict(dec))
print("experts", sorted({c["expert_name"] for c in cands}))
print("OK smoke_real_detectors", "$RUN_ID")
PY
