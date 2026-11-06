#!/usr/bin/env bash
# M2 smoke: grounded structured referring on 1–2 real images → propose → verify → export + report
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
STAGING="data/gold/images_real_smoke_referring"
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

# --- structured (main) ---
STRUCT_ID="$(python - <<PY
from pathlib import Path
from wde.pipeline.runner import run_pipeline, export_run
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
    referring_generator="structured",
)
out = Path(f"data/runs/{rid}")
out.mkdir(parents=True, exist_ok=True)
(out / "smoke_stats.json").write_text(json.dumps(stats, indent=2))
(out / "smoke_tag.txt").write_text("referring_structured\n")
export_run(rid, out / "export.json")
print(stats["run_id"])
PY
)"
echo "STRUCT_ID=$STRUCT_ID"
python scripts/export_viz.py --export "data/runs/$STRUCT_ID/export.json" --out-dir "data/runs/$STRUCT_ID/viz"

# --- mock referring baseline on same images (real detectors, mock referring) ---
MOCK_ID="$(python - <<PY
from pathlib import Path
from wde.pipeline.runner import run_pipeline, export_run
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
)
out = Path(f"data/runs/{rid}")
out.mkdir(parents=True, exist_ok=True)
(out / "smoke_stats.json").write_text(json.dumps(stats, indent=2))
(out / "smoke_tag.txt").write_text("referring_mock\n")
export_run(rid, out / "export.json")
print(stats["run_id"])
PY
)"
echo "MOCK_ID=$MOCK_ID"

COMPARE_DIR="data/runs/compare_mock_vs_structured_referring"
mkdir -p "$COMPARE_DIR"
echo "$STRUCT_ID" > "$COMPARE_DIR/structured_run_id.txt"
echo "$MOCK_ID" > "$COMPARE_DIR/mock_run_id.txt"

python scripts/write_referring_report.py \
  --export "data/runs/$STRUCT_ID/export.json" \
  --mock-export "data/runs/$MOCK_ID/export.json" \
  --out-dir "$COMPARE_DIR"

# Also copy report into structured run dir
cp "$COMPARE_DIR/referring_queries.json" "data/runs/$STRUCT_ID/"
cp "$COMPARE_DIR/REFERRING.md" "data/runs/$STRUCT_ID/"

python - <<PY
import json
from pathlib import Path
exp = json.loads(Path("data/runs/$STRUCT_ID/export.json").read_text())
refs = [q for q in exp["query"] if q["query_type"] == "referring"]
assert refs, "expected referring queries"
grounded = [q for q in refs if (q.get("meta") or {}).get("mock") is False]
assert grounded, "expected meta.mock == false on structured referring"
neg = [q for q in refs if q.get("is_negative") or (q.get("meta") or {}).get("is_negative_hint")]
assert neg, "expected at least one absent/negative referring"
has_constraints = any((q.get("meta") or {}).get("constraints") for q in grounded)
assert has_constraints, "expected constraints in meta"
print(f"referring={len(refs)} grounded={len(grounded)} negative={len(neg)}")
print("sample:", [q["text"] for q in refs[:6]])
print("OK smoke_referring structured=$STRUCT_ID mock=$MOCK_ID")
PY
