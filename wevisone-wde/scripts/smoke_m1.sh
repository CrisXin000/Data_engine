#!/usr/bin/env bash
# One-shot M0+M1 smoke: init → kg → gold → run → export → status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -e .

python -m wde.cli init-db
python -m wde.cli kg init
python -m wde.cli ingest-gold

RUN_ID="$(python - <<'PY'
from pathlib import Path
from wde.pipeline.runner import run_pipeline
from wde.utils import new_run_id
rid = new_run_id()
stats = run_pipeline(sorted(Path("data/gold/images").glob("*.png")), run_id=rid, mode="mock")
print(stats["run_id"])
PY
)"
echo "RUN_ID=$RUN_ID"
mkdir -p "data/runs/$RUN_ID"
python -m wde.cli export --run-id "$RUN_ID" --out "data/runs/$RUN_ID/export.json"
python -m wde.cli status
echo "OK: data/runs/$RUN_ID/export.json"
