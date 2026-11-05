#!/usr/bin/env bash
# Install CPU-only torch + ultralytics (+ transformers/OWLv2) into project .venv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
# CPU torch first (do NOT install CUDA wheels on this box)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[real]"
# CLIP dependency used by YOLO-World open-vocab head
pip install "git+https://github.com/ultralytics/CLIP.git" ftfy || true

mkdir -p data/models
# Pre-download YOLO-World weights into data/models
python - <<'PY'
from pathlib import Path
from ultralytics import YOLO
from wde.config import project_root
root = project_root()
out = root / "data" / "models" / "yolov8s-worldv2.pt"
if not out.exists():
    # YOLO downloads to cwd; move afterwards
    m = YOLO("yolov8s-worldv2.pt")
    src = Path("yolov8s-worldv2.pt")
    if src.exists() and not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        src.rename(out)
    print("weights ->", out, "exists=", out.exists())
else:
    print("weights already present:", out)
# optional: warm HF cache for OWLv2 (may be slow / large)
try:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
    Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
    print("OWLv2 cache warmed")
except Exception as e:
    print("OWLv2 warm skipped:", type(e).__name__, e)
PY

# Convenience symlink at project root for ultralytics default lookup
ln -sfn data/models/yolov8s-worldv2.pt yolov8s-worldv2.pt

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda=", torch.cuda.is_available())
from ultralytics import YOLO
print("ultralytics OK")
import transformers
print("transformers", transformers.__version__)
PY
echo "setup_real_detectors: done"
