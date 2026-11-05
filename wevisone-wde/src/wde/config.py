"""Load YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = project_root()
    cfg["paths"]["root"] = str(root)
    for key in ("db", "images", "gold_images", "gold_annotations", "runs"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            cfg["paths"][key] = str((root / p).resolve())
    return cfg
