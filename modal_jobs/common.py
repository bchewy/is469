from __future__ import annotations

from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
