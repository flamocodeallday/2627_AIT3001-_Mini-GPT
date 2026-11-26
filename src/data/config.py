"""Configuration loader for the data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_data_config(path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    """Load YAML configuration for corpus cleaning, splitting, and tokenization."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file {config_path}, got {type(data).__name__}")
    return data
