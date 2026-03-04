"""Config loader — reads YAML and returns a plain dict."""
from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg
