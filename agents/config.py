"""Shared YAML config loader - used by agents/, env/, loop/, eval/, demo.py."""
import functools
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
