"""Load the experiment configuration (config.yaml at the project root)."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)
