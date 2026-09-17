"""Configuration loading with container and source-tree defaults."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def config_path() -> Path:
    explicit = os.environ.get("AGENT_CONFIG")
    if explicit:
        return Path(explicit)
    container = Path("/app/config/config.yaml")
    if container.exists():
        return container
    return Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> dict:
    path = config_path()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
