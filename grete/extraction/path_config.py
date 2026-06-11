"""Resolve shared Grete extraction paths from config/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config/config.yaml must contain a YAML mapping/object.")
    return config


def _resolve_from_project(raw_path: str) -> Path:
    p = Path(str(raw_path).strip())
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def get_pipeline_extraction_output_dir() -> Path:
    config = _load_config()
    output_dir = (config.get("pipeline") or {}).get("extraction_output_dir")
    if not output_dir:
        raise ValueError("Missing pipeline.extraction_output_dir in config/config.yaml")
    return _resolve_from_project(str(output_dir))


def get_arxiv_download_dir() -> Path:
    config = _load_config()
    download_dir = (config.get("arxiv") or {}).get("download_dir")
    if not download_dir:
        raise ValueError("Missing arxiv.download_dir in config/config.yaml")
    return _resolve_from_project(str(download_dir))
