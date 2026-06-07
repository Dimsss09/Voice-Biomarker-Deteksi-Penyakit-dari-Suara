"""Utilities for downloading and loading the Parkinson voice dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset.yaml"


def load_dataset_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load dataset metadata from a YAML config file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["dataset"]


def download_dataset(force: bool = False) -> Path:
    """Download the configured UCI Parkinson's dataset if it is missing."""
    config = load_dataset_config()
    raw_path = PROJECT_ROOT / config["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force:
        return raw_path

    response = requests.get(config["source_url"], timeout=30)
    response.raise_for_status()
    raw_path.write_bytes(response.content)
    return raw_path


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw tabular feature dataset as a pandas DataFrame."""
    config = load_dataset_config()
    raw_path = PROJECT_ROOT / config["raw_path"]
    if not raw_path.exists():
        raw_path = download_dataset()
    return pd.read_csv(raw_path)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding identifiers and target."""
    config = load_dataset_config()
    excluded = {config["id_column"], config["target_column"]}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
