"""Preprocessing utilities for the UCI Parkinson tabular feature dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data import add_subject_id, get_feature_columns, load_dataset_config


FEATURE_RENAMES = {
    "MDVP:Fo(Hz)": "f0_mean_hz",
    "MDVP:Fhi(Hz)": "f0_max_hz",
    "MDVP:Flo(Hz)": "f0_min_hz",
    "MDVP:Jitter(%)": "jitter_percent",
    "MDVP:Jitter(Abs)": "jitter_abs",
    "MDVP:RAP": "jitter_rap",
    "MDVP:PPQ": "jitter_ppq",
    "Jitter:DDP": "jitter_ddp",
    "MDVP:Shimmer": "shimmer_local",
    "MDVP:Shimmer(dB)": "shimmer_db",
    "Shimmer:APQ3": "shimmer_apq3",
    "Shimmer:APQ5": "shimmer_apq5",
    "MDVP:APQ": "shimmer_apq",
    "Shimmer:DDA": "shimmer_dda",
    "NHR": "noise_to_harmonics_ratio",
    "HNR": "harmonics_to_noise_ratio",
    "RPDE": "rpde",
    "DFA": "dfa",
    "spread1": "spread1",
    "spread2": "spread2",
    "D2": "d2",
    "PPE": "ppe",
}


@dataclass(frozen=True)
class PreparedFeatures:
    """Container for model-ready features and metadata."""

    frame: pd.DataFrame
    feature_columns: list[str]
    original_feature_columns: list[str]
    rename_map: dict[str, str]


def prepare_uci_features(raw_frame: pd.DataFrame) -> PreparedFeatures:
    """Clean and reshape UCI Parkinson data into a training-ready feature table."""
    config = load_dataset_config()
    frame = add_subject_id(raw_frame)
    original_feature_columns = get_feature_columns(frame)

    frame = frame.rename(
        columns={
            config["id_column"]: "recording_id",
            config["target_column"]: "label",
            **FEATURE_RENAMES,
        }
    )
    feature_columns = [FEATURE_RENAMES.get(column, column) for column in original_feature_columns]

    required_columns = ["recording_id", "subject_id", "label", *feature_columns]
    frame = frame[required_columns].drop_duplicates().reset_index(drop=True)

    frame["label"] = frame["label"].astype(int)
    invalid_labels = sorted(set(frame["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Unexpected labels found: {invalid_labels}")

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_total = int(frame[feature_columns].isna().sum().sum())
    if missing_total:
        raise ValueError(f"Prepared feature table contains {missing_total} missing feature values.")

    return PreparedFeatures(
        frame=frame,
        feature_columns=feature_columns,
        original_feature_columns=original_feature_columns,
        rename_map={column: FEATURE_RENAMES.get(column, column) for column in original_feature_columns},
    )
