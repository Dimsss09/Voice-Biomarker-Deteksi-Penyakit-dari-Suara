"""Create the Phase 2 model-ready feature table from the UCI Parkinson dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT, load_dataset_config, load_raw_dataset
from src.preprocess import prepare_uci_features


FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _write_feature_schema(
    feature_columns: list[str],
    original_feature_columns: list[str],
    rename_map: dict[str, str],
) -> Path:
    schema_path = FEATURE_DIR / "feature_schema.json"
    payload = {
        "identifier_columns": ["recording_id", "subject_id"],
        "target_column": "label",
        "feature_columns": feature_columns,
        "original_feature_columns": original_feature_columns,
        "rename_map": rename_map,
        "scaling": {
            "method": "StandardScaler",
            "fit_policy": "Fit on training split only in Phase 3 to avoid data leakage.",
        },
    }
    schema_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return schema_path


def _write_feature_report(prepared_frame: pd.DataFrame, feature_columns: list[str]) -> Path:
    config = load_dataset_config()
    report_path = REPORTS_DIR / "phase2_feature_preparation.md"
    class_counts = prepared_frame["label"].value_counts().sort_index()
    subject_counts = prepared_frame.groupby("label")["subject_id"].nunique().sort_index()
    descriptive = prepared_frame[feature_columns].describe().transpose()
    descriptive.to_csv(REPORTS_DIR / "phase2_feature_describe.csv")

    lines = [
        "# Phase 2 Feature Preparation",
        "",
        "## Inputs",
        "",
        f"- Dataset: {config['name']}",
        f"- Raw file: `{config['raw_path']}`",
        "- Source format: tabular acoustic features from UCI, so no raw-audio resampling or silence trimming was needed in this phase.",
        "",
        "## Outputs",
        "",
        "- `data/features/parkinsons_features.csv`",
        "- `data/features/feature_schema.json`",
        "- `data/features/feature_columns.txt`",
        "- `reports/phase2_feature_describe.csv`",
        "",
        "## Prepared Table",
        "",
        f"- Rows: {prepared_frame.shape[0]}",
        f"- Columns: {prepared_frame.shape[1]}",
        f"- Numeric features: {len(feature_columns)}",
        f"- Subjects: {prepared_frame['subject_id'].nunique()}",
        f"- Recording IDs: {prepared_frame['recording_id'].nunique()}",
        f"- Missing feature values: {int(prepared_frame[feature_columns].isna().sum().sum())}",
        "",
        "## Label Distribution",
        "",
        "| Label | Meaning | Recordings | Subjects |",
        "| --- | --- | ---: | ---: |",
        f"| 0 | Healthy | {int(class_counts.get(0, 0))} | {int(subject_counts.get(0, 0))} |",
        f"| 1 | Parkinson | {int(class_counts.get(1, 0))} | {int(subject_counts.get(1, 0))} |",
        "",
        "## Preprocessing Decisions",
        "",
        "- Renamed raw UCI columns into Python-friendly feature names while preserving a mapping in `feature_schema.json`.",
        "- Added `subject_id` from recording names such as `phon_R01_S01_1`.",
        "- Kept one row per recording and retained `recording_id`, `subject_id`, and `label` for leakage-safe splitting later.",
        "- Standardization will be fitted inside the Phase 3 training pipeline on the training split only. This intentionally avoids fitting `StandardScaler` on the full dataset.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prepare_uci_features(load_raw_dataset())

    feature_path = FEATURE_DIR / "parkinsons_features.csv"
    prepared.frame.to_csv(feature_path, index=False)

    columns_path = FEATURE_DIR / "feature_columns.txt"
    columns_path.write_text("\n".join(prepared.feature_columns) + "\n", encoding="utf-8")

    schema_path = _write_feature_schema(
        feature_columns=prepared.feature_columns,
        original_feature_columns=prepared.original_feature_columns,
        rename_map=prepared.rename_map,
    )
    report_path = _write_feature_report(prepared.frame, prepared.feature_columns)

    print(f"Feature table written to: {feature_path}")
    print(f"Feature schema written to: {schema_path}")
    print(f"Phase 2 report written to: {report_path}")
    print(f"Shape: {prepared.frame.shape[0]} rows x {prepared.frame.shape[1]} columns")


if __name__ == "__main__":
    main()
