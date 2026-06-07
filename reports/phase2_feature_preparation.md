# Phase 2 Feature Preparation

## Inputs

- Dataset: UCI Parkinsons
- Raw file: `data/raw/parkinsons.data`
- Source format: tabular acoustic features from UCI, so no raw-audio resampling or silence trimming was needed in this phase.

## Outputs

- `data/features/parkinsons_features.csv`
- `data/features/feature_schema.json`
- `data/features/feature_columns.txt`
- `reports/phase2_feature_describe.csv`

## Prepared Table

- Rows: 195
- Columns: 25
- Numeric features: 22
- Subjects: 32
- Recording IDs: 195
- Missing feature values: 0

## Label Distribution

| Label | Meaning | Recordings | Subjects |
| --- | --- | ---: | ---: |
| 0 | Healthy | 48 | 8 |
| 1 | Parkinson | 147 | 24 |

## Preprocessing Decisions

- Renamed raw UCI columns into Python-friendly feature names while preserving a mapping in `feature_schema.json`.
- Added `subject_id` from recording names such as `phon_R01_S01_1`.
- Kept one row per recording and retained `recording_id`, `subject_id`, and `label` for leakage-safe splitting later.
- Standardization will be fitted inside the Phase 3 training pipeline on the training split only. This intentionally avoids fitting `StandardScaler` on the full dataset.
