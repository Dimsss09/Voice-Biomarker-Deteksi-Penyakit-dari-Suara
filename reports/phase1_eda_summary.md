# Phase 1 EDA Summary

## Dataset

- Name: UCI Parkinsons
- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
- Raw file: `data/raw/parkinsons.data`
- Shape: 195 rows x 25 columns
- Identifier column: `name`
- Target column: `status`
- Numeric feature columns: 22
- Parsed subjects: 32

## Target Distribution

| Label | Meaning | Count | Percentage |
| --- | --- | ---: | ---: |
| 0 | Healthy | 48 | 24.6% |
| 1 | Parkinson | 147 | 75.4% |

## Data Quality

- Missing values: 0
- Duplicate rows: 0
- Unique recording IDs: 195
- Min recordings per subject: 6
- Max recordings per subject: 7

## Feature Overview

- The UCI file already contains acoustic measurements, so Phase 2 can begin with cleaning, scaling, and feature selection rather than raw-audio extraction.
- Subject IDs can be parsed from `name` values such as `phon_R01_S01_1`, where `S01` is the subject.
- Phase 3 must split by `subject_id` so recordings from the same subject do not leak across train/test.

## Subject Distribution

| Label | Meaning | Subjects | Recordings |
| --- | --- | ---: | ---: |
| 0 | Healthy | 8 | 48 |
| 1 | Parkinson | 24 | 147 |

## Strong Feature Correlations

| Feature A | Feature B | Absolute Correlation |
| --- | --- | ---: |
| `Shimmer:APQ3` | `Shimmer:DDA` | 1.000 |
| `MDVP:RAP` | `Jitter:DDP` | 1.000 |
| `MDVP:Jitter(%)` | `Jitter:DDP` | 0.990 |
| `MDVP:Jitter(%)` | `MDVP:RAP` | 0.990 |
| `MDVP:Shimmer` | `Shimmer:DDA` | 0.988 |
| `MDVP:Shimmer` | `Shimmer:APQ3` | 0.988 |
| `MDVP:Shimmer` | `MDVP:Shimmer(dB)` | 0.987 |
| `MDVP:Shimmer` | `Shimmer:APQ5` | 0.983 |
| `MDVP:Jitter(%)` | `MDVP:PPQ` | 0.974 |
| `MDVP:Shimmer(dB)` | `Shimmer:APQ5` | 0.974 |
| `MDVP:Shimmer(dB)` | `Shimmer:DDA` | 0.963 |
| `MDVP:Shimmer(dB)` | `Shimmer:APQ3` | 0.963 |
| `spread1` | `PPE` | 0.962 |
| `MDVP:Shimmer(dB)` | `MDVP:APQ` | 0.961 |
| `Shimmer:APQ5` | `Shimmer:DDA` | 0.960 |

## Generated Figures

- `reports/class_distribution.png`
- `reports/correlation_heatmap.png`
- `reports/feature_distributions_by_label.png`
