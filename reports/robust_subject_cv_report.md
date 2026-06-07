# Robust Subject-Level Cross-Validation Report

## Setup

- Repeats: 5
- Folds per repeat: 4
- Total folds: 20
- Split grouping: `subject_id`
- Threshold: `0.5` for fold-level classification metrics
- Model template: trained Phase 3 winning pipeline cloned and refit inside each fold

## Metric Stability

| Metric | Mean | Std |
| --- | ---: | ---: |
| roc_auc | 0.802 | 0.123 |
| sensitivity | 0.827 | 0.125 |
| specificity | 0.550 | 0.276 |
| f1 | 0.834 | 0.065 |
| balanced_accuracy | 0.689 | 0.117 |

## Interpretation

- This report checks whether the model family remains useful across multiple subject-level splits.
- It does not replace external validation, because all folds still come from the same UCI dataset.
- Wide standard deviations indicate the dataset is too small for a strong industry claim.
