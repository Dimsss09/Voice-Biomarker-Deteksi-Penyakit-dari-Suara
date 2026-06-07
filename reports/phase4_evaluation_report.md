# Phase 4 Evaluation Report

## Model

- Model: `xgboost`
- Parameters: `{'classifier__learning_rate': 0.05, 'classifier__max_depth': 2, 'classifier__n_estimators': 100}`
- Decision threshold: `0.979`
- Feature count: 22

## Held-Out Test Set

- Subjects: 7
- Recordings: 43

| Label | Meaning | Recordings | Subjects |
| --- | --- | ---: | ---: |
| 0 | Healthy | 12 | 2 |
| 1 | Parkinson | 31 | 5 |

## Metrics

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.997 |
| Sensitivity / Recall | 0.226 |
| Specificity | 1.000 |
| F1 | 0.368 |
| Accuracy | 0.442 |
| Balanced Accuracy | 0.613 |

## Threshold Comparison

| Threshold | ROC-AUC | Sensitivity | Specificity | F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected_validation_threshold (0.979) | 0.997 | 0.226 | 1.000 | 0.368 | 0.613 |
| default_0_5 (0.500) | 0.997 | 1.000 | 0.583 | 0.925 | 0.792 |

## Confusion Matrix

| | Pred Healthy | Pred Parkinson |
| --- | ---: | ---: |
| Actual Healthy | 12 | 0 |
| Actual Parkinson | 24 | 7 |

## Artifacts

- Confusion matrix: `reports/phase4_confusion_matrix.png`
- ROC curve: `reports/phase4_roc_curve.png`
- Test predictions: `reports/phase4_test_predictions.csv`
- Threshold comparison: `reports/phase4_threshold_comparison.csv`
- Permutation importance: `reports/phase4_permutation_importance.csv`

## Notes

- This is a subject-level held-out evaluation, so recordings from test subjects were not used for model selection.
- The validation-selected threshold is conservative. The threshold comparison is included so Phase 5 can choose a demo threshold intentionally.
- Dataset size is small, especially for healthy subjects, so metrics should be read as educational project evidence rather than clinical evidence.
