# Phase 4 Evaluation Report

## Model

- Model: `xgboost`
- Parameters: `{'classifier__learning_rate': 0.05, 'classifier__max_depth': 2, 'classifier__n_estimators': 100}`
- Decision threshold: `0.759`
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
| Sensitivity / Recall | 1.000 |
| Specificity | 0.667 |
| F1 | 0.939 |
| Accuracy | 0.907 |
| Balanced Accuracy | 0.833 |
| Brier Score | 0.083 |

## Bootstrap 95% CI

| Metric | Mean | 95% CI Low | 95% CI High |
| --- | ---: | ---: | ---: |
| roc_auc | 0.997 | 0.986 | 1.000 |
| sensitivity | 1.000 | 1.000 | 1.000 |
| specificity | 0.669 | 0.333 | 1.000 |
| f1 | 0.928 | 0.750 | 1.000 |
| balanced_accuracy | 0.834 | 0.667 | 1.000 |

## Threshold Comparison

| Threshold | ROC-AUC | Sensitivity | Specificity | F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| screening_min_sensitivity_0_90 (0.759) | 0.997 | 1.000 | 0.667 | 0.939 | 0.833 |
| youden_j (0.979) | 0.997 | 0.226 | 1.000 | 0.368 | 0.613 |
| default_0_5 (0.500) | 0.997 | 1.000 | 0.583 | 0.925 | 0.792 |

## Confusion Matrix

| | Pred Healthy | Pred Parkinson |
| --- | ---: | ---: |
| Actual Healthy | 8 | 4 |
| Actual Parkinson | 0 | 31 |

## Artifacts

- Confusion matrix: `reports/phase4_confusion_matrix.png`
- ROC curve: `reports/phase4_roc_curve.png`
- Calibration curve: `reports/phase4_calibration_curve.png`
- Test predictions: `reports/phase4_test_predictions.csv`
- Threshold comparison: `reports/phase4_threshold_comparison.csv`
- Bootstrap CI samples: `reports/phase4_bootstrap_ci.csv`
- Permutation importance: `reports/phase4_permutation_importance.csv`

## Notes

- This is a subject-level held-out evaluation, so recordings from test subjects were not used for model selection.
- The default operating threshold is selected from validation data for screening sensitivity, not optimized on the test set.
- Dataset size is small, especially for healthy subjects, so metrics should be read as educational project evidence rather than clinical evidence.
