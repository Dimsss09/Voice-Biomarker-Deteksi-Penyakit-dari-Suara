# Phase 3 Training Report

## Split Strategy

- Split is stratified at subject level using `subject_id` to avoid leakage between recordings from the same person.
- Validation set is used to choose the model. Test set is held out for Phase 4 evaluation.
- `StandardScaler` is inside each sklearn pipeline and is fitted only on training folds/split.
- The saved decision threshold is a screening threshold chosen on validation data to target at least 90% sensitivity.

| Split | Subjects | Recordings | Healthy | Parkinson |
| --- | ---: | ---: | ---: | ---: |
| Train | 18 | 110 | 24 | 86 |
| Validation | 7 | 42 | 12 | 30 |
| Test holdout | 7 | 43 | 12 | 31 |

## Validation Results

| Model | CV ROC-AUC | Validation ROC-AUC | Screening Threshold | Sensitivity | Specificity | F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.906 | 0.493 | 0.759 | 0.900 | 0.000 | 0.783 | 0.450 |
| logistic_regression | 0.897 | 0.492 | 0.549 | 0.900 | 0.000 | 0.783 | 0.450 |
| random_forest | 0.890 | 0.419 | 0.782 | 0.900 | 0.000 | 0.783 | 0.450 |
| svm_rbf | 0.942 | 0.147 | 0.954 | 0.900 | 0.000 | 0.783 | 0.450 |

## Selected Model

- Best model: `xgboost`
- Best parameters: `{'classifier__learning_rate': 0.05, 'classifier__max_depth': 2, 'classifier__n_estimators': 100}`
- Screening decision threshold: `0.759`
- Youden J threshold: `0.979`
- Saved artifact: `models/best_model.joblib`
- Score table: `reports/phase3_model_scores.csv`
- Split assignments: `reports/phase3_split_assignments.csv`
