# Phase 3 Training Report

## Split Strategy

- Split is stratified at subject level using `subject_id` to avoid leakage between recordings from the same person.
- Validation set is used to choose the model. Test set is held out for Phase 4 evaluation.
- `StandardScaler` is inside each sklearn pipeline and is fitted only on training folds/split.

| Split | Subjects | Recordings | Healthy | Parkinson |
| --- | ---: | ---: | ---: | ---: |
| Train | 18 | 110 | 24 | 86 |
| Validation | 7 | 42 | 12 | 30 |
| Test holdout | 7 | 43 | 12 | 31 |

## Validation Results

| Model | CV ROC-AUC | Validation ROC-AUC | Threshold | Sensitivity | Specificity | F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.906 | 0.493 | 0.979 | 0.400 | 1.000 | 0.571 | 0.700 |
| logistic_regression | 0.897 | 0.492 | 0.951 | 0.433 | 1.000 | 0.605 | 0.717 |
| random_forest | 0.890 | 0.419 | 0.885 | 0.767 | 0.333 | 0.754 | 0.550 |
| svm_rbf | 0.942 | 0.147 | 0.196 | 1.000 | 0.000 | 0.833 | 0.500 |

## Selected Model

- Best model: `xgboost`
- Best parameters: `{'classifier__learning_rate': 0.05, 'classifier__max_depth': 2, 'classifier__n_estimators': 100}`
- Decision threshold: `0.979`
- Saved artifact: `models/best_model.joblib`
- Score table: `reports/phase3_model_scores.csv`
- Split assignments: `reports/phase3_split_assignments.csv`
