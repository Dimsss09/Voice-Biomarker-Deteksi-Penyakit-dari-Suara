# Model Card: Voice Biomarker Parkinson Classifier

## Intended Use

This model is an educational proof of concept for classifying UCI Parkinson voice feature records as healthy (`0`) or Parkinson (`1`). It is designed for portfolio demonstration and research-style experimentation.

It is not a medical device, not a diagnostic system, and must not be used to make clinical decisions.

## Model Details

- Model family: XGBoost classifier inside a scikit-learn pipeline
- Input type: tabular acoustic voice features from UCI Parkinson's dataset
- Feature count: 22
- Default operating threshold: screening threshold selected on validation data for at least 90% sensitivity
- Model artifact: `models/best_model.joblib`

## Dataset

- Source: UCI Parkinson's Voice dataset
- Records: 195
- Parsed subjects: 32
- Healthy subjects: 8
- Parkinson subjects: 24
- Held-out test subjects: 7

The dataset is small and imbalanced. Healthy subjects are especially underrepresented.

## Evaluation

Held-out subject-level test evaluation:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.997 |
| Sensitivity | 1.000 |
| Specificity | 0.667 |
| F1 | 0.939 |
| Balanced Accuracy | 0.833 |
| Brier Score | 0.083 |

Bootstrap 95% confidence intervals are available in `reports/phase4_evaluation_report.md`.

Repeated subject-level CV stability:

| Metric | Mean | Std |
| --- | ---: | ---: |
| ROC-AUC | 0.802 | 0.123 |
| Sensitivity | 0.827 | 0.125 |
| Specificity | 0.550 | 0.276 |
| F1 | 0.834 | 0.065 |
| Balanced Accuracy | 0.689 | 0.117 |

This repeated CV result is a better indicator of model stability than the single held-out split alone.

## Threshold Policy

The default threshold prioritizes screening sensitivity:

- `screening_min_sensitivity_0_90`: primary threshold for demo/screening-style behavior
- `youden_j`: stricter threshold with high specificity but poor recall on test
- `default_0_5`: common reference threshold

Thresholds are documented in `reports/phase4_threshold_comparison.csv`.

## Limitations

- Trained on tabular UCI acoustic features, not live browser audio.
- No external validation dataset yet.
- No demographic fairness analysis yet.
- No calibration method has been fitted yet.
- Test set contains only 7 subjects, so uncertainty remains high.
- Repeated CV shows meaningful variance across subject splits.
- Microphone, room noise, language, accent, and recording protocol robustness are not validated.

## Recommended Next Steps

- Add an external Parkinson voice dataset for validation.
- Train and evaluate a raw-audio feature extraction path equivalent to the browser demo.
- Add repeated subject-level cross-validation reports.
- Add probability calibration on validation data.
- Add data/version tracking and automated CI checks for all pipelines.
- Keep a strong non-diagnostic disclaimer in all user-facing UI.
