# Industry Readiness Checklist

## Current Status

The project is stronger than a simple notebook prototype because it now has subject-level splits, saved model artifacts, threshold policy, held-out evaluation, bootstrap confidence intervals, calibration plot, model card, and reproducible scripts.

It is still not industry-ready for medical or clinical use.

## Completed

- [x] Reproducible data pipeline
- [x] Model-ready feature table
- [x] Subject-level train/validation/test split
- [x] Multiple model candidates
- [x] Saved model artifact
- [x] Held-out test metrics
- [x] Sensitivity, specificity, ROC-AUC, F1, confusion matrix
- [x] Threshold comparison
- [x] Bootstrap confidence intervals
- [x] Calibration plot and Brier score
- [x] Permutation feature importance
- [x] Model card with limitations
- [x] Repeated subject-level cross-validation stability report

## Still Required Before Industry Use

- [ ] External validation on a separate dataset
- [ ] Raw-audio feature extraction validated against browser microphone recordings
- [ ] Probability calibration fitted on validation data
- [ ] External replication of subject-level cross-validation on another dataset
- [ ] Bias/fairness analysis across age, sex, microphone type, language, and recording condition
- [ ] Data provenance and license review
- [ ] CI pipeline that reruns prepare/train/evaluate checks
- [ ] Drift monitoring plan
- [ ] Human review workflow for uncertain predictions
- [ ] Clinical/regulatory review if used beyond education

## Practical Interpretation

Use the current result as a portfolio-grade ML prototype. Present it honestly as a non-diagnostic screening demo with strong caveats, not as a clinically validated Parkinson detector.
