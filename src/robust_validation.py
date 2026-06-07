"""Run repeated subject-level cross-validation for robustness evidence."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.data import PROJECT_ROOT
from src.train import evaluate_predictions, load_feature_table


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _predict_scores(model, x_frame: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(x_frame)[:, 1], index=x_frame.index)
    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(x_frame), index=x_frame.index)
    raise TypeError("Model does not expose probability or decision scores.")


def _write_report(scores: pd.DataFrame, repeats: int, folds: int) -> Path:
    report_path = REPORTS_DIR / "robust_subject_cv_report.md"
    summary = scores[["roc_auc", "sensitivity", "specificity", "f1", "balanced_accuracy"]].agg(["mean", "std"])

    lines = [
        "# Robust Subject-Level Cross-Validation Report",
        "",
        "## Setup",
        "",
        f"- Repeats: {repeats}",
        f"- Folds per repeat: {folds}",
        f"- Total folds: {scores.shape[0]}",
        "- Split grouping: `subject_id`",
        "- Threshold: `0.5` for fold-level classification metrics",
        "- Model template: trained Phase 3 winning pipeline cloned and refit inside each fold",
        "",
        "## Metric Stability",
        "",
        "| Metric | Mean | Std |",
        "| --- | ---: | ---: |",
    ]
    for metric in summary.columns:
        lines.append(f"| {metric} | {summary.loc['mean', metric]:.3f} | {summary.loc['std', metric]:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This report checks whether the model family remains useful across multiple subject-level splits.",
            "- It does not replace external validation, because all folds still come from the same UCI dataset.",
            "- Wide standard deviations indicate the dataset is too small for a strong industry claim.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(MODEL_PATH)
    model_template = artifact["model"]
    feature_columns = artifact["feature_columns"]
    frame = load_feature_table()

    x = frame[feature_columns]
    y = frame["label"]
    groups = frame["subject_id"]
    rows = []
    repeats = 5
    folds = 4

    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42 + repeat)
        for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
            model = clone(model_template)
            model.fit(x.iloc[train_index], y.iloc[train_index])
            y_score = _predict_scores(model, x.iloc[test_index])
            metrics = evaluate_predictions(y.iloc[test_index], y_score, threshold=0.5)
            rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "test_subjects": frame.iloc[test_index]["subject_id"].nunique(),
                    "test_records": len(test_index),
                    **metrics,
                }
            )

    scores = pd.DataFrame(rows)
    scores_path = REPORTS_DIR / "robust_subject_cv_scores.csv"
    scores.to_csv(scores_path, index=False)
    report_path = _write_report(scores, repeats=repeats, folds=folds)

    print(f"Robust CV scores written to: {scores_path}")
    print(f"Robust CV report written to: {report_path}")
    print(f"Mean ROC-AUC: {scores['roc_auc'].mean():.3f} +/- {scores['roc_auc'].std():.3f}")


if __name__ == "__main__":
    main()
