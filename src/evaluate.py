"""Evaluate the trained Parkinson voice classifier on the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, brier_score_loss, confusion_matrix, roc_curve

from src.data import PROJECT_ROOT
from src.train import evaluate_predictions, load_feature_table


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
SPLIT_PATH = PROJECT_ROOT / "reports" / "phase3_split_assignments.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


def _load_test_frame() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing split assignments: {SPLIT_PATH}. Run `python -m src.train` first.")
    frame = load_feature_table()
    split_assignments = pd.read_csv(SPLIT_PATH)
    test_subjects = split_assignments.loc[split_assignments["split"] == "test", "subject_id"]
    test_frame = frame[frame["subject_id"].isin(test_subjects)].copy()
    if test_frame.empty:
        raise ValueError("Test split is empty.")
    return test_frame


def _predict_scores(model, x_frame: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(x_frame)[:, 1], index=x_frame.index)
    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(x_frame), index=x_frame.index)
    raise TypeError("Model does not expose probability or decision scores.")


def _plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    path = REPORTS_DIR / "phase4_confusion_matrix.png"
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Parkinson"],
        yticklabels=["Healthy", "Parkinson"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Held-Out Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _plot_roc_curve(y_true: pd.Series, y_score: pd.Series) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    path = REPORTS_DIR / "phase4_roc_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color="#2563eb", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Held-Out Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _plot_calibration_curve(y_true: pd.Series, y_score: pd.Series) -> Path:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=5, strategy="quantile")
    path = REPORTS_DIR / "phase4_calibration_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1, label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Held-Out Test Calibration")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _write_predictions(test_frame: pd.DataFrame, y_score: pd.Series, y_pred: pd.Series) -> Path:
    path = REPORTS_DIR / "phase4_test_predictions.csv"
    output = test_frame[["recording_id", "subject_id", "label"]].copy()
    output["probability_parkinson"] = y_score.round(6)
    output["predicted_label"] = y_pred.astype(int)
    output.to_csv(path, index=False)
    return path


def _write_threshold_comparison(
    y_true: pd.Series,
    y_score: pd.Series,
    thresholds: dict[str, float],
) -> tuple[Path, pd.DataFrame]:
    path = REPORTS_DIR / "phase4_threshold_comparison.csv"
    rows = []
    for label, threshold in thresholds.items():
        metrics = evaluate_predictions(y_true, y_score, threshold=threshold)
        rows.append({"threshold_name": label, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path, frame


def _bootstrap_subject_ci(
    test_frame: pd.DataFrame,
    y_score: pd.Series,
    threshold: float,
    repeats: int = 1000,
) -> tuple[Path, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(42)
    subject_ids = np.array(sorted(test_frame["subject_id"].unique()))
    scored = test_frame[["subject_id", "label"]].copy()
    scored["score"] = y_score
    rows = []

    for _ in range(repeats):
        sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        sampled = pd.concat([scored[scored["subject_id"] == subject_id] for subject_id in sampled_subjects])
        if sampled["label"].nunique() < 2:
            continue
        metrics = evaluate_predictions(sampled["label"], sampled["score"], threshold=threshold)
        rows.append(metrics)

    path = REPORTS_DIR / "phase4_bootstrap_ci.csv"
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(path, index=False)
    summary: dict[str, dict[str, float]] = {}
    for metric in ["roc_auc", "sensitivity", "specificity", "f1", "balanced_accuracy"]:
        summary[metric] = {
            "mean": float(bootstrap[metric].mean()),
            "ci_low": float(bootstrap[metric].quantile(0.025)),
            "ci_high": float(bootstrap[metric].quantile(0.975)),
        }
    return path, summary


def _write_feature_importance(model, x_test: pd.DataFrame, y_test: pd.Series) -> Path:
    path = REPORTS_DIR / "phase4_permutation_importance.csv"
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc",
    )
    importance = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(path, index=False)
    return path


def _write_evaluation_report(
    artifact: dict,
    metrics: dict[str, float],
    test_frame: pd.DataFrame,
    confusion_path: Path,
    roc_path: Path,
    calibration_path: Path,
    predictions_path: Path,
    threshold_comparison_path: Path,
    threshold_comparison: pd.DataFrame,
    bootstrap_path: Path,
    bootstrap_summary: dict[str, dict[str, float]],
    brier_score: float,
    importance_path: Path,
) -> Path:
    report_path = REPORTS_DIR / "phase4_evaluation_report.md"
    label_counts = test_frame["label"].value_counts().sort_index()
    subject_counts = test_frame.groupby("label")["subject_id"].nunique().sort_index()

    lines = [
        "# Phase 4 Evaluation Report",
        "",
        "## Model",
        "",
        f"- Model: `{artifact['model_name']}`",
        f"- Parameters: `{artifact['best_params']}`",
        f"- Decision threshold: `{artifact['decision_threshold']:.3f}`",
        f"- Feature count: {len(artifact['feature_columns'])}",
        "",
        "## Held-Out Test Set",
        "",
        f"- Subjects: {test_frame['subject_id'].nunique()}",
        f"- Recordings: {test_frame.shape[0]}",
        "",
        "| Label | Meaning | Recordings | Subjects |",
        "| --- | --- | ---: | ---: |",
        f"| 0 | Healthy | {int(label_counts.get(0, 0))} | {int(subject_counts.get(0, 0))} |",
        f"| 1 | Parkinson | {int(label_counts.get(1, 0))} | {int(subject_counts.get(1, 0))} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| ROC-AUC | {metrics['roc_auc']:.3f} |",
        f"| Sensitivity / Recall | {metrics['sensitivity']:.3f} |",
        f"| Specificity | {metrics['specificity']:.3f} |",
        f"| F1 | {metrics['f1']:.3f} |",
        f"| Accuracy | {metrics['accuracy']:.3f} |",
        f"| Balanced Accuracy | {metrics['balanced_accuracy']:.3f} |",
        f"| Brier Score | {brier_score:.3f} |",
        "",
        "## Bootstrap 95% CI",
        "",
        "| Metric | Mean | 95% CI Low | 95% CI High |",
        "| --- | ---: | ---: | ---: |",
    ]

    for metric, values in bootstrap_summary.items():
        lines.append(
            f"| {metric} | {values['mean']:.3f} | {values['ci_low']:.3f} | {values['ci_high']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Comparison",
            "",
            "| Threshold | ROC-AUC | Sensitivity | Specificity | F1 | Balanced Accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in threshold_comparison.iterrows():
        lines.append(
            f"| {row['threshold_name']} ({row['threshold']:.3f}) | {row['roc_auc']:.3f} | "
            f"{row['sensitivity']:.3f} | {row['specificity']:.3f} | {row['f1']:.3f} | "
            f"{row['balanced_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| | Pred Healthy | Pred Parkinson |",
            "| --- | ---: | ---: |",
            f"| Actual Healthy | {int(metrics['tn'])} | {int(metrics['fp'])} |",
            f"| Actual Parkinson | {int(metrics['fn'])} | {int(metrics['tp'])} |",
            "",
            "## Artifacts",
            "",
            f"- Confusion matrix: `{_relative(confusion_path)}`",
            f"- ROC curve: `{_relative(roc_path)}`",
            f"- Calibration curve: `{_relative(calibration_path)}`",
            f"- Test predictions: `{_relative(predictions_path)}`",
            f"- Threshold comparison: `{_relative(threshold_comparison_path)}`",
            f"- Bootstrap CI samples: `{_relative(bootstrap_path)}`",
            f"- Permutation importance: `{_relative(importance_path)}`",
            "",
            "## Notes",
            "",
            "- This is a subject-level held-out evaluation, so recordings from test subjects were not used for model selection.",
            "- The default operating threshold is selected from validation data for screening sensitivity, not optimized on the test set.",
            "- Dataset size is small, especially for healthy subjects, so metrics should be read as educational project evidence rather than clinical evidence.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact["decision_threshold"])
    test_frame = _load_test_frame()

    x_test = test_frame[feature_columns]
    y_test = test_frame["label"]
    y_score = _predict_scores(model, x_test)
    y_pred = (y_score >= threshold).astype(int)

    metrics = evaluate_predictions(y_test, y_score, threshold=threshold)
    brier_score = float(brier_score_loss(y_test, y_score))
    metrics_path = REPORTS_DIR / "phase4_metrics.json"
    metrics_path.write_text(json.dumps({**metrics, "brier_score": brier_score}, indent=2) + "\n", encoding="utf-8")

    confusion_path = _plot_confusion_matrix(y_test, y_pred)
    roc_path = _plot_roc_curve(y_test, y_score)
    calibration_path = _plot_calibration_curve(y_test, y_score)
    predictions_path = _write_predictions(test_frame, y_score, y_pred)
    thresholds = artifact.get("thresholds", {"decision_threshold": threshold, "default_0_5": 0.5})
    threshold_comparison_path, threshold_comparison = _write_threshold_comparison(y_test, y_score, thresholds=thresholds)
    bootstrap_path, bootstrap_summary = _bootstrap_subject_ci(test_frame, y_score, threshold=threshold)
    importance_path = _write_feature_importance(model, x_test, y_test)
    report_path = _write_evaluation_report(
        artifact=artifact,
        metrics=metrics,
        test_frame=test_frame,
        confusion_path=confusion_path,
        roc_path=roc_path,
        calibration_path=calibration_path,
        predictions_path=predictions_path,
        threshold_comparison_path=threshold_comparison_path,
        threshold_comparison=threshold_comparison,
        bootstrap_path=bootstrap_path,
        bootstrap_summary=bootstrap_summary,
        brier_score=brier_score,
        importance_path=importance_path,
    )

    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
