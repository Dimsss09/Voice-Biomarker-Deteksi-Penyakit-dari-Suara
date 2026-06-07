"""Train candidate classifiers for Parkinson voice biomarker detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None

from src.data import PROJECT_ROOT


FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "parkinsons_features.csv"
SCHEMA_PATH = PROJECT_ROOT / "data" / "features" / "feature_schema.json"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitData:
    """Container for leakage-safe train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_feature_schema() -> dict[str, Any]:
    """Load the Phase 2 feature schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_feature_table() -> pd.DataFrame:
    """Load the model-ready Phase 2 feature table."""
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURE_PATH}. Run `python -m src.prepare_features` first.")
    return pd.read_csv(FEATURE_PATH)


def _subject_labels(frame: pd.DataFrame) -> pd.DataFrame:
    subject_labels = frame[["subject_id", "label"]].drop_duplicates()
    duplicated = subject_labels["subject_id"].duplicated(keep=False)
    if duplicated.any():
        repeated = sorted(subject_labels.loc[duplicated, "subject_id"].unique())
        raise ValueError(f"Subjects with conflicting labels found: {repeated}")
    return subject_labels


def split_by_subject(frame: pd.DataFrame) -> SplitData:
    """Create stratified train/validation/test splits without subject leakage."""
    subject_labels = _subject_labels(frame)

    train_val_subjects, test_subjects = train_test_split(
        subject_labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=subject_labels["label"],
    )
    train_subjects, validation_subjects = train_test_split(
        train_val_subjects,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val_subjects["label"],
    )

    train = frame[frame["subject_id"].isin(train_subjects["subject_id"])].copy()
    validation = frame[frame["subject_id"].isin(validation_subjects["subject_id"])].copy()
    test = frame[frame["subject_id"].isin(test_subjects["subject_id"])].copy()
    return SplitData(train=train, validation=validation, test=test)


def _candidate_models(y_train: pd.Series) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    candidates: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "svm_rbf": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            {"classifier__C": [0.5, 1.0, 5.0], "classifier__gamma": ["scale", 0.01, 0.1]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight="balanced",
                            n_estimators=300,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {"classifier__max_depth": [None, 4, 8], "classifier__min_samples_leaf": [1, 2, 4]},
        ),
    }

    if XGBClassifier is not None:
        candidates["xgboost"] = (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                            scale_pos_weight=scale_pos_weight,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.05, 0.1],
            },
        )

    return candidates


def _predict_scores(model: Pipeline, x_frame: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(x_frame)[:, 1], index=x_frame.index)
    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(x_frame), index=x_frame.index)
    raise TypeError("Model does not expose probability or decision scores.")


def choose_threshold(y_true: pd.Series, y_score: pd.Series) -> float:
    """Choose a validation threshold that balances sensitivity and specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite_thresholds = [(threshold, tpr_value - fpr_value) for fpr_value, tpr_value, threshold in zip(fpr, tpr, thresholds)]
    finite_thresholds = [(threshold, youden_j) for threshold, youden_j in finite_thresholds if threshold != float("inf")]
    if not finite_thresholds:
        return 0.5
    best_threshold, _ = max(finite_thresholds, key=lambda item: item[1])
    return float(best_threshold)


def evaluate_predictions(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    """Calculate core binary classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _write_split_assignments(splits: SplitData) -> Path:
    rows = []
    for split_name, split_frame in [
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ]:
        rows.extend(
            {
                "split": split_name,
                "subject_id": subject_id,
                "label": int(group["label"].iloc[0]),
                "recordings": int(group.shape[0]),
            }
            for subject_id, group in split_frame.groupby("subject_id")
        )
    path = REPORTS_DIR / "phase3_split_assignments.csv"
    pd.DataFrame(rows).sort_values(["split", "subject_id"]).to_csv(path, index=False)
    return path


def _write_training_report(
    scores: pd.DataFrame,
    best_model_name: str,
    best_params: dict[str, Any],
    splits: SplitData,
) -> Path:
    report_path = REPORTS_DIR / "phase3_training_report.md"
    split_lines = []
    for split_name, split_frame in [
        ("Train", splits.train),
        ("Validation", splits.validation),
        ("Test holdout", splits.test),
    ]:
        label_counts = split_frame["label"].value_counts().sort_index()
        split_lines.append(
            f"| {split_name} | {split_frame['subject_id'].nunique()} | {split_frame.shape[0]} | "
            f"{int(label_counts.get(0, 0))} | {int(label_counts.get(1, 0))} |"
        )

    lines = [
        "# Phase 3 Training Report",
        "",
        "## Split Strategy",
        "",
        "- Split is stratified at subject level using `subject_id` to avoid leakage between recordings from the same person.",
        "- Validation set is used to choose the model. Test set is held out for Phase 4 evaluation.",
        "- `StandardScaler` is inside each sklearn pipeline and is fitted only on training folds/split.",
        "",
        "| Split | Subjects | Recordings | Healthy | Parkinson |",
        "| --- | ---: | ---: | ---: | ---: |",
        *split_lines,
        "",
        "## Validation Results",
        "",
        "| Model | CV ROC-AUC | Validation ROC-AUC | Threshold | Sensitivity | Specificity | F1 | Balanced Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in scores.sort_values("validation_roc_auc", ascending=False).iterrows():
        lines.append(
            f"| {row['model']} | {row['cv_roc_auc_mean']:.3f} | {row['validation_roc_auc']:.3f} | "
            f"{row['validation_threshold']:.3f} | "
            f"{row['validation_sensitivity']:.3f} | {row['validation_specificity']:.3f} | "
            f"{row['validation_f1']:.3f} | {row['validation_balanced_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Model",
            "",
            f"- Best model: `{best_model_name}`",
            f"- Best parameters: `{best_params}`",
            f"- Decision threshold: `{float(scores.iloc[0]['validation_threshold']):.3f}`",
            "- Saved artifact: `models/best_model.joblib`",
            "- Score table: `reports/phase3_model_scores.csv`",
            "- Split assignments: `reports/phase3_split_assignments.csv`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    schema = load_feature_schema()
    feature_columns = schema["feature_columns"]
    frame = load_feature_table()
    splits = split_by_subject(frame)

    x_train = splits.train[feature_columns]
    y_train = splits.train["label"]
    x_validation = splits.validation[feature_columns]
    y_validation = splits.validation["label"]
    groups = splits.train["subject_id"]

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    best_model = None
    best_model_name = ""
    best_params: dict[str, Any] = {}
    best_threshold = 0.5
    best_validation_auc = -1.0

    for model_name, (pipeline, param_grid) in _candidate_models(y_train).items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train, groups=groups)
        y_score = _predict_scores(search.best_estimator_, x_validation)
        threshold = choose_threshold(y_validation, y_score)
        metrics = evaluate_predictions(y_validation, y_score, threshold=threshold)
        validation_auc = metrics["roc_auc"]

        rows.append(
            {
                "model": model_name,
                "cv_roc_auc_mean": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_model = search.best_estimator_
            best_model_name = model_name
            best_params = search.best_params_
            best_threshold = threshold

    if best_model is None:
        raise RuntimeError("No model was trained.")

    scores = pd.DataFrame(rows).sort_values("validation_roc_auc", ascending=False)
    scores_path = REPORTS_DIR / "phase3_model_scores.csv"
    scores.to_csv(scores_path, index=False)
    split_path = _write_split_assignments(splits)

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "best_params": best_params,
        "decision_threshold": best_threshold,
        "feature_columns": feature_columns,
        "schema": schema,
        "random_state": RANDOM_STATE,
    }
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(artifact, model_path)

    report_path = _write_training_report(
        scores=scores,
        best_model_name=best_model_name,
        best_params=best_params,
        splits=splits,
    )

    print(f"Best model: {best_model_name}")
    print(f"Validation ROC-AUC: {best_validation_auc:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"Scores written to: {scores_path}")
    print(f"Split assignments written to: {split_path}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
