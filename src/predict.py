"""Single-sample audio inference for the Gradio demo."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.audio_features import extract_audio_features
from src.data import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model_artifact(model_path: str | Path = MODEL_PATH) -> dict:
    """Load the trained model artifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}. Run `python -m src.train` first.")
    return joblib.load(model_path)


def predict_audio(audio_path: str | Path, threshold: float | None = None) -> dict[str, object]:
    """Predict Parkinson probability from one audio file."""
    artifact = load_model_artifact()
    feature_columns = artifact["feature_columns"]
    model = artifact["model"]
    threshold = float(threshold if threshold is not None else artifact["decision_threshold"])

    features = extract_audio_features(audio_path, feature_columns)
    probability = float(model.predict_proba(features)[0, 1])
    predicted_label = int(probability >= threshold)
    label_text = "Parkinson" if predicted_label == 1 else "Healthy"

    return {
        "label": label_text,
        "predicted_label": predicted_label,
        "probability_parkinson": probability,
        "threshold": threshold,
        "features": features.iloc[0].to_dict(),
        "model_name": artifact["model_name"],
    }
