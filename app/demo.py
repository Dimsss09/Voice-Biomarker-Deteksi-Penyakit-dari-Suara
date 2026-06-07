"""Gradio browser demo for recording or uploading voice audio."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_features import create_sample_audio
from src.predict import predict_audio


SAMPLE_AUDIO_PATH = PROJECT_ROOT / "data" / "samples" / "demo_vowel_aaaa.wav"
DISCLAIMER = (
    "Ini bukan alat diagnosis medis. Demo ini hanya edukasi/portfolio dan "
    "belum tervalidasi untuk keputusan klinis."
)


def _format_prediction(audio_path: str | None) -> tuple[str, str]:
    if audio_path is None:
        return "Belum ada audio.", "{}"

    try:
        result = predict_audio(audio_path)
    except Exception as exc:
        return f"Gagal memproses audio: {exc}", "{}"

    probability = float(result["probability_parkinson"])
    confidence = probability if result["predicted_label"] == 1 else 1 - probability
    summary = (
        f"Prediksi: {result['label']}\n"
        f"Probabilitas Parkinson: {probability:.3f}\n"
        f"Confidence label terpilih: {confidence:.3f}\n"
        f"Threshold: {result['threshold']:.3f}\n"
        f"Model: {result['model_name']}\n\n"
        f"{DISCLAIMER}"
    )
    feature_preview = json.dumps(result["features"], indent=2)
    return summary, feature_preview


def build_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    create_sample_audio(SAMPLE_AUDIO_PATH)
    with gr.Blocks(title="Voice Biomarker Parkinson Demo") as demo:
        gr.Markdown("# Voice Biomarker Parkinson Demo")
        gr.Markdown(
            "Tekan rekam lalu ucapkan vokal **aaaa** secara stabil selama sekitar 5 detik, "
            "atau unggah file audio. Rekaman diproses di memori dan tidak disimpan oleh aplikasi."
        )
        gr.Markdown(f"**Disclaimer:** {DISCLAIMER}")

        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Rekam atau unggah audio",
        )
        with gr.Row():
            predict_button = gr.Button("Prediksi", variant="primary")
            clear_button = gr.ClearButton()

        prediction = gr.Textbox(label="Hasil prediksi", lines=8)
        features = gr.Code(label="Fitur audio yang diekstrak", language="json")
        gr.Examples(examples=[str(SAMPLE_AUDIO_PATH)], inputs=audio, label="Contoh audio")

        predict_button.click(_format_prediction, inputs=audio, outputs=[prediction, features])
        clear_button.add([audio, prediction, features])
    return demo


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
