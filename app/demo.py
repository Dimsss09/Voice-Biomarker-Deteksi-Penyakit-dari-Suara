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

THEME_CSS = """
html, body, gradio-app, .gradio-container {
    background:
        linear-gradient(135deg, rgba(6, 24, 38, 0.94), rgba(17, 41, 52, 0.88)),
        repeating-linear-gradient(90deg, rgba(125, 211, 252, 0.10) 0 1px, transparent 1px 24px),
        repeating-linear-gradient(0deg, rgba(45, 212, 191, 0.08) 0 1px, transparent 1px 18px) !important;
    color: #e5f6ff;
    min-height: 100vh;
    width: 100%;
}
body {
    margin: 0 !important;
    overflow-x: hidden;
}
.gradio-container {
    max-width: none !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 24px clamp(18px, 5vw, 72px) 32px !important;
    box-sizing: border-box;
}
.main {
    max-width: none !important;
}
.contain {
    max-width: none !important;
}
.hero-shell {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background:
        linear-gradient(120deg, rgba(8, 47, 73, 0.82), rgba(15, 118, 110, 0.42)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.10) 0 2px, transparent 2px 16px);
    border-radius: 8px;
    padding: clamp(22px, 3vw, 38px);
    margin: 0 0 18px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
}
.hero-shell:after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 88px;
    background: repeating-linear-gradient(90deg, rgba(45,212,191,0.00) 0 10px, rgba(45,212,191,0.26) 10px 12px);
    opacity: 0.55;
    animation: driftWave 8s linear infinite;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 14px;
    position: relative;
    z-index: 1;
}
.voice-logo {
    width: 58px;
    height: 58px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    align-items: center;
    gap: 4px;
    padding: 10px;
    border-radius: 8px;
    background: rgba(2, 6, 23, 0.56);
    border: 1px solid rgba(125, 211, 252, 0.44);
}
.voice-logo span {
    display: block;
    width: 100%;
    border-radius: 99px;
    background: linear-gradient(180deg, #67e8f9, #2dd4bf);
    animation: pulseBar 1.35s ease-in-out infinite;
}
.voice-logo span:nth-child(1) { height: 18px; animation-delay: 0s; }
.voice-logo span:nth-child(2) { height: 30px; animation-delay: .12s; }
.voice-logo span:nth-child(3) { height: 42px; animation-delay: .24s; }
.voice-logo span:nth-child(4) { height: 28px; animation-delay: .36s; }
.voice-logo span:nth-child(5) { height: 20px; animation-delay: .48s; }
.hero-title {
    margin: 0;
    font-size: 34px;
    line-height: 1.08;
    letter-spacing: 0;
    color: #f8fafc;
}
.hero-subtitle {
    margin: 12px 0 0;
    max-width: 920px;
    color: #c7e8f3;
    font-size: 16px;
    line-height: 1.55;
    position: relative;
    z-index: 1;
}
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 22px;
    position: relative;
    z-index: 1;
}
.metric-pill {
    border: 1px solid rgba(226, 232, 240, 0.20);
    border-radius: 8px;
    padding: 12px;
    background: rgba(15, 23, 42, 0.42);
}
.metric-pill strong {
    display: block;
    color: #ffffff;
    font-size: 20px;
}
.metric-pill span {
    display: block;
    color: #bae6fd;
    font-size: 12px;
    margin-top: 2px;
}
.notice-band {
    border-left: 4px solid #38bdf8;
    background: rgba(8, 47, 73, 0.55);
    border-radius: 8px;
    padding: 12px 14px;
    color: #e0f2fe;
    margin-bottom: 18px;
}
.intro-copy {
    color: #eff6ff;
    margin: 0 0 18px;
    font-size: 15px;
}
.result-card textarea {
    font-size: 16px !important;
    line-height: 1.55 !important;
}
.technical-details {
    opacity: 0.92;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    border-radius: 8px !important;
}
button.primary {
    background: linear-gradient(90deg, #0f766e, #0369a1) !important;
    border: 0 !important;
}
@keyframes pulseBar {
    0%, 100% { transform: scaleY(.65); opacity: .72; }
    50% { transform: scaleY(1.18); opacity: 1; }
}
@keyframes driftWave {
    from { transform: translateX(0); }
    to { transform: translateX(34px); }
}
@media (max-width: 720px) {
    .hero-title { font-size: 26px; }
    .metric-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero-shell { padding: 20px; }
}
"""

HERO_HTML = f"""
<section class="hero-shell">
  <div class="brand-row">
    <div class="voice-logo" aria-label="Voice waveform logo">
      <span></span><span></span><span></span><span></span><span></span>
    </div>
    <div>
      <h1 class="hero-title">Voice Biomarker Parkinson Demo</h1>
      <p class="hero-subtitle">
        Rekam vokal <strong>aaaa</strong> sekitar 5 detik atau unggah audio.
        Sistem membaca pola suara, lalu memberi indikasi edukatif yang mudah dipahami.
      </p>
    </div>
  </div>
  <div class="metric-strip">
    <div class="metric-pill"><strong>0.997</strong><span>Held-out ROC-AUC</span></div>
    <div class="metric-pill"><strong>1.000</strong><span>Sensitivity test</span></div>
    <div class="metric-pill"><strong>0.667</strong><span>Specificity test</span></div>
    <div class="metric-pill"><strong>0.802</strong><span>Repeated CV ROC-AUC</span></div>
  </div>
</section>
<div class="notice-band">
  <strong>Disclaimer:</strong> {DISCLAIMER}
</div>
"""


def _format_prediction(audio_path: str | None) -> tuple[str, str]:
    if audio_path is None:
        return "Silakan rekam suara atau unggah audio terlebih dahulu.", "{}"

    try:
        result = predict_audio(audio_path)
    except Exception as exc:
        return f"Audio belum bisa diproses.\n\nPenyebab teknis: {exc}", "{}"

    probability = float(result["probability_parkinson"])
    percentage = probability * 100
    if probability >= 0.75:
        risk_text = "Indikasi pola suara mirip Parkinson terdeteksi tinggi."
        tone = "Gunakan hasil ini sebagai sinyal edukatif saja, bukan kesimpulan medis."
    elif probability >= 0.45:
        risk_text = "Pola suara berada di area abu-abu."
        tone = "Coba rekam ulang di ruangan yang lebih tenang untuk melihat apakah hasilnya konsisten."
    else:
        risk_text = "Indikasi pola suara mirip Parkinson terdeteksi rendah."
        tone = "Hasil rendah tidak berarti bebas risiko medis."
    summary = (
        f"{risk_text}\n\n"
        f"Skor indikasi: {percentage:.1f}%\n\n"
        f"Makna hasil:\n"
        f"- Skor lebih tinggi berarti pola fitur suara lebih mirip data Parkinson pada dataset latihan.\n"
        f"- Hasil bisa berubah karena kualitas mikrofon, noise, durasi rekaman, dan cara mengucapkan vokal.\n"
        f"- {tone}\n\n"
        f"{DISCLAIMER}"
    )
    feature_preview = json.dumps(result["features"], indent=2)
    return summary, feature_preview


def build_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    create_sample_audio(SAMPLE_AUDIO_PATH)
    with gr.Blocks(title="Voice Biomarker Parkinson Demo") as demo:
        gr.HTML(HERO_HTML)
        gr.HTML(
            '<p class="intro-copy">Rekaman diproses di memori dan tidak disimpan oleh aplikasi. '
            'Gunakan ruangan yang tenang agar pembacaan pola suara lebih stabil.</p>'
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Rekam atau unggah audio",
                )
                with gr.Row():
                    predict_button = gr.Button("Prediksi", variant="primary")
                    clear_button = gr.ClearButton()
                gr.Markdown(
                    "Protokol: ucapkan vokal **aaaa** stabil sekitar 5 detik. "
                    "Jangan gunakan hasil ini sebagai keputusan medis."
                )
            with gr.Column(scale=4):
                prediction = gr.Textbox(label="Hasil prediksi", lines=11, elem_classes=["result-card"])
                with gr.Accordion("Detail teknis fitur audio", open=False, elem_classes=["technical-details"]):
                    features = gr.Code(label="Fitur yang diekstrak", language="json")
        gr.Examples(examples=[str(SAMPLE_AUDIO_PATH)], inputs=audio, label="Contoh audio")

        predict_button.click(_format_prediction, inputs=audio, outputs=[prediction, features])
        clear_button.add([audio, prediction, features])
    return demo


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=7860, css=THEME_CSS)


if __name__ == "__main__":
    main()
