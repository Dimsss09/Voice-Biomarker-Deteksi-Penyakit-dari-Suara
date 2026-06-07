"""Audio preprocessing and feature extraction for the browser demo."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf


TARGET_SAMPLE_RATE = 22_050


def load_audio(audio_path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load, resample, trim silence, and normalize a mono audio file."""
    signal, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)
    if signal.size == 0:
        raise ValueError("Audio is empty.")

    signal, _ = librosa.effects.trim(signal, top_db=30)
    if signal.size == 0:
        raise ValueError("Audio only contains silence.")

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float64), target_sr


def _safe_call(default: float, func, *args):
    try:
        value = func(*args)
        if value is None or not np.isfinite(value):
            return default
        return float(value)
    except Exception:
        return default


def _parselmouth_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    sound = parselmouth.Sound(signal, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    pitch_values = pitch.selected_array["frequency"]
    voiced = pitch_values[pitch_values > 0]

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)

    shimmer_local = _safe_call(0.0, call, [sound, point_process], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)
    shimmer_apq3 = _safe_call(0.0, call, [sound, point_process], "Get shimmer (apq3)", 0, 0, 75, 500, 1.3, 1.6)
    shimmer_apq5 = _safe_call(0.0, call, [sound, point_process], "Get shimmer (apq5)", 0, 0, 75, 500, 1.3, 1.6)
    shimmer_apq = _safe_call(0.0, call, [sound, point_process], "Get shimmer (apq11)", 0, 0, 75, 500, 1.3, 1.6)

    hnr = _safe_call(0.0, call, harmonicity, "Get mean", 0, 0)
    nhr = float(1 / (10 ** (hnr / 10))) if hnr > 0 else 0.0

    jitter_percent = _safe_call(0.0, call, point_process, "Get jitter (local)", 0, 0, 75, 500, 1.3) * 100
    jitter_abs = _safe_call(0.0, call, point_process, "Get jitter (local, absolute)", 0, 0, 75, 500, 1.3)
    jitter_rap = _safe_call(0.0, call, point_process, "Get jitter (rap)", 0, 0, 75, 500, 1.3)
    jitter_ppq = _safe_call(0.0, call, point_process, "Get jitter (ppq5)", 0, 0, 75, 500, 1.3)

    return {
        "f0_mean_hz": float(np.mean(voiced)) if voiced.size else 0.0,
        "f0_max_hz": float(np.max(voiced)) if voiced.size else 0.0,
        "f0_min_hz": float(np.min(voiced)) if voiced.size else 0.0,
        "jitter_percent": jitter_percent,
        "jitter_abs": jitter_abs,
        "jitter_rap": jitter_rap,
        "jitter_ppq": jitter_ppq,
        "jitter_ddp": jitter_rap * 3,
        "shimmer_local": shimmer_local,
        "shimmer_db": 20 * np.log10(1 + shimmer_local) if shimmer_local > 0 else 0.0,
        "shimmer_apq3": shimmer_apq3,
        "shimmer_apq5": shimmer_apq5,
        "shimmer_apq": shimmer_apq,
        "shimmer_dda": shimmer_apq3 * 3,
        "noise_to_harmonics_ratio": nhr,
        "harmonics_to_noise_ratio": hnr,
    }


def extract_audio_features(audio_path: str | Path, feature_columns: list[str]) -> pd.DataFrame:
    """Extract a single-row feature table compatible with the trained model."""
    signal, sample_rate = load_audio(audio_path)
    features = _parselmouth_features(signal, sample_rate)

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(signal)

    features.update(
        {
            # UCI nonlinear measures are not directly reproducible from one short
            # browser clip, so we use stable spectral proxies for demo inference.
            "rpde": float(np.std(mfcc[0]) / (np.mean(np.abs(mfcc[0])) + 1e-8)),
            "dfa": float(np.mean(spectral_bandwidth) / sample_rate),
            "spread1": float(np.mean(mfcc[1])) if mfcc.shape[0] > 1 else 0.0,
            "spread2": float(np.std(mfcc[1])) if mfcc.shape[0] > 1 else 0.0,
            "d2": float(np.mean(spectral_centroid) / sample_rate),
            "ppe": float(np.mean(zcr)),
        }
    )

    row = {column: float(features.get(column, 0.0)) for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def create_sample_audio(output_path: str | Path, duration: float = 5.0, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Create a simple sustained vowel-like sample for demo smoke tests."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = np.minimum(1.0, np.linspace(0, 4, time.size)) * np.minimum(1.0, np.linspace(4, 0, time.size))
    signal = (
        0.55 * np.sin(2 * np.pi * 150 * time)
        + 0.25 * np.sin(2 * np.pi * 300 * time)
        + 0.12 * np.sin(2 * np.pi * 450 * time)
    )
    signal = signal * envelope
    sf.write(output_path, signal, sample_rate)
    return output_path
