# Phase 5 Demo Report

## Demo

- App: `app/demo.py`
- Framework: Gradio
- Inputs: browser microphone and audio upload
- Protocol: record a stable `aaaa` vowel for about 5 seconds
- Sample audio: `data/samples/demo_vowel_aaaa.wav`
- Visual polish: full-screen spectrogram-style background, animated waveform logo, metric strip, and two-column prediction layout
- Prediction copy: user-friendly Indonesian summary with indication score and plain-language interpretation

## Inference Flow

1. Audio is loaded as mono.
2. Audio is resampled to 22.05 kHz.
3. Leading and trailing silence are trimmed.
4. Amplitude is normalized.
5. Acoustic features are extracted with librosa and praat-parselmouth.
6. The saved model artifact predicts Parkinson probability.
7. The screening threshold from validation is used for the displayed label.

## Privacy

The app processes audio in memory. It does not persist user recordings.

## Limitation

The trained model uses UCI tabular features. The browser audio extraction path approximates a compatible feature vector for education/demo use, but it has not been clinically validated against live microphone recordings.

## Screenshot Evidence

- `reports/screenshots/demo_home.png`
