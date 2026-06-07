# Deteksi Dini Parkinson dari Pola Suara Menggunakan Machine Learning

Proyek ini membangun pipeline machine learning untuk membedakan suara sehat dan suara dengan indikasi Parkinson berdasarkan fitur akustik. Target awal menggunakan dataset tabular UCI Parkinson's Voice, lalu menyediakan demo browser untuk rekam suara atau unggah audio.

> Disclaimer: proyek ini bersifat edukatif/portfolio dan bukan alat diagnosis medis.

## Status Fase

- [x] Fase 0 - Inisialisasi proyek
- [x] Fase 1 - Pengumpulan data dan EDA awal
- [x] Fase 2 - Pra-pemrosesan dan ekstraksi/seleksi fitur
- [x] Fase 3 - Pelatihan model
- [x] Fase 4 - Evaluasi
- [x] Fase 5 - Demo interaktif browser
- [x] Fase 6 - Dokumentasi final

## Struktur Repo

```text
data/
  raw/          Dataset mentah
  processed/    Data/audio terproses
  features/     Tabel fitur siap latih
notebooks/      Eksperimen dan EDA notebook
src/            Kode pipeline
models/         Model tersimpan
reports/        Laporan dan visualisasi
app/            Demo interaktif
config/         Konfigurasi dataset/pipeline
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Jalankan Pipeline Data

```powershell
python -m src.download_data
python -m src.eda
python -m src.prepare_features
python -m src.train
python -m src.evaluate
python -m src.robust_validation
```

## Jalankan Demo Browser

```powershell
python -m app.demo
```

Buka `http://127.0.0.1:7860`, lalu:

1. Tekan rekam.
2. Ucapkan vokal `aaaa` secara stabil sekitar 5 detik.
3. Klik `Prediksi`.

Demo juga menerima unggahan file audio dan menyertakan sample audio sintetis untuk smoke test.

Output utama:

- Dataset: `data/raw/parkinsons.data`
- Ringkasan EDA: `reports/phase1_eda_summary.md`
- Ringkasan subject: `reports/subject_summary.csv`
- Visualisasi: `reports/class_distribution.png`, `reports/correlation_heatmap.png`, `reports/feature_distributions_by_label.png`
- Tabel fitur siap latih: `data/features/parkinsons_features.csv`
- Skema fitur: `data/features/feature_schema.json`
- Model terbaik: `models/best_model.joblib`
- Laporan training: `reports/phase3_training_report.md`
- Laporan evaluasi: `reports/phase4_evaluation_report.md`
- Visual evaluasi: `reports/phase4_confusion_matrix.png`, `reports/phase4_roc_curve.png`
- Model card: `reports/model_card.md`
- Industry readiness checklist: `reports/industry_readiness_checklist.md`
- Robust CV report: `reports/robust_subject_cv_report.md`
- Demo app: `app/demo.py`
- Audio inference: `src/predict.py`, `src/audio_features.py`
- Demo report: `reports/phase5_demo_report.md`
- Finalization report: `reports/phase6_finalization_report.md`

## Dataset

Dataset awal memakai UCI Parkinson's Voice. File ini sudah berisi fitur akustik siap pakai seperti pitch, jitter, shimmer, HNR, RPDE, DFA, dan PPE dengan target `status`:

- `0`: Healthy
- `1`: Parkinson

## Hasil Evaluasi

Held-out subject-level test set:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.997 |
| Sensitivity | 1.000 |
| Specificity | 0.667 |
| F1 | 0.939 |
| Balanced Accuracy | 0.833 |

Repeated subject-level cross-validation:

| Metric | Mean | Std |
| --- | ---: | ---: |
| ROC-AUC | 0.802 | 0.123 |
| Sensitivity | 0.827 | 0.125 |
| Specificity | 0.550 | 0.276 |

Angka repeated CV lebih konservatif dan lebih realistis untuk membaca stabilitas model.

## Catatan Penting

- Model dilatih pada fitur tabular UCI, bukan dataset audio browser langsung.
- Demo audio mengekstrak fitur pendekatan dari rekaman mikrofon, sehingga hasil demo adalah ilustrasi edukatif.
- Belum ada external validation, fairness analysis, atau validasi klinis.
- Jangan memakai output proyek ini untuk diagnosis atau keputusan medis.
