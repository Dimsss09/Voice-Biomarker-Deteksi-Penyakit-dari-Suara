# Deteksi Dini Parkinson dari Pola Suara Menggunakan Machine Learning

Proyek ini membangun pipeline machine learning untuk membedakan suara sehat dan suara dengan indikasi Parkinson berdasarkan fitur akustik. Target awal menggunakan dataset tabular UCI Parkinson's Voice, lalu fase berikutnya akan menambahkan preprocessing audio, training model, evaluasi, dan demo browser untuk rekam suara atau unggah audio.

> Disclaimer: proyek ini bersifat edukatif/portfolio dan bukan alat diagnosis medis.

## Status Fase

- [x] Fase 0 - Inisialisasi proyek
- [x] Fase 1 - Pengumpulan data dan EDA awal
- [x] Fase 2 - Pra-pemrosesan dan ekstraksi/seleksi fitur
- [x] Fase 3 - Pelatihan model
- [x] Fase 4 - Evaluasi
- [ ] Fase 5 - Demo interaktif browser
- [ ] Fase 6 - Dokumentasi final

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
```

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

## Dataset

Dataset awal memakai UCI Parkinson's Voice. File ini sudah berisi fitur akustik siap pakai seperti pitch, jitter, shimmer, HNR, RPDE, DFA, dan PPE dengan target `status`:

- `0`: Healthy
- `1`: Parkinson
