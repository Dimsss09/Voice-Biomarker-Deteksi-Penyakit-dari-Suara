# Deteksi Dini Parkinson dari Pola Suara Menggunakan Machine Learning

Proyek ini membangun pipeline machine learning untuk membedakan suara sehat dan suara dengan indikasi Parkinson berdasarkan fitur akustik. Target awal menggunakan dataset tabular UCI Parkinson's Voice, lalu fase berikutnya akan menambahkan preprocessing audio, training model, evaluasi, dan demo browser untuk rekam suara atau unggah audio.

> Disclaimer: proyek ini bersifat edukatif/portfolio dan bukan alat diagnosis medis.

## Status Fase

- [x] Fase 0 - Inisialisasi proyek
- [ ] Fase 1 - Pengumpulan data dan EDA awal
- [ ] Fase 2 - Pra-pemrosesan dan ekstraksi/seleksi fitur
- [ ] Fase 3 - Pelatihan model
- [ ] Fase 4 - Evaluasi
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

## Jalankan Fase 1

```powershell
python -m src.download_data
python -m src.eda
```

Output utama:

- Dataset: `data/raw/parkinsons.data`
- Ringkasan EDA: `reports/phase1_eda_summary.md`
- Visualisasi: `reports/class_distribution.png`, `reports/correlation_heatmap.png`, `reports/feature_distributions_by_label.png`

## Dataset

Dataset awal memakai UCI Parkinson's Voice. File ini sudah berisi fitur akustik siap pakai seperti pitch, jitter, shimmer, HNR, RPDE, DFA, dan PPE dengan target `status`:

- `0`: Healthy
- `1`: Parkinson
