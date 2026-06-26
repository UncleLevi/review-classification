# Restaurant Rating Classification Web App

Aplikasi web berbasis **Streamlit** untuk memprediksi rating bintang (1–5) dari teks ulasan restoran menggunakan model **LightGBM** + **SHAP** untuk interpretabilitas.

Aplikasi mendukung:
- Prediksi rating dari satu ulasan (teks bebas)
- Input dalam **Bahasa Indonesia** (auto-translate via Google Translate) atau **Bahasa Inggris**
- Visualisasi confidence model + distribusi probabilitas per kelas
- Penjelasan kata-kata yang paling berpengaruh terhadap prediksi (**SHAP top words**)

---

## Struktur Folder

```text
review_classification/
|-- app.py
|-- requirements.txt
|-- README.md
`-- models/
    |-- lgb_pipeline.pkl
    |-- lgb_preprocessor.pkl
    |-- lgb_model.pkl
    `-- lgb_features.pkl
```

---

## Artefak Model

Simpan artefak berikut dari notebook ke folder `models/`:

```python
import joblib

joblib.dump(full_pipeline,   "models/lgb_pipeline.pkl")
joblib.dump(preprocessor,    "models/lgb_preprocessor.pkl")
joblib.dump(lgb_model,       "models/lgb_model.pkl")
joblib.dump(feature_names,   "models/lgb_features.pkl")
```

| File | Isi |
|------|-----|
| `lgb_pipeline.pkl` | Pipeline lengkap (prep → model), dipakai untuk `predict` dan `predict_proba` |
| `lgb_preprocessor.pkl` | Hanya tahap preprocessing/transformasi fitur, dipakai untuk SHAP |
| `lgb_model.pkl` | Model LightGBM mentah (`LGBMClassifier`) |
| `lgb_features.pkl` | List nama fitur output preprocessor, dipakai untuk label SHAP |

> **Catatan:** Jika `lgb_preprocessor.pkl` tidak ada, aplikasi akan mencoba mengambil step `prep` dari dalam `lgb_pipeline.pkl` secara otomatis.

---

## Cara Kerja Aplikasi

1. User memasukkan teks ulasan restoran (Bahasa Inggris atau Indonesia).
2. Jika bahasa Indonesia dipilih, teks otomatis diterjemahkan ke Bahasa Inggris via `deep-translator` (Google Translate).
3. Teks dikirim ke `lgb_pipeline` untuk menghasilkan prediksi kelas (0–4, mewakili 1–5 bintang).
4. Confidence model ditampilkan dari `predict_proba`.
5. SHAP `TreeExplainer` dijalankan pada `lgb_model` untuk mengidentifikasi kata-kata yang paling mempengaruhi prediksi.

### Kelas Output

| Kelas | Label |
|-------|-------|
| 0 | 1 Star ★☆☆☆☆ |
| 1 | 2 Stars ★★☆☆☆ |
| 2 | 3 Stars ★★★☆☆ |
| 3 | 4 Stars ★★★★☆ |
| 4 | 5 Stars ★★★★★ |

---

## Menjalankan Secara Lokal

1. Buat virtual environment:

```powershell
python -m venv .venv
```

2. Aktifkan virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependency:

```powershell
pip install -r requirements.txt
```

4. Jalankan aplikasi dari folder `review_classification`:

```powershell
streamlit run app.py
```

5. Buka URL lokal yang muncul di terminal, biasanya `http://localhost:8501`.

---

## Dependency Utama

| Package | Kegunaan |
|---------|----------|
| `streamlit` | Framework web app |
| `lightgbm` | Model klasifikasi |
| `shap` | Interpretabilitas model (SHAP TreeExplainer) |
| `scikit-learn` | Pipeline, transformer, preprocessing |
| `deep-translator` | Auto-translate Indonesia → English |
| `matplotlib` | Visualisasi SHAP bar chart |
| `pandas`, `numpy`, `scipy` | Manipulasi data & matriks |
| `joblib` | Load/save model artefak |

---

## Rekomendasi Deployment Gratis

### Opsi Utama: Streamlit Community Cloud

Paling cocok karena:

- Gratis untuk project kecil dan demo
- Tidak perlu backend tambahan
- Native untuk aplikasi Streamlit
- Proses deploy paling sederhana

### Alternatif Gratis

- Hugging Face Spaces
- Render Free Web Service

---

## Deploy ke Streamlit Community Cloud

1. Upload folder project ke GitHub repository.
2. Pastikan file berikut ada:
   - `app.py`
   - `requirements.txt`
   - `models/lgb_pipeline.pkl`
   - `models/lgb_preprocessor.pkl`
   - `models/lgb_model.pkl`
   - `models/lgb_features.pkl`
3. Buka `https://share.streamlit.io/`.
4. Login dengan akun GitHub.
5. Klik **New app**.
6. Pilih repository, branch, dan set main file path ke `app.py`.
7. Klik **Deploy**.

> **Perhatian:** File `.pkl` yang besar (>100 MB) perlu menggunakan [Git LFS](https://git-lfs.github.com/). File `lgb_model.pkl` (~2 MB) dan `lgb_pipeline.pkl` (~2 MB) masih dalam batas normal GitHub.

---

## Error Umum dan Solusinya

### 1. `Pipeline file not found: models/lgb_pipeline.pkl`

Solusi:

- Pastikan file `lgb_pipeline.pkl` ada di folder `models/`
- Jalankan ulang sel export di notebook untuk membuat ulang artefak

### 2. `LightGBM model could not be loaded`

Penyebab:

- `lgb_model.pkl` tidak ditemukan dan pipeline tidak punya step `model`

Solusi:

- Simpan `lgb_model.pkl` secara terpisah menggunakan `joblib.dump`
- Atau pastikan pipeline memiliki named step bernama `"model"`

### 3. `Preprocessor is required for SHAP explanation`

Penyebab:

- `lgb_preprocessor.pkl` tidak ditemukan dan pipeline tidak punya step `prep`

Solusi:

- Simpan preprocessor secara terpisah: `joblib.dump(preprocessor, "models/lgb_preprocessor.pkl")`
- SHAP explanation tidak akan muncul, tapi prediksi tetap berjalan

### 4. SHAP tidak menampilkan kata apapun

Penyebab:

- Tidak ada token dari input yang cocok dengan nama fitur di `lgb_features.pkl`
- Teks terlalu pendek atau tidak mengandung kata yang dikenal model

Solusi:

- Gunakan ulasan yang lebih deskriptif dan panjang
- Pastikan `lgb_features.pkl` berisi nama fitur yang benar dari preprocessor

### 5. Terjemahan gagal (`Translation failed`)

Penyebab:

- Tidak ada koneksi internet atau Google Translate membatasi request

Solusi:

- Aplikasi otomatis fallback ke teks asli (tanpa terjemahan)
- Prediksi tetap berjalan meskipun terjemahan gagal

### 6. Progress bar / confidence tidak muncul

Penyebab:

- Pipeline tidak memiliki method `predict_proba`

Solusi:

- Ini normal untuk beberapa konfigurasi model
- Prediksi rating tetap ditampilkan tanpa confidence score

---

## Catatan Teknis

- Aplikasi menggunakan `st.cache_resource` untuk memuat model hanya sekali per sesi server.
- `ReviewLengthExtractor` didefinisikan ulang di `app.py` agar joblib dapat melakukan deserialisasi pipeline dengan benar (menghindari `ModuleNotFoundError`).
- Sidebar disembunyikan secara default via CSS untuk tampilan yang lebih bersih.
- Responsive design mendukung tampilan mobile (breakpoint 360px, 480px, 768px).
