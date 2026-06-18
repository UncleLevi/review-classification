# Sentiment / Rating Classifier Web App

Project ini mengubah pipeline machine learning berbasis notebook menjadi aplikasi web menggunakan Streamlit. Aplikasi mendukung prediksi satu review, probabilitas kelas jika model mendukung `predict_proba`, dan batch prediction lewat upload CSV.

## Struktur Folder

```text
review_classification/
|-- app.py
|-- utils.py
|-- requirements.txt
|-- README.md
`-- models/
    `-- model.pkl
```

Opsional jika Anda memakai artefak lama yang dipisah:

```text
models/
|-- model.pkl
|-- vectorizer.pkl
`-- label_encoder.pkl
```

## Cara Kerja Sesuai Notebook

Notebook final Anda melakukan inferensi dengan alur berikut:

1. preprocessing teks manual
2. pipeline `FeatureUnion`
3. `TfidfVectorizer`
4. fitur panjang review
5. model klasifikasi

Karena itu, cara paling aman untuk deployment adalah menyimpan `1 file pipeline lengkap` ke `models/model.pkl`.

## Artefak yang Direkomendasikan

Simpan pipeline terbaik langsung dari notebook:

```python
import joblib

joblib.dump(model_nb, "models/model.pkl")
```

Catatan:

- `model_nb`, `model_xgb`, atau `model_lgb` pada notebook sudah berupa pipeline lengkap.
- Untuk notebook final Anda, `Naive Bayes` justru memberi hasil terbaik di evaluasi akhir.
- Anda tidak wajib menyimpan `vectorizer.pkl` terpisah jika pipeline sudah lengkap.

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

## Rekomendasi Deployment Gratis

### Opsi Utama: Streamlit Community Cloud

Paling cocok karena:

- gratis untuk project kecil dan demo
- tidak perlu backend tambahan
- native untuk aplikasi Streamlit
- proses deploy paling sederhana

### Alternatif Gratis

- Hugging Face Spaces
- Render Free Web Service

## Deploy ke Streamlit Community Cloud

1. Upload folder project ke GitHub repository.
2. Pastikan file berikut ada:
   - `app.py`
   - `utils.py`
   - `requirements.txt`
   - `models/model.pkl`
3. Buka `https://share.streamlit.io/`.
4. Login dengan akun GitHub.
5. Klik **New app**.
6. Pilih repository, branch, dan set main file path ke `app.py`.
7. Klik **Deploy**.

## Error Umum dan Solusinya

### 1. `File tidak ditemukan: models/model.pkl`

Solusi:

- pastikan file pipeline ada di folder `models/`
- atau ubah path model di sidebar aplikasi

### 2. `Vectorizer tidak ditemukan`

Penyebab:

- file model yang di-load bukan pipeline lengkap

Solusi:

- pakai pipeline penuh dari notebook
- atau sediakan `vectorizer.pkl` jika model memang dipisah

### 3. `LookupError` terkait NLTK stopwords / wordnet

Penyebab:

- resource NLTK belum tersedia

Solusi:

- aplikasi akan mencoba mengunduh resource otomatis
- jika deploy, pastikan environment punya akses untuk instal dependency normal

### 4. Hasil prediksi tidak konsisten

Penyebab:

- preprocessing web app berbeda dengan notebook

Solusi:

- `utils.py` saat ini sudah disesuaikan dengan notebook final:
  - contraction expansion
  - lowercase
  - hapus URL, HTML, dan karakter tertentu
  - stopword removal dengan negation words tetap dipertahankan
  - lemmatization

### 5. `predict_proba` tidak tersedia

Solusi:

- prediksi tetap jalan
- hanya probabilitas/confidence yang tidak akan tampil
