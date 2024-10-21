# Analisis Data pada Bike Sharing Dataset Tahun 2011-2012 ✨

## Setup Environment - Anaconda
1. Buat lingkungan baru dengan nama `main-ds` dan Python versi 3.9:
   ```bash
   conda create --name main-ds python=3.9
   ```
2. Aktifkan lingkungan tersebut:
   ```bash
   conda activate main-ds
   ```
3. Instal semua ketergantungan yang diperlukan dari berkas `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Setup Environment - Shell/Terminal
1. Buat direktori untuk proyek analisis data:
   ```bash
   mkdir proyek_analisis_data
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd proyek_analisis_data
   ```
3. Instal `pipenv` dan buat lingkungan virtual:
   ```bash
   pipenv install
   pipenv shell
   ```
4. Instal semua ketergantungan yang diperlukan dari berkas `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Run Streamlit App
Jalankan aplikasi Streamlit menggunakan perintah berikut:
```bash
streamlit run dashboard.py
```

## Catatan
Pastikan Anda memiliki semua file data yang diperlukan dalam folder yang sama dengan berkas `dashboard.py` untuk menjalankan aplikasi dengan benar.
