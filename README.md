# Game Matrix AI - Sistem Prediksi Berbasis RAG

## Peringatan dan Disclaimer

> **PENTING: Ini Hanya Eksperimental Saja, untuk tujuan Pembelajaran Personal.**
>
> Penulis tidak bertanggung jawab atas segala bentuk penyalahgunaan, kerugian, atau kerusakan yang timbul dari penggunaan program ini oleh pihak lain. Gunakan dengan risiko dan tanggung jawab Anda sendiri.

## Ikhtisar

Proyek ini adalah sistem AI canggih yang dirancang untuk memprediksi urutan angka 4 digit untuk berbagai "permainan" (`GameCode`). Sistem ini memanfaatkan arsitektur **Retrieval-Augmented Generation (RAG)** yang modern, menggabungkan kekuatan model deep learning untuk prediksi deret waktu (LSTM) dengan database vektor (ChromaDB) untuk memori kontekstual jangka panjang.

## Arsitektur Sistem

Sistem ini terdiri dari beberapa komponen utama yang bekerja secara harmonis:

- **`train.py`**: Skrip utama untuk melakukan *online training*. Skrip ini secara terus-menerus melatih dan menyempurnakan model dengan data historis baru untuk setiap `GameCode`.
- **`main.py`**: Skrip utama untuk menjalankan pipeline prediksi. Skrip ini mengorkestrasi proses RAG, menghasilkan prediksi final, dan menyimpannya ke database.
- **`src/config.py`**: File konfigurasi terpusat untuk semua hyperparameter dan pengaturan, memastikan konsistensi di seluruh proyek.
- **`src/embedding_models.py`**: Berisi model-model Autoencoder (`DWTAutoencoder`, `Time2VecAutoencoder`) yang bertanggung jawab untuk membuat representasi vektor padat (embeddings) dari data historis.
- **`src/prediction_models.py`**: Berisi `LSTMModel`, mesin prediksi inti yang dirancang untuk memahami pola sekuensial dalam data deret waktu.
- **`src/rag_pipeline.py`**: Jantung dari sistem. Pipeline ini mengambil situasi saat ini, mencari data historis yang paling relevan dari ChromaDB menggunakan embeddings, dan memberikan konteks yang kaya ini ke model LSTM untuk prediksi yang lebih cerdas.
- **`src/chromadb_handler.py`**: Mengelola semua interaksi dengan database vektor ChromaDB, termasuk menyimpan dan mengambil embeddings.
- **`src/data_ingestion.py`**: Menangani pengambilan data dari SQL Server, pra-pemrosesan, dan pembuatan chunk data.

## Fitur Unggulan

- **Online Training**: Sistem dapat terus dilatih dengan data baru tanpa harus memulai dari awal, membuatnya selalu relevan.
- **Retrieval-Augmented Generation (RAG)**: Meningkatkan akurasi prediksi dengan memberikan model contoh historis yang relevan, mengatasi keterbatasan memori model standar.
- **Multi-Shot Ensemble Prediction**: Menghasilkan set prediksi yang kaya dan beragam dengan memaksa model untuk mengeksplorasi berbagai skenario "bagaimana jika", menghindari output yang monoton dan repetitif.
- **Dynamic Context Expansion**: Strategi training cerdas di mana model secara otomatis menggunakan konteks historis yang lebih luas jika gagal membuat prediksi yang benar pada percobaan awal.
- **Akselerasi GPU**: Memanfaatkan sepenuhnya GPU NVIDIA melalui CUDA untuk training dan inferensi berkecepatan tinggi.
- **Arsitektur Multi-Tenant**: Semua model, scaler, dan data chunk diorganisir dengan rapi ke dalam folder terpisah untuk setiap `GameCode`, memastikan integritas data dan skalabilitas.

## Prasyarat

### Perangkat Lunak
- **Sistem Operasi**: Fedora Workstation (Sangat Direkomendasikan)
- **Python**: 3.13+
- **NVIDIA Stack**:
  - **Driver NVIDIA**: Versi 13+ (atau yang kompatibel dengan CUDA 13)
  - **CUDA Toolkit**: Versi 13.0+
- **Microsoft SQL Server**

### Perangkat Keras (Konfigurasi yang Diuji)
- **GPU**: NVIDIA GeForce RTX 4060 8GB
- **CPU**: Intel Core i9-13900H
- **RAM**: 64GB
- **Penyimpanan**: 1TB SSD

> **Catatan Instalasi Penting**: Untuk performa dan stabilitas maksimal, sangat disarankan untuk menginstal driver NVIDIA dan CUDA Toolkit langsung dari situs resmi NVIDIA. **Jangan gunakan installer dari repositori pihak ketiga seperti RPM Fusion.** Jika Anda sudah menginstalnya, hapus terlebih dahulu atau gunakan flag `--allowerasing` saat menjalankan skrip instalasi resmi dari NVIDIA untuk memastikan instalasi yang bersih.

## Pengaturan & Instalasi

1.  **Clone Repositori**:
    ```bash
    git clone <url-repositori-anda>
    cd game-matrix-ai
    ```

2.  **Buat Lingkungan Virtual Python**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows, gunakan `venv\Scripts\activate`
    ```

3.  **Instal Dependensi**:
    ```bash
    pip install torch pandas numpy scikit-learn joblib chromadb sqlalchemy pymssql
    ```

4.  **Konfigurasi Database**:
    - Buka file `src/config.py`.
    - Perbarui variabel `SQL_CONNECTION_STRING` dengan kredensial MS SQL Server Anda.

## Cara Menjalankan

Sistem ini beroperasi dalam alur kerja dua tahap:

### 1. Training Model

Jalankan skrip training terlebih dahulu. Proses ini akan mengolah data historis, melatih model embedding dan prediksi untuk setiap `GameCode` yang aktif (`IsActive = 1`), dan menyimpan file model (`.pth`) serta artefak (`.joblib`, `.json`) ke dalam direktori `trained_models/` dan `artifacts/`, yang diorganisir per `GameCode`.

```bash
python train.py
```

Setelah proses training untuk sebuah `GameCode` selesai, statusnya (`IsActive`) di tabel `TRAININGGAME` akan otomatis diubah menjadi `0`.

### 2. Menghasilkan Prediksi

Setelah model dilatih, jalankan skrip prediksi utama. Proses ini akan:
- Mengindeks data historis ke dalam database vektor ChromaDB.
- Menjalankan pipeline RAG untuk setiap `GameCode`.
- Menghasilkan satu set prediksi final yang beragam.
- Menyimpan prediksi tersebut ke tabel `AINumberGame` di database.

```bash
python main.py
```

## Struktur Proyek

```
game-matrix-ai/
├── artifacts/
│   └── <GameCode>_scaler.joblib
│   └── <GameCode>_feature_keys.json
├── chroma_db/
│   └── ... (Data ChromaDB)
├── docs/
│   └── <GameCode>/
│       └── chunk_...json
├── trained_models/
│   └── <GameCode>/            # Folder spesifik untuk setiap game
│       ├── DWT_Autoencoder_model.pth
│       ├── Time2Vec_model.pth
│       ├── TransformerAutoencoder_model.pth
│       ├── LSTM_model.pth
│       ├── GRU_model.pth
│       ├── MLP_model.pth
│       ├── TCN_model.pth
│       └── XGBoost_models/    # Direktori untuk model-model XGBoost
│           └── xgb_model_...json
├── src/
│   ├── __init__.py
│   ├── chromadb_handler.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── embedding_models.py
│   ├── prediction_models.py
│   └── rag_pipeline.py
├── main.py
├── train.py
└── README.md
```