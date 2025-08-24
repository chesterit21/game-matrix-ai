# main.py

import os
import shutil
import pandas as pd
import json
import numpy as np
from src.data_ingestion import fetch_data_from_sql_server, preprocess_data_for_embedding, save_documents_to_folder, save_predictions_to_db
from src.config import SQL_CONNECTION_STRING, DOCS_FOLDER, CHROMA_DB_PERSIST_DIR, CHROMA_COLLECTION_NAME, TRAINED_FILES_DIR, ARTIFACTS_DIR, PIPELINE_WINDOW_SIZE, OVERLAP_SIZE, EMBEDDING_MODEL_TYPES, PREDICTION_HORIZON, N_RETRIEVED_CHUNKS
from src.embedding_models import generate_embedding_for_chunk
from src.chromadb_handler import initialize_chromadb, store_embeddings_in_chromadb
import joblib
from src.rag_pipeline import run_rag_prediction

SQL_QUERY_MASTER = "SELECT GameCode FROM TRAININGGAME WHERE IsActive = 1;"
SQL_QUERY_LOG = "SELECT Id, GameCode, Periode, LogResult, [As], Kop, Kepala, Ekor,DateResultInGame FROM LogGame WHERE GameCode='{gameCode}' AND Periode > 1150 ORDER BY Periode ASC;"
SQL_QUERY_HISTORICAL = "SELECT LogResult FROM LogGame WHERE GameCode='{gameCode}' ORDER BY Periode DESC;"

def decode_features_to_numbers(predicted_features: np.ndarray, scaler, feature_keys: list) -> set:
    """
    Mengubah array fitur yang diprediksi menjadi satu set angka 4-digit yang beragam.
    Logika ini disamakan dengan _decode_predicted_features_to_number di train.py untuk konsistensi
    dan dioptimalkan menggunakan operasi vectorized NumPy.
    """
    if predicted_features is None or predicted_features.size == 0:
        return set()

    pred_df = pd.DataFrame([predicted_features], columns=feature_keys)

    numerical_cols = ['As', 'Kop', 'Kepala', 'Ekor', 'Id']
    cols_to_transform = [col for col in numerical_cols if col in pred_df.columns]
    cols_for_number = ['As', 'Kop', 'Kepala', 'Ekor']

    if not all(c in pred_df.columns for c in cols_for_number):
        return {"0000"}

    inversed_features = scaler.inverse_transform(pred_df[cols_to_transform])
    inversed_df = pd.DataFrame(inversed_features, columns=cols_to_transform)

    # Gunakan logika yang sama persis dengan train.py untuk konsistensi
    vals = inversed_df[cols_for_number].values[0] # Ambil baris pertama
    as_candidates = {int(np.clip(f(vals[0]), 0, 9)) for f in (np.floor, np.ceil)}
    kop_candidates = {int(np.clip(f(vals[1]), 0, 9)) for f in (np.floor, np.ceil)}
    kepala_candidates = {int(np.clip(f(vals[2]), 0, 9)) for f in (np.floor, np.ceil)}
    ekor_candidates = {int(np.clip(f(vals[3]), 0, 9)) for f in (np.floor, np.ceil)}

    predicted_numbers = {f"{a}{k}{ke}{e}" for a in as_candidates for k in kop_candidates for ke in kepala_candidates for e in ekor_candidates}
    return predicted_numbers

def run_prediction_pipeline_for_game_code(game_code: str):
    """
    Menjalankan pipeline prediksi lengkap untuk satu game_code, menggabungkan hasil
    dari semua model, mengisi sisa prediksi dari data historis, dan menyimpan ke DB.
    """
    print(f"--- Memulai Pipeline Prediksi Gabungan untuk GameCode: {game_code} ---")

    # Definisikan path artefak spesifik untuk game_code ini
    game_docs_folder = os.path.join(DOCS_FOLDER, game_code)
    scaler_path = os.path.join(ARTIFACTS_DIR, f"{game_code}_scaler.joblib")
    feature_keys_path = os.path.join(ARTIFACTS_DIR, f"{game_code}_feature_keys.json")
    game_model_dir = os.path.join(TRAINED_FILES_DIR, game_code)

    try:
        query = SQL_QUERY_LOG.format(gameCode=game_code)
        df_raw = fetch_data_from_sql_server(SQL_CONNECTION_STRING, query)
        if df_raw.empty:
            print(f"Tidak ada data log untuk GameCode {game_code}. Melewatkan.")
            return
        print(f"Berhasil mengambil {len(df_raw)} baris data.")
    except Exception as e:
        print(f"Gagal mengambil data log untuk {game_code}: {e}")
        return

    # Ingest data ke ChromaDB (hanya perlu sekali per game_code)
    try:
        processed_chunks = preprocess_data_for_embedding(df_raw, chunk_size=PIPELINE_WINDOW_SIZE, overlap=OVERLAP_SIZE, scaler_path=scaler_path, feature_keys_path=feature_keys_path)
        if processed_chunks:
            # Gunakan path model pertama yang ada untuk menghasilkan embedding untuk ingestion
            first_model_type = EMBEDDING_MODEL_TYPES[0]
            first_embedding_model_path = os.path.join(game_model_dir, f"{first_model_type}_model.pth")
            if not os.path.exists(first_embedding_model_path):
                 print(f"Peringatan: Model embedding awal '{first_model_type}' tidak ditemukan untuk ingestion ChromaDB.")
                 return

            save_documents_to_folder(processed_chunks, game_docs_folder)
            documents_with_embeddings = []
            for filename in os.listdir(game_docs_folder):
                if filename.endswith(".json"):
                    file_path = os.path.join(game_docs_folder, filename)
                    with open(file_path, 'r') as f: doc_data = json.load(f)
                    embedding_vector = generate_embedding_for_chunk(doc_data, model_type=first_model_type, model_path=first_embedding_model_path)
                    doc_data["embedding_vector"] = embedding_vector.tolist()
                    documents_with_embeddings.append(doc_data)
            
            chroma_collection = initialize_chromadb(persist_directory=CHROMA_DB_PERSIST_DIR, collection_name=CHROMA_COLLECTION_NAME)
            store_embeddings_in_chromadb(chroma_collection, documents_with_embeddings)
            print(f"Data untuk {game_code} berhasil diindeks ke ChromaDB.")
    except Exception as e:
        print(f"Gagal memproses atau mengindeks data untuk RAG: {e}")
        return

    Temp_Data_Predictions = set()

    for model_type in EMBEDDING_MODEL_TYPES:
        print(f"\n-- Menjalankan RAG dengan model embedding: {model_type} --")
        # Sesuaikan nama file model dengan output dari train.py yang baru
        embedding_model_path = os.path.join(game_model_dir, f"{model_type}_model.pth")
        prediction_model_path = os.path.join(game_model_dir, "LSTM_model.pth")

        if not all(os.path.exists(p) for p in [embedding_model_path, prediction_model_path]):
            print(f"Error: Model untuk '{game_code}' tipe '{model_type}' tidak ditemukan. Melewatkan.")
            continue

        # Muat scaler dan feature_keys yang diperlukan untuk decoding
        try:
            scaler = joblib.load(scaler_path)
            with open(feature_keys_path, 'r') as f:
                feature_keys = json.load(f)
        except FileNotFoundError:
            print(f"Error: scaler.joblib atau feature_keys.json tidak ditemukan. Pastikan train.py sudah dijalankan.")
            continue

        if len(df_raw) < PIPELINE_WINDOW_SIZE:
            print(f"Data mentah terlalu pendek. Melewatkan RAG untuk {model_type}.")
            continue
            
        current_query_data = df_raw.tail(PIPELINE_WINDOW_SIZE).copy()
        
        try:
            # Panggil RAG pipeline yang sudah diupdate
            predicted_features = run_rag_prediction(
                current_input_df=current_query_data,
                chroma_db_persist_dir=CHROMA_DB_PERSIST_DIR,
                chroma_collection_name=CHROMA_COLLECTION_NAME,
                embedding_model_path=embedding_model_path,
                prediction_model_path=prediction_model_path,
                scaler_path=scaler_path,
                feature_keys_path=feature_keys_path,
                embedding_model_type=model_type,
                n_retrieved_chunks=N_RETRIEVED_CHUNKS
            )
            
            if predicted_features is not None and predicted_features.size > 0:
                # Gunakan logika decoding yang lebih canggih, mirip dengan train.py
                predicted_numbers = decode_features_to_numbers(predicted_features, scaler, feature_keys)
                for num in predicted_numbers:
                    Temp_Data_Predictions.add(num)
                print(f"Prediksi dari model {model_type}: {predicted_numbers}. Total unik sekarang: {len(Temp_Data_Predictions)}.")

        except Exception as e:
            print(f"Gagal menjalankan pipeline RAG untuk model {model_type}: {e}")

    if not Temp_Data_Predictions:
        print(f"Tidak ada prediksi yang dihasilkan untuk {game_code}.")
        return

    if len(Temp_Data_Predictions) < PREDICTION_HORIZON:
        print(f"Jumlah prediksi ({len(Temp_Data_Predictions)}) kurang dari {PREDICTION_HORIZON}. Mengisi dari data historis...")
        try:
            hist_query = SQL_QUERY_HISTORICAL.format(gameCode=game_code)
            df_hist = fetch_data_from_sql_server(SQL_CONNECTION_STRING, hist_query)
            if not df_hist.empty:
                historical_numbers = [str(num).zfill(4) for num in df_hist['LogResult'].dropna().unique() if len(str(num)) == 4]
                for hist_num in historical_numbers:
                    if len(Temp_Data_Predictions) >= PREDICTION_HORIZON: break
                    Temp_Data_Predictions.add(hist_num)
                print(f"Setelah pengisian, total prediksi unik: {len(Temp_Data_Predictions)}.")
        except Exception as e:
            print(f"Gagal mengambil data historis: {e}")

    final_predictions_list = sorted(list(Temp_Data_Predictions))
    final_predictions_string = "*".join(final_predictions_list)

    try:
        save_predictions_to_db(SQL_CONNECTION_STRING, game_code, final_predictions_string)
    except Exception as e:
        print(f"Gagal menyimpan prediksi final ke database untuk {game_code}: {e}")

def main():
    """
    Fungsi utama untuk menjalankan pipeline prediksi untuk semua game code dari master.
    """
    # Best Practice: Clean up artifacts from previous runs to prevent contamination.
    # This ensures that only data from the current run is used for indexing.
    # We clean the base docs folder. Game-specific folders will be created inside the pipeline.
    if os.path.exists(DOCS_FOLDER): shutil.rmtree(DOCS_FOLDER)
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(TRAINED_FILES_DIR, exist_ok=True)

    print("Memulai proses prediksi untuk semua game code dari master...")
    try:
        df_master = fetch_data_from_sql_server(SQL_CONNECTION_STRING, SQL_QUERY_MASTER)
        game_codes = df_master['GameCode'].tolist()
        if not game_codes:
            print("Tidak ada game code yang ditemukan. Berhenti.")
            return
        print(f"Ditemukan {len(game_codes)} game code untuk diproses: {game_codes}")
    except Exception as e:
        print(f"Gagal mengambil daftar game code: {e}")
        return

    for game_code in game_codes:
        run_prediction_pipeline_for_game_code(game_code)

    print("\n--- Semua Proses Prediksi Selesai ---")

if __name__ == "__main__":
    main()
