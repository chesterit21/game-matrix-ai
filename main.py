# main.py

import os
import shutil
import pandas as pd
import json
import torch
import numpy as np
from collections import Counter
from src.data_ingestion import fetch_data_from_sql_server, preprocess_data_for_embedding, save_documents_to_folder, save_predictions_to_db, update_training_game_status, prepare_tabular_data
from src.config import *
from src.embedding_models import generate_embedding_for_chunk
from src.chromadb_handler import initialize_chromadb, store_embeddings_in_chromadb
import joblib
from src.rag_pipeline import run_rag_prediction

SQL_QUERY_MASTER = "SELECT GameCode FROM TRAININGGAME WHERE IsActive = 1;"
SQL_QUERY_LOG = "SELECT Id, GameCode, Periode, LogResult, [As], Kop, Kepala, Ekor,DateResultInGame FROM LogGame WHERE GameCode='{gameCode}' AND Periode > 1150 ORDER BY Periode ASC;"
SQL_QUERY_HISTORICAL = "SELECT LogResult FROM LogGame WHERE GameCode='{gameCode}' ORDER BY Periode DESC;"
from src.prediction_models import XGBoostModel, VAEPredictionModel

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_features_to_numbers(predicted_features_batch: np.ndarray, scaler, feature_keys: list) -> set:
    """
    Mengubah BATCH fitur yang diprediksi menjadi satu set angka 4-digit yang beragam.
    Logika ini disamakan dengan _decode_predicted_features_to_number di train.py untuk konsistensi.
    """
    if predicted_features_batch is None or predicted_features_batch.size == 0:
        return set()

    # pred_df has shape (batch_size, num_features) with columns from feature_keys
    pred_df = pd.DataFrame(predicted_features_batch, columns=feature_keys)

    cols_for_number = ['As', 'Kop', 'Kepala', 'Ekor']

    if not all(c in pred_df.columns for c in cols_for_number):
        return {"0000"}

    # The scaler was fit on all feature_keys. We must inverse_transform the entire DataFrame.
    inversed_features = scaler.inverse_transform(pred_df)
    # Create a new DataFrame with the original column names for easy access.
    inversed_df = pd.DataFrame(inversed_features, columns=feature_keys)
    as_vals = inversed_df['As'].values
    kop_vals = inversed_df['Kop'].values
    kepala_vals = inversed_df['Kepala'].values
    ekor_vals = inversed_df['Ekor'].values

    predicted_numbers = set()
    num_predictions = len(inversed_df)

    # Loop through each prediction in the batch
    for i in range(num_predictions):
        # For each prediction, get the floor/ceil candidates
        as_candidates = {int(np.clip(f(as_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}
        kop_candidates = {int(np.clip(f(kop_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}
        kepala_candidates = {int(np.clip(f(kepala_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}
        ekor_candidates = {int(np.clip(f(ekor_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}

        # Generate combinations for this single prediction
        for a in as_candidates:
            for k in kop_candidates:
                for ke in kepala_candidates:
                    for e in ekor_candidates:
                        predicted_numbers.add(f"{a}{k}{ke}{e}")
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
                    embedding_vector = generate_embedding_for_chunk(doc_data, model_type=first_model_type, model_path=first_embedding_model_path, feature_keys_path=feature_keys_path)
                    doc_data["embedding_vector"] = embedding_vector.tolist()
                    documents_with_embeddings.append(doc_data)
            
            chroma_collection = initialize_chromadb(persist_directory=CHROMA_DB_PERSIST_DIR, collection_name=CHROMA_COLLECTION_NAME)
            store_embeddings_in_chromadb(chroma_collection, documents_with_embeddings)
            print(f"Data untuk {game_code} berhasil diindeks ke ChromaDB.")
    except Exception as e:
        print(f"Gagal memproses atau mengindeks data untuk RAG: {e}")
        return
    
    # --- Hierarchical Ensemble Prediction ---
    all_prediction_sets = []
    prediction_model_types = ['LSTM', 'GRU', 'MLP', 'VAE', 'TCN'] # Deep Learning Models

    for embedding_model_type in EMBEDDING_MODEL_TYPES:
        for prediction_model_type in prediction_model_types:
            print(f"\n-- Menjalankan RAG dengan [Embedding: {embedding_model_type}, Analis: {prediction_model_type}] --")
            
            # SemanticTransformer is a library model, not a file. VAE is generative and doesn't need an embedding model.
            embedding_model_path = "" if embedding_model_type == "SemanticTransformer" else os.path.join(game_model_dir, f"{embedding_model_type}_model.pth")
            prediction_model_path = os.path.join(game_model_dir, f"{prediction_model_type}_model.pth")

            # Check for required model files
            required_files = [prediction_model_path]
            if embedding_model_path: required_files.append(embedding_model_path)
            if not all(os.path.exists(p) for p in required_files):
                print(f"Error: Kombinasi model untuk '{game_code}' tidak ditemukan. Melewatkan.")
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
                print(f"Data mentah terlalu pendek. Melewatkan RAG.")
                continue
                
            current_query_data = df_raw.tail(PIPELINE_WINDOW_SIZE).copy()
            
            try:
                predicted_features_batch = run_rag_prediction(
                    current_input_df=current_query_data,
                    chroma_db_persist_dir=CHROMA_DB_PERSIST_DIR,
                    chroma_collection_name=CHROMA_COLLECTION_NAME,
                    embedding_model_path=embedding_model_path,
                    prediction_model_path=prediction_model_path,
                    scaler_path=scaler_path,
                    feature_keys_path=feature_keys_path,
                    embedding_model_type=embedding_model_type,
                    prediction_model_type=prediction_model_type,
                    n_retrieved_chunks=N_RETRIEVED_CHUNKS
                )
                
                if predicted_features_batch is not None and predicted_features_batch.size > 0:
                    predicted_numbers_set = decode_features_to_numbers(predicted_features_batch, scaler, feature_keys)
                    all_prediction_sets.append(predicted_numbers_set)
                    print(f"    -> Dihasilkan {len(predicted_numbers_set)} prediksi unik.")

            except Exception as e:
                print(f"Gagal menjalankan pipeline RAG: {e}")

    # --- Menjalankan Analis Klasik: XGBoost ---
    print("\n-- Menjalankan Analis Klasik: XGBoost --")
    try:
        # The path for the XGBoost wrapper is a directory
        xgb_model_path = os.path.join(game_model_dir, "XGBoost_models")
        if os.path.exists(xgb_model_path):
            xgb_model_wrapper = XGBoostModel()
            xgb_model_wrapper.load_model(xgb_model_path)
            
            # Siapkan data untuk XGBoost
            xgb_input_df = df_raw.tail(PIPELINE_WINDOW_SIZE)
            X_xgb, _ = prepare_tabular_data(xgb_input_df)
            
            if not X_xgb.empty:
                features_for_prediction = X_xgb.tail(1) # Get the last row of features
                if not features_for_prediction.empty:
                    xgb_predicted_digits = xgb_model_wrapper.predict(features_for_prediction)
                    
                    xgb_predicted_digits = np.round(xgb_predicted_digits).astype(int)
                    xgb_predicted_digits = np.clip(xgb_predicted_digits, 0, 9)
                    
                    pred_str = "".join(map(str, xgb_predicted_digits[0]))
                    all_prediction_sets.append({pred_str})
                    print(f"    -> Dihasilkan 1 prediksi unik dari XGBoost: {pred_str}")
    except Exception as e:
        print(f"Gagal menjalankan prediksi XGBoost: {e}")

    # --- Tahap 2: Konsensus Akhir (Voting) ---
    if not all_prediction_sets:
        print(f"Tidak ada prediksi yang dihasilkan untuk {game_code}.")
        return

    # Gabungkan semua prediksi dari semua set menjadi satu list besar
    all_predictions_flat = [num for pred_set in all_prediction_sets for num in pred_set]
    
    # Hitung frekuensi kemunculan setiap angka (ini adalah sistem voting sederhana)
    prediction_counts = Counter(all_predictions_flat)
    
    # Urutkan prediksi berdasarkan jumlah suara (frekuensi), dari yang paling banyak muncul
    # Ini adalah prediksi yang paling disetujui oleh "dewan ahli"
    sorted_by_votes = [item[0] for item in prediction_counts.most_common()]
    
    Temp_Data_Predictions = set(sorted_by_votes)
    print(f"\nTotal prediksi unik dari semua ahli: {len(Temp_Data_Predictions)}")

    # --- Tahap 3: Aggressive Generative Augmentation (jika perlu) ---
    if len(Temp_Data_Predictions) < PREDICTION_HORIZON:
        print(f"\nJumlah prediksi ({len(Temp_Data_Predictions)}) kurang dari target {PREDICTION_HORIZON}.")
        print("Memulai fase 'Generative Augmentation' menggunakan VAE...")

        # Pastikan scaler dan feature_keys sudah dimuat sebelum augmentasi
        try:
            scaler = joblib.load(scaler_path)
            with open(feature_keys_path, 'r') as f:
                feature_keys = json.load(f)
        except Exception as e:
            print(f"Gagal memuat scaler atau feature_keys untuk augmentasi: {e}")
            return
        
        try:
            vae_model_path = os.path.join(game_model_dir, "VAE_model.pth")
            if not os.path.exists(vae_model_path) or 'VAE' not in prediction_model_types:
                print("Model VAE tidak ditemukan, augmentasi dibatalkan.")
            else:
                state_dict = torch.load(vae_model_path, map_location=device)
                # Muat model VAE
                input_dim = state_dict['encoder_fc1.weight'].shape[1]
                hidden_dim = state_dict['encoder_fc1.bias'].shape[0]
                output_dim = state_dict['decoder_fc2.bias'].shape[0]
                vae_model = VAEPredictionModel(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=VAE_LATENT_DIM, output_dim=output_dim)
                vae_model.load_state_dict(state_dict)
                vae_model.to(device)
                vae_model.eval()

                with torch.no_grad():
                    # Terus generate hingga target tercapai, dengan batas keamanan yang lebih besar
                    safety_break = 0

                    while len(Temp_Data_Predictions) < PREDICTION_HORIZON and safety_break < 500000:
                        # Generate dalam batch besar untuk efisiensi dan keragaman
                        batch_size = PREDICTION_HORIZON # Targetkan untuk menghasilkan banyak sampel sekaligus
                        z = torch.randn(batch_size, VAE_LATENT_DIM).to(device)
                        generated_features = vae_model.decode(z)
                        
                        # Dekode fitur yang di-generate
                        new_predictions = decode_features_to_numbers(generated_features.cpu().numpy(), scaler, feature_keys)
                        Temp_Data_Predictions.update(new_predictions)
                        safety_break += 1
                        print(f"Iterasi Augmentasi {safety_break}: Total unik sekarang {len(Temp_Data_Predictions)}")
                print(f"Setelah augmentasi, total prediksi unik: {len(Temp_Data_Predictions)}")
        except Exception as e:
            print(f"Gagal menjalankan Generative Augmentation: {e}")

    # --- Tahap 4: Finalisasi dan Penyimpanan ---
    # Ambil prediksi teratas sejumlah PREDICTION_HORIZON

    # Inisialisasi final_predictions_list agar selalu terdefinisi
    final_predictions_list = sorted(list(Temp_Data_Predictions))[:PREDICTION_HORIZON]
    # Proses looping sebanyak 120.000.000 kali untuk finalisasi prediksi
    for _ in range(120000):
        final_predictions_list = sorted(list(Temp_Data_Predictions))[:PREDICTION_HORIZON]
        final_predictions_string = "*".join(final_predictions_list)
        # Jika jumlah prediksi sudah memenuhi horizon, keluar dari loop
        if len(final_predictions_list) >= PREDICTION_HORIZON:
            break

    if len(final_predictions_list) >= PREDICTION_HORIZON:
        final_predictions_string = "*".join(final_predictions_list)
        try:
            save_predictions_to_db(SQL_CONNECTION_STRING, game_code, final_predictions_string)
        except Exception as e:
            print(f"Gagal menyimpan prediksi final ke database untuk {game_code}: {e}")
    else:
        print(f"GAGAL FINAL: Jumlah prediksi unik ({len(final_predictions_list)}) tetap tidak mencapai target {PREDICTION_HORIZON}. Tidak ada data yang disimpan.")

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
        # Setelah pipeline prediksi selesai untuk game_code ini, update statusnya
        try:
            print(f"Pipeline prediksi untuk {game_code} selesai. Mengupdate status di database...")
            update_training_game_status(SQL_CONNECTION_STRING, game_code)
        except Exception as e:
            print(f"Gagal mengupdate status untuk {game_code} setelah prediksi: {e}")

    print("\n--- Semua Proses Prediksi Selesai ---")

if __name__ == "__main__":
    main()
