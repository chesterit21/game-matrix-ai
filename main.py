# main.py

import os
import shutil
import pandas as pd
import json
import torch
import numpy as np
from collections import Counter
import itertools
import random
from src.data_ingestion import fetch_data_from_sql_server, preprocess_data_for_embedding, save_documents_to_folder, save_predictions_to_db, update_training_game_status, prepare_tabular_data
from src.config import *
from src.embedding_models import generate_embedding_for_chunk
from src.chromadb_handler import initialize_chromadb, store_embeddings_in_chromadb
import joblib
from src.rag_pipeline import run_rag_prediction

SQL_QUERY_MASTER = "SELECT GameCode FROM TRAININGGAME WHERE IsActive = 1;"
SQL_QUERY_LOG = "SELECT Id, GameCode, Periode, LogResult, [As], Kop, Kepala, Ekor, DateResultInGame FROM LogGame WHERE GameCode='{gameCode}' ORDER BY Periode DESC;"
SQL_QUERY_HISTORICAL = "SELECT LogResult FROM LogGame WHERE GameCode='{gameCode}' ORDER BY Periode DESC;"
from src.prediction_models import XGBoostModel, VAEPredictionModel

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_features_to_numbers(predicted_features_batch: np.ndarray, scaler, feature_keys: list) -> set:
    """
    Mengubah BATCH fitur yang diprediksi menjadi satu set angka 4-digit yang beragam.
    """
    if predicted_features_batch is None or predicted_features_batch.size == 0:
        return set()

    pred_df = pd.DataFrame(predicted_features_batch, columns=feature_keys)
    cols_for_number = ['As', 'Kop', 'Kepala', 'Ekor']
    if not all(c in pred_df.columns for c in cols_for_number):
        return {"0000"}

    inversed_features = scaler.inverse_transform(pred_df)
    inversed_df = pd.DataFrame(inversed_features, columns=feature_keys)

    digit_vals = [inversed_df[col].values for col in cols_for_number]
    floor_cands = [np.clip(np.floor(vals), 0, 9).astype(int) for vals in digit_vals]
    ceil_cands = [np.clip(np.ceil(vals), 0, 9).astype(int) for vals in digit_vals]

    all_combinations = set()
    for p in itertools.product([0, 1], repeat=4):
        cands = [floor_cands[i] if p[i] == 0 else ceil_cands[i] for i in range(4)]
        numbers = cands[0] * 1000 + cands[1] * 100 + cands[2] * 10 + cands[3]
        all_combinations.update(np.char.zfill(numbers.astype(str), 4))
        
    return all_combinations

def intelligent_augmentation_loop(predictions: set, historical_data: pd.DataFrame, target_count: int) -> set:
    """
    [FIXED] Loop augmentasi cerdas yang berhenti tepat waktu.
    """
    print("Memulai fase 'Intelligent Augmentation Loop'...")
    if historical_data.empty or 'LogResult' not in historical_data.columns:
        print("Peringatan: Data historis tidak valid. Melewatkan augmentasi cerdas.")
        return predictions

    historical_logs = set(historical_data['LogResult'])
    mutation_strategies = [[1, -1], [2, -1], [1, -1, 2, -2]]
    
    for i, increments in enumerate(mutation_strategies):
        if len(predictions) >= target_count:
            break
        print(f"Wave {i+1}/{len(mutation_strategies)}: Mutating with increments {increments}. Current predictions: {len(predictions)}")
        
        base_for_mutation = list(predictions.union(historical_logs))
        if not base_for_mutation:
            continue

        random.shuffle(base_for_mutation)

        # Loop dengan break internal untuk mencegah overshoot
        for number_str in base_for_mutation:
            if len(predictions) >= target_count:
                break
            if len(number_str) != 4 or not number_str.isdigit():
                continue
            
            digits = [int(d) for d in number_str]
            for digit_idx in range(len(digits)):
                if len(predictions) >= target_count:
                    break
                for inc in increments:
                    new_digits = digits[:]
                    new_digits[digit_idx] = (new_digits[digit_idx] + inc) % 10
                    new_number = "".join(map(str, new_digits))
                    predictions.add(new_number) # Set akan menangani duplikat
                    if len(predictions) >= target_count:
                        break

    print(f"Setelah Intelligent Augmentation, total prediksi unik: {len(predictions)}")
    return predictions

def heuristic_augmentation(existing_predictions: set, target_count: int) -> set:
    """
    [REFACTORED & FIXED] Jaring pengaman terakhir.
    """
    print(f"\nMemulai 'Heuristic Augmentation' (Loop) untuk memastikan {target_count} prediksi...")
    if not existing_predictions:
        print("Peringatan: Tidak ada prediksi dasar. Mengisi dengan angka acak.")
        while len(existing_predictions) < target_count:
            existing_predictions.add(f"{random.randint(0, 9999):04d}")
        return existing_predictions

    base_predictions = list(existing_predictions)
    safety_break = 0
    # Beri batas iterasi yang wajar untuk mencegah loop tak terbatas
    max_iterations = (target_count - len(existing_predictions)) * 20 + 1000

    while len(existing_predictions) < target_count and safety_break < max_iterations:
        sample = random.choice(base_predictions)
        digits = [int(d) for d in sample]
        pos_to_change = random.randint(0, 3)
        original_digit = digits[pos_to_change]
        new_digit = random.randint(0, 9)
        while new_digit == original_digit:
            new_digit = random.randint(0, 9)
            
        digits[pos_to_change] = new_digit
        new_prediction = "".join(map(str, digits))
        
        existing_predictions.add(new_prediction)
        safety_break += 1

    if len(existing_predictions) < target_count:
        print(f"Peringatan: Heuristic augmentation berhenti pada {safety_break} iterasi.")
    else:
        print(f"Heuristic augmentation berhasil. Total prediksi: {len(existing_predictions)}")
        
    return existing_predictions

def run_prediction_pipeline_for_game_code(game_code: str):
    """
    Menjalankan pipeline prediksi lengkap untuk satu game_code.
    """
    print(f"--- Memulai Pipeline Prediksi Gabungan untuk GameCode: {game_code} ---")

    game_docs_folder = os.path.join(DOCS_FOLDER, game_code)
    scaler_path = os.path.join(ARTIFACTS_DIR, f"{game_code}_scaler.joblib")
    feature_keys_path = os.path.join(ARTIFACTS_DIR, f"{game_code}_feature_keys.json")
    game_model_dir = os.path.join(TRAINED_FILES_DIR, game_code)

    try:
        query = SQL_QUERY_LOG.format(gameCode=game_code)
        df_raw = fetch_data_from_sql_server(SQL_CONNECTION_STRING, query)
        df_raw = df_raw.iloc[::-1].reset_index(drop=True)
        if df_raw.empty:
            print(f"Tidak ada data log untuk GameCode {game_code}. Melewatkan.")
            return
        print(f"Berhasil mengambil {len(df_raw)} baris data.")
    except Exception as e:
        print(f"Gagal mengambil data log untuk {game_code}: {e}")
        return

    try:
        processed_chunks = preprocess_data_for_embedding(df_raw, chunk_size=PIPELINE_WINDOW_SIZE, overlap=OVERLAP_SIZE, scaler_path=scaler_path, feature_keys_path=feature_keys_path)
        if processed_chunks:
            first_model_type = EMBEDDING_MODEL_TYPES[0]
            first_embedding_model_path = os.path.join(game_model_dir, f"{first_model_type}_model.pth")
            if not os.path.exists(first_embedding_model_path):
                 print(f"Peringatan: Model embedding awal '{first_model_type}' tidak ditemukan.")
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
    except Exception as e: print(f"Gagal memproses atau mengindeks data untuk RAG: {e}")
    
    all_prediction_sets = []
    prediction_model_types = ['LSTM', 'GRU', 'MLP', 'VAE', 'TCN']

    for embedding_model_type in EMBEDDING_MODEL_TYPES:
        for prediction_model_type in prediction_model_types:
            print(f"\n-- Menjalankan RAG dengan [Embedding: {embedding_model_type}, Analis: {prediction_model_type}] --")
            embedding_model_path = "" if embedding_model_type == "SemanticTransformer" else os.path.join(game_model_dir, f"{embedding_model_type}_model.pth")
            prediction_model_path = os.path.join(game_model_dir, f"{prediction_model_type}_model.pth")

            if not all(os.path.exists(p) for p in [prediction_model_path] + ([embedding_model_path] if embedding_model_path else [])):
                print(f"Error: Kombinasi model untuk '{game_code}' tidak ditemukan. Melewatkan.")
                continue

            try:
                scaler = joblib.load(scaler_path)
                with open(feature_keys_path, 'r') as f: feature_keys = json.load(f)
            except FileNotFoundError:
                print(f"Error: scaler.joblib atau feature_keys.json tidak ditemukan.")
                continue

            if len(df_raw) < PIPELINE_WINDOW_SIZE: continue
            current_query_data = df_raw.tail(PIPELINE_WINDOW_SIZE).copy()
            
            try:
                predicted_features_batch = run_rag_prediction(current_input_df=current_query_data, chroma_db_persist_dir=CHROMA_DB_PERSIST_DIR, chroma_collection_name=CHROMA_COLLECTION_NAME, embedding_model_path=embedding_model_path, prediction_model_path=prediction_model_path, scaler_path=scaler_path, feature_keys_path=feature_keys_path, embedding_model_type=embedding_model_type, prediction_model_type=prediction_model_type, n_retrieved_chunks=N_RETRIEVED_CHUNKS)
                if predicted_features_batch is not None and predicted_features_batch.size > 0:
                    predicted_numbers_set = decode_features_to_numbers(predicted_features_batch, scaler, feature_keys)
                    all_prediction_sets.append(predicted_numbers_set)
                    print(f"    -> Dihasilkan {len(predicted_numbers_set)} prediksi unik.")
            except Exception as e: print(f"Gagal menjalankan pipeline RAG: {e}")

    print("\n-- Menjalankan Analis Klasik: XGBoost --")
    try:
        xgb_model_path = os.path.join(game_model_dir, "XGBoost_models")
        if os.path.exists(xgb_model_path):
            xgb_model_wrapper = XGBoostModel()
            xgb_model_wrapper.load_model(xgb_model_path)
            X_xgb, _ = prepare_tabular_data(df_raw.tail(PIPELINE_WINDOW_SIZE))
            if not X_xgb.empty:
                features_for_prediction = X_xgb.tail(1)
                if not features_for_prediction.empty:
                    xgb_predicted_digits = xgb_model_wrapper.predict(features_for_prediction)
                    pred_str = "".join(map(str, np.clip(np.round(xgb_predicted_digits).astype(int), 0, 9)[0]))
                    all_prediction_sets.append({pred_str})
                    print(f"    -> Dihasilkan 1 prediksi unik dari XGBoost: {pred_str}")
    except Exception as e: print(f"Gagal menjalankan prediksi XGBoost: {e}")

    if not all_prediction_sets:
        print(f"Tidak ada prediksi yang dihasilkan untuk {game_code}.")
        return

    all_predictions_flat = [num for pred_set in all_prediction_sets for num in pred_set]
    Temp_Data_Predictions = set(item[0] for item in Counter(all_predictions_flat).most_common())
    print(f"\nTotal prediksi unik dari semua ahli: {len(Temp_Data_Predictions)}")

    # --- [REFACTORED] Tahap Augmentasi Bertingkat ---
    if len(Temp_Data_Predictions) < PREDICTION_HORIZON:
        Temp_Data_Predictions = intelligent_augmentation_loop(Temp_Data_Predictions, df_raw, PREDICTION_HORIZON)

    if len(Temp_Data_Predictions) < PREDICTION_HORIZON:
        print(f"\nJumlah prediksi ({len(Temp_Data_Predictions)}) masih kurang. Memulai fase 'Generative Augmentation' (VAE)...")
        try:
            scaler = joblib.load(scaler_path)
            with open(feature_keys_path, 'r') as f: feature_keys = json.load(f)
            vae_model_path = os.path.join(game_model_dir, "VAE_model.pth")
            if not os.path.exists(vae_model_path) or 'VAE' not in prediction_model_types:
                print("Model VAE tidak ditemukan, augmentasi VAE dibatalkan.")
            else:
                state_dict = torch.load(vae_model_path, map_location=device)
                input_dim, hidden_dim, output_dim = state_dict['encoder_fc1.weight'].shape[1], state_dict['encoder_fc1.bias'].shape[0], state_dict['decoder_fc2.bias'].shape[0]
                vae_model = VAEPredictionModel(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=VAE_LATENT_DIM, output_dim=output_dim)
                vae_model.load_state_dict(state_dict)
                vae_model.to(device)
                vae_model.eval()
                with torch.no_grad():
                    safety_break, noise_factor = 0, 1.0
                    while len(Temp_Data_Predictions) < PREDICTION_HORIZON and safety_break < 30:
                        batch_size = int((PREDICTION_HORIZON - len(Temp_Data_Predictions)) * 1.2) + 1
                        z = torch.randn(batch_size, VAE_LATENT_DIM).to(device) * noise_factor
                        generated_features = vae_model.decode(z)
                        new_predictions = decode_features_to_numbers(generated_features.cpu().numpy(), scaler, feature_keys)
                        # [FIXED] Add one-by-one to prevent overshoot
                        for pred in new_predictions:
                            if len(Temp_Data_Predictions) < PREDICTION_HORIZON:
                                Temp_Data_Predictions.add(pred)
                            else:
                                break
                        safety_break += 1
                        noise_factor += 0.1
                        print(f"Iterasi Augmentasi VAE {safety_break}: Total unik sekarang {len(Temp_Data_Predictions)}")
        except Exception as e: print(f"Gagal menjalankan Generative Augmentation: {e}")

    final_predictions_set = heuristic_augmentation(Temp_Data_Predictions, PREDICTION_HORIZON)
    
    final_predictions_list = sorted(list(final_predictions_set))[:PREDICTION_HORIZON]

    if len(final_predictions_list) >= PREDICTION_HORIZON:
        final_predictions_string = "*".join(final_predictions_list)
        try:
            save_predictions_to_db(SQL_CONNECTION_STRING, game_code, final_predictions_string)
            print(f"Prediksi baru untuk GameCode '{game_code}' berhasil disimpan ke database.")
        except Exception as e: print(f"Gagal menyimpan prediksi final ke database untuk {game_code}: {e}")
    else:
        print(f"GAGAL FINAL: Jumlah prediksi unik ({len(final_predictions_list)}) tetap tidak mencapai target {PREDICTION_HORIZON}. Tidak ada data yang disimpan.")

def main():
    """
    Fungsi utama untuk menjalankan pipeline prediksi untuk semua game code dari master.
    """
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
        try:
            print(f"Pipeline prediksi untuk {game_code} selesai. Mengupdate status di database...")
            update_training_game_status(SQL_CONNECTION_STRING, game_code)
        except Exception as e: print(f"Gagal mengupdate status untuk {game_code} setelah prediksi: {e}")

    print("\n--- Semua Proses Prediksi Selesai ---")

if __name__ == "__main__":
    main()