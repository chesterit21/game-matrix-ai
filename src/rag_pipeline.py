# src/rag_pipeline.py

import json
import os
import numpy as np
import pandas as pd
import torch
from src.chromadb_handler import initialize_chromadb
from src.embedding_models import generate_embedding_for_chunk
# Impor model dan fungsi prediksi yang sudah diupdate
from src.prediction_models import LSTMModel, predict_with_lstm
from sklearn.impute import SimpleImputer
import joblib
from .config import NUM_PREDICTION_SHOTS, PREDICTION_NOISE_LEVEL

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _preprocess_query_data(query_df: pd.DataFrame, scaler_path: str, feature_keys_path: str) -> dict:
    """Fungsi helper untuk pra-pemrosesan data query tunggal."""
    # Implementasi fungsi ini diasumsikan sudah benar dan tidak diubah
    df_processed = query_df.copy()
    if 'DateResultInGame' in df_processed.columns:
        cleaned_date_str = df_processed['DateResultInGame'].astype(str).str.replace(r'^\w+,\s*', '', regex=True)
        df_processed['Periode_DT'] = pd.to_datetime(cleaned_date_str, format='%d %b %Y', errors='coerce')
    elif 'Periode' in df_processed.columns:
        df_processed['Periode_DT'] = pd.to_datetime(df_processed['Periode'], unit='s', errors='coerce')
    else:
        raise ValueError("DataFrame harus memiliki kolom 'DateResultInGame' atau 'Periode'.")
    df_processed.dropna(subset=['Periode_DT'], inplace=True)
    df_processed.sort_values(by='Periode_DT', inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    df_processed['hour'] = df_processed['Periode_DT'].dt.hour
    df_processed['day_of_week'] = df_processed['Periode_DT'].dt.dayofweek
    df_processed['month'] = df_processed['Periode_DT'].dt.month
    df_processed['year'] = df_processed['Periode_DT'].dt.year
    if 'LogResult' in df_processed.columns:
        if df_processed['LogResult'].isnull().any():
            imputer_str = SimpleImputer(strategy='most_frequent')
            df_processed[['LogResult']] = imputer_str.fit_transform(df_processed[['LogResult']])
        df_processed['LogResult'] = df_processed['LogResult'].astype(str)
        df_processed['log_result_front'] = pd.to_numeric(df_processed['LogResult'].str[0:2], errors='coerce')
        df_processed['log_result_mid'] = pd.to_numeric(df_processed['LogResult'].str[1:3], errors='coerce')
        df_processed['log_result_back'] = pd.to_numeric(df_processed['LogResult'].str[2:4], errors='coerce')
        categorical_cols = ['log_result_front', 'log_result_mid', 'log_result_back']
        # Best Practice: Avoid 'inplace=True' on chained assignments to prevent FutureWarning.
        # Explicitly assign the result back to the DataFrame columns.
        for col in categorical_cols: df_processed[col] = df_processed[col].fillna(-1)
    numerical_cols = ['As', 'Kop', 'Kepala', 'Ekor', 'Id']
    imputer_num = SimpleImputer(strategy='mean')
    df_processed[numerical_cols] = imputer_num.fit_transform(df_processed[numerical_cols])
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    columns_to_drop = [col for col in ['LogResult', 'DateResultInGame', 'Periode', 'Periode_DT'] if col in df_processed.columns]
    features_df = df_processed.drop(columns=columns_to_drop)
    with open(feature_keys_path, 'r') as f: feature_keys = json.load(f)
    final_features_df = pd.DataFrame(columns=feature_keys)
    for col in feature_keys: final_features_df[col] = features_df.get(col, 0.0)
    query_chunk_sequence = final_features_df.to_dict(orient='records')
    return {
        "chunk_id": "query_chunk",
        "features_sequence": query_chunk_sequence
    }

def run_rag_prediction(
    current_input_df: pd.DataFrame,
    chroma_db_persist_dir: str,
    chroma_collection_name: str,
    embedding_model_path: str,
    prediction_model_path: str,
    scaler_path: str,
    feature_keys_path: str,
    embedding_model_type: str,
    n_retrieved_chunks: int = 20
) -> np.ndarray:
    """
    Menjalankan pipeline RAG menggunakan model embedding dan LSTM yang telah dilatih.
    """
    # 1. Inisialisasi ChromaDB untuk retrieval
    chroma_collection = initialize_chromadb(persist_directory=chroma_db_persist_dir, collection_name=chroma_collection_name)

    # 2. Pra-pemrosesan data query dan hasilkan embeddingnya
    query_document = _preprocess_query_data(current_input_df, scaler_path, feature_keys_path)
    query_embedding = generate_embedding_for_chunk(
        query_document, 
        model_type=embedding_model_type, 
        model_path=embedding_model_path,
        feature_keys_path=feature_keys_path
    )

    # 3. Ambil (retrieve) chunk historis yang relevan dari ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_retrieved_chunks
    )
    retrieved_documents_text_list = results.get("documents", [[]])[0]

    # 4. Gabungkan data query dengan data historis yang diambil
    with open(feature_keys_path, 'r') as f: feature_keys = json.load(f)
    
    all_retrieved_sequences = []
    for doc_text in retrieved_documents_text_list:
        try: all_retrieved_sequences.extend(json.loads(doc_text))
        except (json.JSONDecodeError, TypeError): continue

    query_numerical_sequence = [[step.get(key, 0.0) for key in feature_keys] for step in query_document["features_sequence"]]
    retrieved_numerical_sequence = [[step.get(key, 0.0) for key in feature_keys] for step in all_retrieved_sequences]
    
    # Urutan input untuk LSTM: data historis diikuti oleh data query terbaru
    # Pastikan tidak ada duplikasi antara data query dan data yang diambil
    combined_sequence = retrieved_numerical_sequence + query_numerical_sequence
    # Hapus duplikat sambil mempertahankan urutan (untuk Python 3.7+)
    unique_sequence = list(dict.fromkeys(map(tuple, combined_sequence)))
    augmented_input_sequence = np.array([list(item) for item in unique_sequence])

    if augmented_input_sequence.size == 0:
        print("Peringatan: Tidak ada data input yang cukup untuk prediksi setelah retrieval.")
        return np.array([])

    # 5. Muat LSTM model yang sudah dilatih
    state_dict = torch.load(prediction_model_path, map_location=device)
    input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
    hidden_dim = state_dict['lstm.weight_hh_l0'].shape[1]
    output_dim = state_dict['fc.weight'].shape[0]
    num_layers = len([key for key in state_dict if 'lstm.weight_ih_l' in key])
    
    prediction_model = LSTMModel(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim, 
        num_layers=num_layers
    )
    prediction_model.load_state_dict(state_dict)
    # Best Practice: Pindahkan model prediksi ke perangkat yang benar (GPU/CPU).
    prediction_model.to(device)
    
    # 6. Lakukan prediksi menggunakan LSTM
    # Fungsi predict_with_lstm akan mengembalikan fitur-fitur dari timestep selanjutnya
    # --- Implementasi Multi-Shot Prediction ---
    base_input_tensor = torch.tensor(augmented_input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    all_predicted_features = []

    with torch.no_grad():
        for shot in range(NUM_PREDICTION_SHOTS):
            # 1. Buat input yang sedikit berbeda untuk setiap tembakan dengan menambahkan noise
            if shot > 0:
                # Gunakan noise yang lebih kecil untuk input agar tidak terlalu menyimpang dari konteks RAG
                input_noise = torch.randn_like(base_input_tensor) * (PREDICTION_NOISE_LEVEL / 5)
                live_input_tensor = base_input_tensor + input_noise
            else:
                live_input_tensor = base_input_tensor

            # 2. Dapatkan prediksi untuk tembakan ini
            # Kita tidak memanggil predict_with_lstm karena kita sudah punya modelnya di sini
            prediction_output = prediction_model(live_input_tensor)
            all_predicted_features.append(prediction_output.squeeze(0).cpu().numpy())

    # Kembalikan sebagai batch numpy array
    # Shape: (NUM_PREDICTION_SHOTS, num_features)
    return np.array(all_predicted_features)