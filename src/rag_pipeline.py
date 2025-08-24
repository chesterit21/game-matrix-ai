# src/rag_pipeline.py

import json
import os
import numpy as np
import pandas as pd
import torch
from src.embedding_models import generate_embedding_for_chunk
from src.chromadb_handler import initialize_chromadb, store_embeddings_in_chromadb
# Impor model dan fungsi prediksi yang sudah diupdate
from src.prediction_models import *
from sklearn.impute import SimpleImputer
import joblib
from collections import Counter
from src.data_ingestion import fetch_data_from_sql_server, preprocess_data_for_embedding, save_documents_to_folder, save_predictions_to_db, update_training_game_status
from src.config import *

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_rag_prediction(
    current_input_df: pd.DataFrame,
    chroma_db_persist_dir: str,
    chroma_collection_name: str,
    embedding_model_path: str,
    prediction_model_path: str,
    scaler_path: str,
    feature_keys_path: str,
    embedding_model_type: str,
    prediction_model_type: str,
    n_retrieved_chunks: int = 20
) -> np.ndarray:
    """
    Menjalankan pipeline RAG menggunakan model embedding dan prediksi yang telah dilatih.
    """
    # --- Special Case for VAE: Pure Generative Sampling ---
    if prediction_model_type == 'VAE':
        print("    -> Menggunakan VAE untuk generasi sampel acak...")
        state_dict = torch.load(prediction_model_path, map_location=device)
        input_dim = state_dict['encoder_fc1.weight'].shape[1]
        hidden_dim = state_dict['encoder_fc1.bias'].shape[0]
        latent_dim = state_dict['encoder_fc_mu.bias'].shape[0]
        output_dim = state_dict['decoder_fc2.bias'].shape[0]
        prediction_model = VAEPredictionModel(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, output_dim=output_dim)
        prediction_model.load_state_dict(state_dict)
        prediction_model.to(device)
        prediction_model.eval()
        with torch.no_grad():
            # Generate a large batch of samples to ensure diversity
            num_samples = NUM_PREDICTION_SHOTS * 100 
            z = torch.randn(num_samples, VAE_LATENT_DIM).to(device)
            generated_features = prediction_model.decode(z)
            return generated_features.cpu().numpy()

    # 1. Inisialisasi ChromaDB untuk retrieval
    chroma_collection = initialize_chromadb(persist_directory=chroma_db_persist_dir, collection_name=chroma_collection_name)

    # 2. Pra-pemrosesan data query dan hasilkan embeddingnya
    # Gunakan pipeline preprocessing yang sama dengan training untuk konsistensi
    processed_query_chunks = preprocess_data_for_embedding(current_input_df, chunk_size=len(current_input_df), overlap=0, scaler_path=scaler_path, feature_keys_path=feature_keys_path)
    if not processed_query_chunks:
        raise ValueError("Gagal memproses data query untuk RAG.")
    query_document = processed_query_chunks[0]
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
        try: 
            # The stored document is a JSON string of a list of dicts
            sequence_list = json.loads(doc_text)
            # Convert list of dicts to list of lists
            numerical_sequence = [[step.get(key, 0.0) for key in feature_keys] for step in sequence_list]
            all_retrieved_sequences.extend(numerical_sequence)
        except (json.JSONDecodeError, TypeError): 
            continue

    query_numerical_sequence = [[step.get(key, 0.0) for key in feature_keys] for step in query_document["features_sequence"]]
    
    # Urutan input: data historis yang relevan diikuti oleh data query terbaru
    combined_sequence = all_retrieved_sequences + query_numerical_sequence
    
    # Hapus duplikat sambil mempertahankan urutan (untuk Python 3.7+)
    # Convert lists to tuples to make them hashable for dict.fromkeys
    unique_sequence_tuples = list(dict.fromkeys(map(tuple, combined_sequence)))
    augmented_input_sequence = np.array([list(item) for item in unique_sequence_tuples])

    if augmented_input_sequence.size == 0:
        print("Peringatan: Tidak ada data input yang cukup untuk prediksi setelah retrieval.")
        return np.array([])
    # 5. Muat LSTM model yang sudah dilatih
    state_dict = torch.load(prediction_model_path, map_location=device)
    
    if prediction_model_type == 'LSTM':
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
    elif prediction_model_type == 'GRU':
        input_dim = state_dict['gru.weight_ih_l0'].shape[1]
        hidden_dim = state_dict['gru.weight_hh_l0'].shape[1]
        output_dim = state_dict['fc.weight'].shape[0]
        num_layers = len([key for key in state_dict if 'gru.weight_ih_l' in key])
        prediction_model = GRUModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            num_layers=num_layers
        )
    elif prediction_model_type == 'MLP':
        # Infer MLP dimensions from its state_dict
        input_dim = state_dict['network.0.weight'].shape[1]
        hidden_dims = [state_dict['network.0.bias'].shape[0], state_dict['network.3.bias'].shape[0]]
        output_dim = state_dict['network.6.bias'].shape[0]
        prediction_model = MLPModel(
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim
        )
    elif prediction_model_type == 'TCN':
        # Best Practice: Infer from a more stable key. The downsample layer is reliable.
        # If it doesn't exist, it means input_dim == num_channels[0], but that's an edge case.
        input_dim = state_dict['network.0.downsample.weight'].shape[1]
        output_dim = state_dict['linear.bias'].shape[0]
        prediction_model = TCNModel(
            input_dim=input_dim, output_dim=output_dim, 
            num_channels=TCN_NUM_CHANNELS, kernel_size=TCN_KERNEL_SIZE, 
            dropout=TCN_DROPOUT
        )
    else:
        raise ValueError(f"Tipe model prediksi tidak dikenal: {prediction_model_type}")

    prediction_model.load_state_dict(state_dict)
    # Best Practice: Pindahkan model prediksi ke perangkat yang benar (GPU/CPU).
    prediction_model.to(device)
    
    # 6. Lakukan prediksi menggunakan model yang sesuai
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
            if prediction_model_type == 'MLP':
                # MLP memerlukan input dengan ukuran tetap.
                # Ambil data terbaru sejumlah (PIPELINE_WINDOW_SIZE - 1)
                required_len = PIPELINE_WINDOW_SIZE - 1
                if live_input_tensor.shape[1] < required_len:
                    # Pad if necessary, though RAG should provide enough context.
                    padding_needed = required_len - live_input_tensor.shape[1]
                    padding = torch.zeros(1, padding_needed, live_input_tensor.shape[2]).to(device)
                    mlp_input = torch.cat([padding, live_input_tensor], dim=1)
                else:
                    mlp_input = live_input_tensor[:, -required_len:, :]
                prediction_output = prediction_model(mlp_input)
            else: # LSTM, GRU, TCN
                prediction_output = prediction_model(live_input_tensor)
            all_predicted_features.append(prediction_output.squeeze(0).cpu().numpy())

    # Kembalikan sebagai batch numpy array
    # Shape: (NUM_PREDICTION_SHOTS, num_features)
    return np.array(all_predicted_features)