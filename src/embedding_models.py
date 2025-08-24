# src/embedding_models.py

import numpy as np
import torch
import torch.nn as nn
import math
import json
import os

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Time2Vec(nn.Module):
    """Implementasi layer Time2Vec."""
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], -1)

class Time2VecAutoencoder(nn.Module):
    """Model Autoencoder yang menggabungkan fitur asli dengan embedding Time2Vec."""
    def __init__(self, num_features: int, sequence_length: int, latent_dim: int, t2v_out_dim: int = 32):
        super(Time2VecAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.t2v_out_dim = t2v_out_dim
        self.time2vec = Time2Vec(in_features=1, out_features=t2v_out_dim)
        
        encoder_input_dim = (num_features + t2v_out_dim) * sequence_length
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, encoder_input_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_input_dim // 2, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_input_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_input_dim // 2, encoder_input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        tau = torch.arange(self.sequence_length, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        tau = tau.repeat(batch_size, 1, 1)
        time_embedding = self.time2vec(tau)
        combined_features = torch.cat([x, time_embedding], dim=-1)
        flat_features = combined_features.view(batch_size, -1)
        latent_representation = self.encoder(flat_features)
        reconstructed_flat = self.decoder(latent_representation)
        return latent_representation, reconstructed_flat

# --- Model 2: DWT + Autoencoder (Placeholder) ---
class DWTAutoencoder(nn.Module):
    """Implementasi Autoencoder standar."""
    def __init__(self, input_dim, latent_dim):
        super(DWTAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstructed_x = self.decoder(latent_representation)
        return latent_representation, reconstructed_x

# --- Fungsi Generator Embedding --- 
def generate_embedding_for_chunk(
    chunk_data: dict, 
    model_type: str, 
    model_path: str,
    feature_keys_path: str
) -> np.ndarray:
    """
    Menghasilkan embedding untuk satu chunk data menggunakan model yang sesuai pada perangkat yang ditentukan.
    """
    features_sequence = chunk_data.get("features_sequence", [])
    if not features_sequence:
        print(f"Peringatan: 'features_sequence' kosong. Mengembalikan embedding nol.")
        return np.zeros(128, dtype=np.float32)

    feature_keys = []
    if os.path.exists(feature_keys_path):
        with open(feature_keys_path, 'r') as f:
            feature_keys = json.load(f)
    else:
        all_keys = set().union(*(d.keys() for d in features_sequence))
        feature_keys = sorted(list(all_keys))

    numerical_sequence = [[step.get(key, 0.0) for key in feature_keys] for step in features_sequence]
    # 1. Pindahkan tensor input ke perangkat (CUDA/CPU)
    input_tensor = torch.tensor(numerical_sequence, dtype=torch.float32).to(device)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: File model tidak ditemukan di '{model_path}'. Jalankan train.py terlebih dahulu.")

    # 2. Muat model langsung ke perangkat yang ditentukan
    state_dict = torch.load(model_path, map_location=device)
    
    model = None
    if model_type == "Time2Vec":
        latent_dim = state_dict['encoder.2.bias'].shape[0]
        sequence_length = len(features_sequence)
        t2v_out_dim = state_dict['time2vec.w'].shape[1] + 1
        original_num_features = (state_dict['encoder.0.weight'].shape[1] // sequence_length) - t2v_out_dim
        model = Time2VecAutoencoder(num_features=original_num_features, sequence_length=sequence_length, latent_dim=latent_dim, t2v_out_dim=t2v_out_dim)
    
    elif model_type == "DWT_Autoencoder":
        input_dim = state_dict['encoder.0.weight'].shape[1]
        latent_dim = state_dict['encoder.2.bias'].shape[0]
        model = DWTAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    
    else:
        raise ValueError(f"Tipe model tidak dikenal: {model_type}")

    model.load_state_dict(state_dict)
    # 3. Best Practice: Pindahkan model ke perangkat yang benar (GPU/CPU) setelah dimuat.
    model.to(device)
    model.eval()

    with torch.no_grad():
        if model_type == "Time2Vec":
            embedding_result, _ = model(input_tensor.unsqueeze(0))
        else:
            flat_input_tensor = input_tensor.view(1, -1)
            embedding_result, _ = model(flat_input_tensor)
            
        # 4. Pindahkan hasil kembali ke CPU sebelum konversi ke NumPy
        embedding_result = embedding_result.squeeze(0).cpu().numpy()

    return embedding_result.astype(np.float32)
