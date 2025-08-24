# src/prediction_models.py

import torch
import torch.nn as nn
import numpy as np

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Prediksi Berbasis LSTM ---
class LSTMModel(nn.Module):
    """
    Model LSTM (Long Short-Term Memory) untuk prediksi deret waktu.
    Ini adalah arsitektur yang jauh lebih kuat untuk menangkap pola sekuensial
    dibandingkan dengan model linear sederhana.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout_prob: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layer LSTM: memproses sekuens input
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Memudahkan penanganan dimensi tensor
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Layer Linear: memetakan output LSTM ke dimensi prediksi yang diinginkan
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Inisialisasi hidden state dan cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass melalui LSTM
        # lstm_out berisi output dari setiap timestep
        # (h_n, c_n) adalah hidden dan cell state terakhir
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Kita hanya tertarik pada output dari timestep terakhir untuk membuat prediksi
        last_timestep_out = lstm_out[:, -1, :]
        
        # Lewatkan output terakhir ke layer fully connected
        out = self.fc(last_timestep_out)
        return out


def predict_with_lstm(
    model: LSTMModel,
    input_sequence: np.ndarray, # Berbentuk (sequence_length, num_features)
    prediction_horizon: int = 1, # Biasanya, kita prediksi satu langkah ke depan dalam loop
) -> np.ndarray:
    """
    Melakukan prediksi satu langkah ke depan menggunakan model LSTM.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Pindahkan input ke perangkat dan tambahkan dimensi batch
        current_input = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Dapatkan prediksi dari model
        prediction_output = model(current_input)
        
        # Pindahkan hasil kembali ke CPU untuk operasi selanjutnya
        return prediction_output.squeeze(0).cpu().numpy()