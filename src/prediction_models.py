# src/prediction_models.py

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import weight_norm
import pandas as pd
import xgboost as xgb
import os

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

class GRUModel(nn.Module):
    """
    Model GRU (Gated Recurrent Unit) untuk prediksi deret waktu.
    Alternatif yang lebih ringan dari LSTM, seringkali lebih cepat dengan performa yang sebanding.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout_prob: float = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layer GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Layer Linear
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Inisialisasi hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass melalui GRU
        gru_out, _ = self.gru(x, h0)
        
        # Ambil output dari timestep terakhir dan lewatkan ke layer fully connected
        out = self.fc(gru_out[:, -1, :])
        return out

class MLPModel(nn.Module):
    """
    Model MLP (Multi-Layer Perceptron) untuk prediksi deret waktu.
    Model ini mem-flatten sekuens input dan menggunakannya sebagai fitur untuk prediksi.
    Berguna sebagai baseline atau untuk menangkap hubungan non-sekuensial.
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(MLPModel, self).__init__()
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the sequence input if it's not already flat
        # x shape: [batch_size, seq_len, num_features]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

class VAEPredictionModel(nn.Module):
    """
    Model VAE (Variational Autoencoder) untuk prediksi generatif.
    Model ini belajar distribusi probabilitas dari data, memungkinkan kita untuk
    men-sample prediksi yang beragam dan baru.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int):
        super(VAEPredictionModel, self).__init__()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim) # Untuk mean
        self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim) # Untuk log variance
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()

    def encode(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.relu(self.encoder_fc1(x))
        return self.encoder_fc_mu(h), self.encoder_fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h)) # Sigmoid untuk menormalkan output ke [0,1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # VAE loss = Reconstruction Loss + KL Divergence
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class TemporalBlock(nn.Module):
    """Blok fundamental untuk TCN, terdiri dari dua lapisan konvolusi kausal."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv1 = weight_norm(self.conv1)
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = weight_norm(self.conv2)
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    """
    Model TCN (Temporal Convolutional Network) untuk prediksi deret waktu.
    Menggunakan konvolusi kausal yang di-dilatasi untuk menangkap dependensi temporal.
    """
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # TCN expects input of shape (N, C_in, L_in)
        x = x.permute(0, 2, 1)
        y = self.network(x)
        return self.linear(y[:, :, -1])

class PatternClassifier(nn.Module):
    """
    Model klasifikasi sederhana untuk memprediksi digit berdasarkan pola.
    Menggunakan Entity Embedding untuk input kategori (pola).
    """
    def __init__(self, num_patterns: int, embedding_dim: int, output_dim: int):
        super(PatternClassifier, self).__init__()
        self.embedding = nn.Embedding(num_patterns, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), # *2 karena ada 2 pola (depan & belakang)
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x_pattern_front, x_pattern_back):
        embedded_front = self.embedding(x_pattern_front)
        embedded_back = self.embedding(x_pattern_back)
        combined = torch.cat([embedded_front, embedded_back], dim=1)
        return self.network(combined)


# Wrapper untuk XGBoost akan ditangani di luar kelas nn.Module,
# biasanya menggunakan library seperti `xgboost` atau `lightgbm`.
class XGBoostModel:
    """
    Wrapper class for the XGBoost model to integrate into our pipeline.
    This version handles multi-output regression by training a separate
    model for each target digit.
    """
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist', 
                'device': 'cuda' # Enable GPU acceleration
            }
        else:
            self.params = params
        #self.model = xgb.XGBRegressor(**self.params)
        self.models = {}

    def train(self, X_train, y_train):
        """
        Trains a separate XGBoost model for each column in y_train.
        y_train is expected to be a pandas DataFrame.
        """
        for col in y_train.columns:
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_train, y_train[col], verbose=False)
            self.models[col] = model

    def predict(self, X_test):
        """
        Predicts each target and combines them into a single numpy array.
        """
        predictions = {}
        for col, model in self.models.items():
            predictions[col] = model.predict(X_test)
        # Return as a DataFrame first, then convert to numpy
        return pd.DataFrame(predictions).values

    def save_model(self, path):
        """
        Saves each individual model to a file within the given directory path.
        """
        os.makedirs(path, exist_ok=True)
        for col, model in self.models.items():
            if model:
                model.save_model(os.path.join(path, f"xgb_model_{col}.json"))

    def load_model(self, path):
        """
        Loads all individual models from a directory.
        """
        self.models = {}
        if not os.path.isdir(path):
            print(f"Warning: XGBoost model path is not a directory: {path}")
            return
        for f in os.listdir(path):
            if f.startswith("xgb_model_") and f.endswith(".json"):
                col_name = f.replace("xgb_model_", "").replace(".json", "")
                model = xgb.XGBRegressor(**self.params)
                model.load_model(os.path.join(path, f))
                self.models[col_name] = model

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

# --- Generative Adversarial Network (Placeholder) ---
class GANGenerator(nn.Module):
    """
    Bagian Generator dari GAN. Mengambil noise acak dan menghasilkan urutan fitur.
    Ini adalah placeholder untuk implementasi yang lebih kompleks (misalnya, menggunakan lapisan LSTM/Conv1D).
    """
    def __init__(self, latent_dim: int, seq_len: int, feature_dim: int):
        super(GANGenerator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, seq_len * feature_dim),
            nn.Tanh() # Tanh umum untuk GAN untuk menskalakan output ke [-1, 1]
        )

    def forward(self, z):
        img_flat = self.network(z)
        # Ubah bentuk menjadi (batch_size, seq_len, feature_dim)
        return img_flat.view(img_flat.size(0), self.seq_len, self.feature_dim)

class GANDiscriminator(nn.Module):
    """
    Bagian Diskriminator dari GAN. Mengambil urutan fitur dan mengklasifikasikannya sebagai asli atau palsu.
    """
    def __init__(self, seq_len: int, feature_dim: int):
        super(GANDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(seq_len * feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Menghasilkan probabilitas
        )

    def forward(self, seq):
        seq_flat = seq.view(seq.size(0), -1)
        return self.network(seq_flat)

class GANModel:
    """
    Wrapper untuk GAN yang akan digunakan untuk generasi. Kita hanya memerlukan generator untuk prediksi.
    Logika training akan lebih kompleks dan berada di train.py.
    """
    def __init__(self, latent_dim, seq_len, feature_dim):
        self.generator = GANGenerator(latent_dim, seq_len, feature_dim).to(device)

    def generate(self, num_samples: int, latent_dim: int):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, latent_dim).to(device)
            generated_sequences = self.generator(noise)
            # Output dari Tanh adalah [-1, 1]. Pre-processing menggunakan MinMaxScaler [0, 1].
            # Kita perlu menskalakan kembali outputnya.
            return (generated_sequences + 1) / 2

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(path, "gan_generator.pth"))

    def load_model(self, path):
        generator_path = os.path.join(path, "gan_generator.pth")
        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=device))