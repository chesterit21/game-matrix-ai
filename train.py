# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import joblib
import random
import itertools
from src.config import *
from src.data_ingestion import fetch_data_from_sql_server, preprocess_data_for_embedding, update_training_game_status, prepare_tabular_data
from src.embedding_models import Time2VecAutoencoder, DWTAutoencoder, TransformerAutoencoder
from src.prediction_models import LSTMModel, GRUModel, MLPModel, VAEPredictionModel, vae_loss_function, TCNModel, XGBoostModel
from sklearn.metrics import accuracy_score

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

# --- Konfigurasi Augmentasi Data Training ---
# Faktor untuk augmentasi: 1.0 = tidak ada augmentasi, 2.0 = gandakan dataset
TRAINING_AUGMENTATION_FACTOR = 2.0

SQL_QUERY_MASTER = "SELECT GameCode FROM TRAININGGAME WHERE IsActive = 1."
SQL_QUERY_LOG = "SELECT Id, GameCode, Periode, LogResult, [As], Kop, Kepala, Ekor,DateResultInGame FROM LogGame WHERE GameCode='{gameCode}' AND Periode > (SELECT MAX(Periode) FROM LogGame WHERE GameCode = '{gameCode}')-365 ORDER BY Periode ASC;"

def augment_training_data(historical_data: pd.DataFrame, augmentation_factor: float) -> pd.DataFrame:
    """
    Menghasilkan data training sintetis menggunakan logika augmentasi cerdas.
    """
    if augmentation_factor <= 1.0 or historical_data.empty or 'LogResult' not in historical_data.columns:
        return pd.DataFrame()

    num_original = len(historical_data)
    target_count = int(num_original * augmentation_factor)
    num_to_generate = target_count - num_original

    if num_to_generate <= 0:
        return pd.DataFrame()

    print(f"Memulai augmentasi data training. Target: {num_to_generate} baris data sintetis.")

    base_data = set(historical_data['LogResult'].dropna())
    if not base_data:
        print("Peringatan: Tidak ada data 'LogResult' valid untuk dijadikan basis augmentasi.")
        return pd.DataFrame()

    generated_data = set()
    mutation_strategies = [[1, -1], [2, -1], [1, -1, 2, -2]]
    
    while len(generated_data) < num_to_generate:
        for increments in mutation_strategies:
            if len(generated_data) >= num_to_generate:
                break
            
            base_for_mutation = list(base_data.union(generated_data))
            random.shuffle(base_for_mutation)

            for number_str in base_for_mutation:
                if len(generated_data) >= num_to_generate:
                    break
                if len(number_str) != 4 or not number_str.isdigit():
                    continue
                
                digits = [int(d) for d in number_str]
                pos_to_change = random.randint(0, 3)
                inc = random.choice(increments)
                
                new_digits = digits[:]
                new_digits[pos_to_change] = (new_digits[pos_to_change] + inc) % 10
                new_number = "".join(map(str, new_digits))
                
                if new_number not in base_data and new_number not in generated_data:
                    generated_data.add(new_number)

    print(f"Berhasil menghasilkan {len(generated_data)} data sintetis unik.")

    augmented_rows = []
    for num_str in generated_data:
        augmented_rows.append({
            'LogResult': num_str,
            'As': int(num_str[0]),
            'Kop': int(num_str[1]),
            'Kepala': int(num_str[2]),
            'Ekor': int(num_str[3]),
            'Periode': -1, 
            'GameCode': historical_data['GameCode'].iloc[0] if not historical_data.empty else 'AUGMENTED'
        })
    
    return pd.DataFrame(augmented_rows)

class OnlineTrainingManager:
    """Mengelola proses training online sekuensial untuk satu GameCode."""

    def __init__(self, game_code: str, full_data_df: pd.DataFrame):
        self.game_code = game_code
        self.full_data_df = full_data_df
        self.training_log = []

        self.scaler_path = os.path.join(ARTIFACTS_DIR, f"{self.game_code}_scaler.joblib")
        self.feature_keys_path = os.path.join(ARTIFACTS_DIR, f"{self.game_code}_feature_keys.json")

        self.scaler = None
        self.feature_keys = None

        self._initialize_models()

    def _initialize_models(self):
        """Inisialisasi semua model, optimizer, dan criterion."""
        self.models = {}
        self.optimizers = {}
        self.criterions = {}

        if not self.feature_keys:
            print("Membuat feature_keys awal...")
            preprocess_data_for_embedding(self.full_data_df.head(PIPELINE_WINDOW_SIZE), PIPELINE_WINDOW_SIZE, 0, self.scaler_path, self.feature_keys_path)
            self.scaler = joblib.load(self.scaler_path)
            with open(self.feature_keys_path, 'r') as f: self.feature_keys = json.load(f)

        num_features = len(self.feature_keys)

        for model_type in EMBEDDING_MODEL_TYPES:
            if model_type == "Time2Vec":
                self.models[model_type] = Time2VecAutoencoder(num_features, PIPELINE_WINDOW_SIZE, EMBEDDING_DIM).to(device)
            elif model_type == "TransformerAutoencoder":
                self.models[model_type] = TransformerAutoencoder(num_features=num_features, sequence_length=PIPELINE_WINDOW_SIZE, latent_dim=EMBEDDING_DIM, nhead=TRANSFORMER_NHEAD, dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, num_layers=LSTM_NUM_LAYERS).to(device)
            elif model_type == "DWT_Autoencoder":
                self.models[model_type] = DWTAutoencoder(PIPELINE_WINDOW_SIZE * num_features, EMBEDDING_DIM).to(device)
            else:
                continue
            
            self.optimizers[model_type] = optim.Adam(self.models[model_type].parameters(), lr=LEARNING_RATE)
            self.criterions[model_type] = nn.MSELoss()

        self.models['LSTM'] = LSTMModel(input_dim=num_features, hidden_dim=LSTM_HIDDEN_DIM, output_dim=num_features, num_layers=LSTM_NUM_LAYERS).to(device)
        self.optimizers['LSTM'] = optim.Adam(self.models['LSTM'].parameters(), lr=LEARNING_RATE)
        self.criterions['LSTM'] = nn.MSELoss()

        self.models['GRU'] = GRUModel(input_dim=num_features, hidden_dim=GRU_HIDDEN_DIM, output_dim=num_features, num_layers=LSTM_NUM_LAYERS).to(device)
        self.optimizers['GRU'] = optim.Adam(self.models['GRU'].parameters(), lr=LEARNING_RATE)
        self.criterions['GRU'] = nn.MSELoss()

        mlp_input_dim = num_features * (PIPELINE_WINDOW_SIZE - 1)
        self.models['MLP'] = MLPModel(input_dim=mlp_input_dim, hidden_dims=MLP_HIDDEN_DIMS, output_dim=num_features).to(device)
        self.optimizers['MLP'] = optim.Adam(self.models['MLP'].parameters(), lr=LEARNING_RATE)
        self.criterions['MLP'] = nn.MSELoss()

        vae_input_dim = num_features * (PIPELINE_WINDOW_SIZE - 1)
        self.models['VAE'] = VAEPredictionModel(input_dim=vae_input_dim, hidden_dim=MLP_HIDDEN_DIMS[0], latent_dim=VAE_LATENT_DIM, output_dim=num_features).to(device)
        self.optimizers['VAE'] = optim.Adam(self.models['VAE'].parameters(), lr=LEARNING_RATE)
        self.criterions['VAE'] = vae_loss_function

        self.models['TCN'] = TCNModel(input_dim=num_features, output_dim=num_features, num_channels=TCN_NUM_CHANNELS, kernel_size=TCN_KERNEL_SIZE, dropout=TCN_DROPOUT).to(device)
        self.optimizers['TCN'] = optim.Adam(self.models['TCN'].parameters(), lr=LEARNING_RATE)
        self.criterions['TCN'] = nn.MSELoss()

        self.models['XGBoost'] = XGBoostModel()

        self.prediction_model_types = ['LSTM', 'GRU', 'MLP', 'VAE', 'TCN', 'XGBoost']

        print(f"Semua model untuk {self.game_code} telah diinisialisasi.")

    def _decode_predicted_features_to_number(self, predicted_features_batch: torch.Tensor) -> set[str]:
        """
        Mendekode satu batch tensor fitur yang diprediksi menjadi sekumpulan nomor 4 digit unik.
        """
        pred_df = pd.DataFrame(predicted_features_batch.cpu().numpy(), columns=self.feature_keys)
        cols_for_number = ['As', 'Kop', 'Kepala', 'Ekor']
        if not all(c in pred_df.columns for c in cols_for_number):
            return {"0000"}

        inversed_features = self.scaler.inverse_transform(pred_df)
        inversed_df = pd.DataFrame(inversed_features, columns=self.feature_keys)

        as_vals, kop_vals, kepala_vals, ekor_vals = inversed_df['As'].values, inversed_df['Kop'].values, inversed_df['Kepala'].values, inversed_df['Ekor'].values

        predicted_numbers = set()
        for i in range(len(inversed_df)):
            as_candidates = {int(np.clip(f(as_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}
            kop_candidates = {int(np.clip(f(kop_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}
            kepala_candidates = {int(np.clip(f(kepala_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}
            ekor_candidates = {int(np.clip(f(ekor_vals[i]), 0, 9)) for f in (np.floor, np.ceil)}

            for a, k, ke, e in itertools.product(as_candidates, kop_candidates, kepala_candidates, ekor_candidates):
                predicted_numbers.add(f"{a}{k}{ke}{e}")

        return predicted_numbers

    def _train_step(self, current_window_df: pd.DataFrame, epoch_multiplier: int = 1):
        """Lakukan satu langkah training (fine-tuning) pada semua model."""
        full_processed_chunks = preprocess_data_for_embedding(current_window_df, chunk_size=len(current_window_df), overlap=0, scaler_path=self.scaler_path, feature_keys_path=self.feature_keys_path)
        
        if not full_processed_chunks:
            return
        
        full_sequence_data = [list(step.values()) for step in full_processed_chunks[0]['features_sequence']]
        full_window_tensor = torch.tensor([full_sequence_data], dtype=torch.float32).to(device)

        standard_window_df = current_window_df.tail(PIPELINE_WINDOW_SIZE)
        if len(standard_window_df) >= PIPELINE_WINDOW_SIZE:
            std_processed_chunks = preprocess_data_for_embedding(standard_window_df, chunk_size=len(standard_window_df), overlap=0, scaler_path=self.scaler_path, feature_keys_path=self.feature_keys_path)
            std_sequence_data = [list(step.values()) for step in std_processed_chunks[0]['features_sequence']]
            std_window_tensor = torch.tensor([std_sequence_data], dtype=torch.float32).to(device)

            for model_type in EMBEDDING_MODEL_TYPES:
                model, optimizer, criterion = self.models[model_type], self.optimizers[model_type], self.criterions[model_type]
                model.train()
                for _ in range(ONLINE_TRAINING_EPOCHS):
                    optimizer.zero_grad()
                    if model_type == "Time2Vec":
                        _, reconstructed = model(std_window_tensor)
                        with torch.no_grad():
                            t2v_model, batch_size, seq_len = model.time2vec, std_window_tensor.shape[0], std_window_tensor.shape[1]
                            tau = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
                            time_embedding = t2v_model(tau)
                            target = torch.cat([std_window_tensor, time_embedding], dim=-1).view(batch_size, -1)
                    elif model_type == "TransformerAutoencoder":
                        _, reconstructed = model(std_window_tensor)
                        target = std_window_tensor.view(1, -1)
                    else:
                        flat_window = std_window_tensor.view(1, -1)
                        _, reconstructed = model(flat_window)
                        target = flat_window
                    loss = criterion(reconstructed, target)
                    loss.backward()
                    optimizer.step()

        X_full_seq = full_window_tensor[:, :-1, :]
        y = full_window_tensor[:, -1, :]
        for pred_model_type in self.prediction_model_types:
            if pred_model_type == 'XGBoost': continue
            pred_model, pred_optimizer, pred_criterion = self.models[pred_model_type], self.optimizers[pred_model_type], self.criterions[pred_model_type]
            pred_model.train()
            for _ in range(ONLINE_TRAINING_EPOCHS):
                pred_optimizer.zero_grad()
                if pred_model_type in ['MLP', 'VAE']:
                    X_fixed = X_full_seq[:, -(PIPELINE_WINDOW_SIZE - 1):, :]
                    outputs = pred_model(X_fixed)
                    loss = pred_criterion(outputs, y, *pred_model.get_params() if pred_model_type == 'VAE' else [])
                else:
                    outputs = pred_model(X_full_seq)
                    loss = pred_criterion(outputs, y)
                loss.backward()
                pred_optimizer.step()
        
        try:
            X_xgb, y_xgb = prepare_tabular_data(current_window_df)
            if not X_xgb.empty and not y_xgb.empty:
                self.models['XGBoost'].train(X_xgb, y_xgb)
        except Exception as e: print(f"Gagal melatih XGBoost: {e}")

    def run_online_training(self):
        """Menjalankan seluruh proses training online sekuensial."""
        print(f"\nMemulai training sekuensial untuk {self.game_code}...")
        real_data_df = self.full_data_df[self.full_data_df['Periode'] != -1].copy().reset_index(drop=True)
        num_rows = len(real_data_df)

        for window_end_idx in range(PIPELINE_WINDOW_SIZE, num_rows):
            is_win = False
            retries = 0
            while not is_win and retries <= MAX_TRAINING_RETRIES:
                window_expansion = retries * RETRY_WINDOW_EXPANSION_STEP
                start_idx = max(0, window_end_idx - PIPELINE_WINDOW_SIZE - window_expansion)
                end_idx = window_end_idx
                
                current_window_df = self.full_data_df.iloc[start_idx:end_idx]
                target_periode = real_data_df.iloc[window_end_idx]['Periode']
                target_logresult = real_data_df.iloc[window_end_idx]['LogResult']

                if retries == 0:
                    print(f"\n{'='*80}\nJendela Training: {current_window_df.iloc[0]['Periode']} -> {current_window_df.iloc[-1]['Periode']} | Target: {target_periode} ({target_logresult})\n{'-'*80}")

                all_predictions = set()
                prediction_input_df = current_window_df.tail(PIPELINE_WINDOW_SIZE)
                processed_input = preprocess_data_for_embedding(prediction_input_df, chunk_size=len(prediction_input_df), overlap=0, scaler_path=self.scaler_path, feature_keys_path=self.feature_keys_path)

                if not processed_input:
                    print("    -> Gagal memproses input untuk prediksi, melewatkan periode ini.")
                    retries = MAX_TRAINING_RETRIES + 1
                    continue

                sequence_data = [list(step.values()) for step in processed_input[0]['features_sequence']]
                base_live_input_tensor = torch.tensor([sequence_data], dtype=torch.float32).to(device)

                prediction_model = self.models['LSTM']
                prediction_model.eval()

                with torch.no_grad():
                    for shot in range(NUM_PREDICTION_SHOTS):
                        input_noise = torch.randn_like(base_live_input_tensor) * (PREDICTION_NOISE_LEVEL / 2) if retries > 0 or shot > 0 else torch.zeros_like(base_live_input_tensor)
                        live_input_tensor = base_live_input_tensor + input_noise
                        base_predicted_output = prediction_model(live_input_tensor)
                        
                        num_variations_per_shot = TARGET_UNIQUE_PREDICTIONS // NUM_PREDICTION_SHOTS
                        expanded_predictions = base_predicted_output.repeat(num_variations_per_shot, 1)
                        noise = torch.randn_like(expanded_predictions) * PREDICTION_NOISE_LEVEL
                        varied_predictions_batch = expanded_predictions + noise

                        shot_predictions = self._decode_predicted_features_to_number(varied_predictions_batch)
                        all_predictions.update(shot_predictions)

                        if len(all_predictions) >= TARGET_UNIQUE_PREDICTIONS: break

                is_win = str(target_logresult) in all_predictions
                result_details = f"MENANG (percobaan ke-{retries + 1})" if is_win else "KALAH"
                log_entry = f"Periode: {target_periode}, Prediksi ({len(all_predictions)} unik), Aktual: {target_logresult}, Hasil: {result_details}"
                print(log_entry)
                if is_win: print("="*80)
                self.training_log.append(log_entry)
                
                if not is_win: retries += 1

        self.self_evaluate()
        print(f"\nProses training dan evaluasi untuk {self.game_code} selesai.")
        self.save_final_models()

    def self_evaluate(self):
        """Mengevaluasi performa model."""
        print("\n--- Memulai Evaluasi Mandiri ---")
        win_count = sum(1 for log in self.training_log if "MENANG" in log)
        total_periods = len(self.training_log)
        
        if total_periods > 0:
            win_rate = (win_count / total_periods) * 100
            print(f"Tingkat Kemenangan Simulasi: {win_rate:.2f}% ({win_count} dari {total_periods} periode)")
            eval_path = os.path.join(ARTIFACTS_DIR, f"{self.game_code}_evaluation_log.json")
            with open(eval_path, 'w') as f:
                json.dump({"game_code": self.game_code, "total_periods_simulated": total_periods, "wins": win_count, "win_rate_percent": win_rate}, f, indent=4)
            print(f"Hasil evaluasi disimpan di: {eval_path}")
        else:
            print("Tidak ada data log training untuk evaluasi.")

    def save_final_models(self):
        """Menyimpan semua model setelah training selesai."""
        print(f"Menyimpan model final untuk {self.game_code}...")
        game_code_trained_files_dir = os.path.join(TRAINED_FILES_DIR, self.game_code)
        os.makedirs(game_code_trained_files_dir, exist_ok=True)
        for model_name, model in self.models.items():
            model_path = os.path.join(game_code_trained_files_dir, f"{model_name}_model.pth" if isinstance(model, nn.Module) else "XGBoost_models")
            if isinstance(model, XGBoostModel):
                model.save_model(model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"Model disimpan di: {model_path}")

def main():
    """
    Fungsi utama untuk mengorkestrasi proses training untuk semua game code.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(TRAINED_FILES_DIR, exist_ok=True)

    try:
        df_master = fetch_data_from_sql_server(SQL_CONNECTION_STRING, SQL_QUERY_MASTER)
        game_codes = df_master['GameCode'].tolist()
    except Exception as e:
        print(f"Gagal mengambil daftar game code: {e}")
        return

    print(f"Ditemukan {len(game_codes)} game code untuk training: {game_codes}")

    for game_code in game_codes:
        print(f"\n--- Memproses GameCode: {game_code} ---")
        try:
            df_full_log = fetch_data_from_sql_server(SQL_CONNECTION_STRING, SQL_QUERY_LOG.format(gameCode=game_code))
            if len(df_full_log) < PIPELINE_WINDOW_SIZE + 1:
                print(f"Data untuk {game_code} tidak cukup ({len(df_full_log)} baris). Melewatkan.")
                continue
            
            # --- Augmentasi Data Training ---
            df_augmented = augment_training_data(df_full_log, TRAINING_AUGMENTATION_FACTOR)
            if not df_augmented.empty:
                df_combined = pd.concat([df_full_log, df_augmented], ignore_index=True)
                print(f"Ukuran dataset setelah augmentasi: {len(df_combined)} baris.")
            else:
                df_combined = df_full_log

            trainer = OnlineTrainingManager(game_code, df_combined)
            trainer.run_online_training()

            # print(f"Training untuk {game_code} selesai. Mengupdate status di database...")
            # update_training_game_status(SQL_CONNECTION_STRING, game_code)

        except Exception as e:
            print(f"Terjadi error saat memproses {game_code}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n--- Semua Proses Training Selesai ---")

if __name__ == "__main__":
    main()