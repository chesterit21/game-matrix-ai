# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import joblib
from src.config import SQL_CONNECTION_STRING, ARTIFACTS_DIR, TRAINED_FILES_DIR, PIPELINE_WINDOW_SIZE, ONLINE_TRAINING_EPOCHS, LEARNING_RATE, TARGET_UNIQUE_PREDICTIONS, MAX_TRAINING_RETRIES, RETRY_WINDOW_EXPANSION_STEP, PREDICTION_NOISE_LEVEL, NUM_PREDICTION_SHOTS, EMBEDDING_MODEL_TYPES, EMBEDDING_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS
from src.data_ingestion import fetch_data_from_sql_server, preprocess_data_for_embedding, update_training_game_status
from src.embedding_models import Time2VecAutoencoder, DWTAutoencoder
from src.prediction_models import LSTMModel

# --- Konfigurasi Global ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

SQL_QUERY_MASTER = "SELECT GameCode FROM TRAININGGAME WHERE IsActive = 1;"
SQL_QUERY_LOG = "SELECT Id, GameCode, Periode, LogResult, [As], Kop, Kepala, Ekor,DateResultInGame FROM LogGame WHERE GameCode='{gameCode}' AND Periode > (SELECT TOP 1 Periode FROM LogGame WHERE GameCode = '{gameCode}' ORDER BY Periode DESC)-365 ORDER BY Periode ASC;"

class OnlineTrainingManager:
    """Mengelola proses training online sekuensial untuk satu GameCode."""

    def __init__(self, game_code: str, full_data_df: pd.DataFrame):
        self.game_code = game_code
        self.full_data_df = full_data_df
        self.training_log = []

        # Definisikan path artefak spesifik untuk game_code ini
        self.scaler_path = os.path.join(ARTIFACTS_DIR, f"{self.game_code}_scaler.joblib")
        self.feature_keys_path = os.path.join(ARTIFACTS_DIR, f"{self.game_code}_feature_keys.json")

        # Muat scaler dan feature keys untuk decoding prediksi
        try:
            self.scaler = joblib.load(self.scaler_path)
            with open(self.feature_keys_path, 'r') as f:
                self.feature_keys = json.load(f)
        except FileNotFoundError as e:
            print(f"Info: Scaler/feature_keys untuk {self.game_code} tidak ditemukan. Akan dibuat saat training pertama.")
            self.scaler = None
            self.feature_keys = None

        self._initialize_models()

    def _initialize_models(self):
        """Inisialisasi semua model, optimizer, dan criterion."""
        self.models = {}
        self.optimizers = {}
        self.criterions = {}

        # Jika feature_keys belum ada, kita harus membuatnya terlebih dahulu
        if not self.feature_keys:
            print("Membuat feature_keys awal...")
            # Lakukan preprocessing pada data sampel untuk mendapatkan daftar fitur
            preprocess_data_for_embedding(self.full_data_df.head(PIPELINE_WINDOW_SIZE), PIPELINE_WINDOW_SIZE, 0, self.scaler_path, self.feature_keys_path)
            # FIX: Load the newly created scaler and feature keys into the instance
            self.scaler = joblib.load(self.scaler_path)
            with open(self.feature_keys_path, 'r') as f: self.feature_keys = json.load(f) # Reload feature keys as well

        num_features = len(self.feature_keys)

        for model_type in EMBEDDING_MODEL_TYPES:
            if model_type == "Time2Vec":
                self.models[model_type] = Time2VecAutoencoder(num_features, PIPELINE_WINDOW_SIZE, EMBEDDING_DIM).to(device)
            else: # DWT_Autoencoder
                self.models[model_type] = DWTAutoencoder(PIPELINE_WINDOW_SIZE * num_features, EMBEDDING_DIM).to(device)
            
            self.optimizers[model_type] = optim.Adam(self.models[model_type].parameters(), lr=LEARNING_RATE)
            self.criterions[model_type] = nn.MSELoss()

        self.models['LSTM'] = LSTMModel(input_dim=num_features, hidden_dim=LSTM_HIDDEN_DIM, output_dim=num_features, num_layers=LSTM_NUM_LAYERS).to(device)
        self.optimizers['LSTM'] = optim.Adam(self.models['LSTM'].parameters(), lr=LEARNING_RATE)
        self.criterions['LSTM'] = nn.MSELoss()

        print(f"Semua model untuk {self.game_code} telah diinisialisasi.")

    def _decode_predicted_features_to_number(self, predicted_features_batch: torch.Tensor) -> set[str]:
        """
        Mendekode satu batch tensor fitur yang diprediksi menjadi sekumpulan nomor 4 digit unik.
        Fungsi ini sangat dioptimalkan menggunakan operasi vectorized NumPy.
        """
        pred_df = pd.DataFrame(predicted_features_batch.cpu().numpy(), columns=self.feature_keys)

        numerical_cols = ['As', 'Kop', 'Kepala', 'Ekor', 'Id']
        cols_to_transform = [col for col in numerical_cols if col in pred_df.columns]
        cols_for_number = ['As', 'Kop', 'Kepala', 'Ekor']
        if not all(c in pred_df.columns for c in cols_for_number):
            return {"0000"}

        inversed_features = self.scaler.inverse_transform(pred_df[cols_to_transform])
        inversed_df = pd.DataFrame(inversed_features, columns=cols_to_transform)

        # FIX: The meshgrid approach was memory-explosive.
        # Reverting to a memory-safe iterative approach that generates combinations
        # per prediction, not across the entire batch.

        # Get all values at once for efficiency
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

    def _train_step(self, current_window_df: pd.DataFrame, epoch_multiplier: int = 1):
        """Lakukan satu langkah training (fine-tuning) pada semua model."""
        # Ini adalah jendela data penuh, yang bisa lebih besar dari standar saat percobaan ulang.
        full_processed_chunks = preprocess_data_for_embedding(current_window_df, chunk_size=len(current_window_df), overlap=0, scaler_path=self.scaler_path, feature_keys_path=self.feature_keys_path)
        
        num_epochs = ONLINE_TRAINING_EPOCHS

        if not full_processed_chunks:
            return
        
        full_sequence_data = [list(step.values()) for step in full_processed_chunks[0]['features_sequence']]
        full_window_tensor = torch.tensor([full_sequence_data], dtype=torch.float32).to(device)

        # --- Training Autoencoder (Ukuran Tetap) ---
        # Autoencoder memiliki arsitektur tetap, jadi kita harus melatihnya dengan data berukuran standar.
        # Kita ambil data terbaru dari jendela saat ini sejumlah PIPELINE_WINDOW_SIZE.
        standard_window_df = current_window_df.tail(PIPELINE_WINDOW_SIZE)
        if len(standard_window_df) >= PIPELINE_WINDOW_SIZE:
            std_processed_chunks = preprocess_data_for_embedding(standard_window_df, chunk_size=len(standard_window_df), overlap=0, scaler_path=self.scaler_path, feature_keys_path=self.feature_keys_path)
            std_sequence_data = [list(step.values()) for step in std_processed_chunks[0]['features_sequence']]
            std_window_tensor = torch.tensor([std_sequence_data], dtype=torch.float32).to(device)

            # Latih setiap model embedding dengan data berukuran standar
            for model_type in EMBEDDING_MODEL_TYPES:
                model = self.models[model_type]
                optimizer = self.optimizers[model_type]
                criterion = self.criterions[model_type]
                model.train()
                
                for _ in range(num_epochs):
                    optimizer.zero_grad()
                    if model_type == "Time2Vec":
                        _, reconstructed = model(std_window_tensor) # Gunakan tensor ukuran standar
                        with torch.no_grad():
                            t2v_model = model.time2vec
                            batch_size = std_window_tensor.shape[0]
                            seq_len = std_window_tensor.shape[1]
                            tau = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                            tau = tau.repeat(batch_size, 1, 1)
                            time_embedding = t2v_model(tau)
                            combined_features = torch.cat([std_window_tensor, time_embedding], dim=-1)
                            target = combined_features.view(batch_size, -1)
                    else: # DWT_Autoencoder
                        flat_window = std_window_tensor.view(1, -1) # Gunakan tensor ukuran standar
                        _, reconstructed = model(flat_window)
                        target = flat_window
                    loss = criterion(reconstructed, target)
                    loss.backward()
                    optimizer.step()

        # --- Training LSTM (Ukuran Dinamis) ---
        # LSTM dapat menangani panjang sekuens yang bervariasi, jadi kita gunakan jendela penuh (yang diperluas)
        # untuk memberinya konteks historis yang lebih kaya.
        lstm_model = self.models['LSTM']
        lstm_optimizer = self.optimizers['LSTM']
        lstm_criterion = self.criterions['LSTM']
        lstm_model.train()
        
        # Input (X) adalah semua langkah dalam jendela penuh kecuali yang terakhir.
        # Target (y) adalah HANYA langkah terakhir dari jendela penuh.
        X = full_window_tensor[:, :-1, :]
        y = full_window_tensor[:, -1, :]
        
        for _ in range(num_epochs):
            lstm_optimizer.zero_grad()
            outputs = lstm_model(X)
            loss = lstm_criterion(outputs, y)
            loss.backward()
            lstm_optimizer.step()

    def run_online_training(self):
        """Menjalankan seluruh proses training online sekuensial."""
        print(f"\nMemulai training sekuensial untuk {self.game_code}...")
        num_rows = len(self.full_data_df)
        
        current_period_idx = 0
        while current_period_idx < (num_rows - PIPELINE_WINDOW_SIZE):
            is_win = False
            retries = 0

            while not is_win and retries <= MAX_TRAINING_RETRIES:
                # Logika Augmentasi Kontekstual: Perluas jendela training ke belakang setiap kali gagal.
                window_expansion = retries * RETRY_WINDOW_EXPANSION_STEP
                current_training_window_size = PIPELINE_WINDOW_SIZE + window_expansion
                
                # Pastikan start_idx tidak kurang dari 0
                start_idx = max(0, current_period_idx - window_expansion)
                end_idx = current_period_idx + PIPELINE_WINDOW_SIZE
                
                current_window_df = self.full_data_df.iloc[start_idx:end_idx]
                actual_window_size = len(current_window_df)

                target_periode = self.full_data_df.iloc[end_idx]['Periode']
                target_logresult = self.full_data_df.iloc[end_idx]['LogResult']

                # Logging disederhanakan untuk fokus pada hasil.
                # Informasi detail hanya akan dicetak pada percobaan pertama untuk konteks.
                if retries == 0:
                    window_start_periode = current_window_df.iloc[0]['Periode']
                    window_end_periode = self.full_data_df.iloc[end_idx - 1]['Periode']
                    print("\n" + "="*80)
                    print(f"Jendela Training: {window_start_periode} -> {window_end_periode} | Target Prediksi: Periode {target_periode} (Aktual: {target_logresult})")
                    print("-"*80)

                # --- Sesi Generasi Prediksi Multi-Shot Parallel ---
                # BEST PRACTICE: Mengatasi "mode collapse" dengan pendekatan ensemble.
                # Kita tidak hanya mengandalkan satu prediksi dasar. Sebaliknya, kita membuat
                # beberapa "tembakan" (shots) prediksi. Setiap tembakan dimulai dengan input
                # yang sedikit diubah, memaksa model untuk menjelajahi jalur prediksi yang berbeda.
                all_predictions = set()
                # Gunakan data terbaru dengan ukuran standar untuk prediksi agar konsisten
                prediction_input_df = current_window_df.tail(PIPELINE_WINDOW_SIZE)
                processed_input = preprocess_data_for_embedding(prediction_input_df, chunk_size=len(prediction_input_df), overlap=0, scaler_path=self.scaler_path, feature_keys_path=self.feature_keys_path)

                if not processed_input:
                    print("    -> Gagal memproses input untuk prediksi, melewatkan periode ini.")
                    retries = MAX_TRAINING_RETRIES + 1 # Paksa lanjut ke periode berikutnya
                    continue

                sequence_data = [list(step.values()) for step in processed_input[0]['features_sequence']]
                base_live_input_tensor = torch.tensor([sequence_data], dtype=torch.float32).to(device)

                lstm_model = self.models['LSTM']
                lstm_model.eval()

                with torch.no_grad():
                    # Loop untuk setiap "tembakan" prediksi
                    for shot in range(NUM_PREDICTION_SHOTS):
                        # 1. Buat input yang sedikit berbeda untuk setiap tembakan dengan menambahkan noise
                        # Pada percobaan ulang (retries > 0), kita gunakan noise yang lebih besar untuk mendorong eksplorasi
                        if retries > 0 or shot > 0:
                            input_noise = torch.randn_like(base_live_input_tensor) * (PREDICTION_NOISE_LEVEL / 2)
                            live_input_tensor = base_live_input_tensor + input_noise
                        else:
                            live_input_tensor = base_live_input_tensor

                        # 2. Panggil LSTM untuk mendapatkan prediksi dasar untuk tembakan ini
                        base_predicted_output = lstm_model(live_input_tensor)
                        
                        # 3. Tentukan jumlah variasi yang akan dibuat per tembakan
                        num_variations_per_shot = TARGET_UNIQUE_PREDICTIONS // NUM_PREDICTION_SHOTS
                        
                        # 4. Buat batch variasi dari prediksi dasar ini
                        expanded_predictions = base_predicted_output.repeat(num_variations_per_shot, 1)
                        noise = torch.randn_like(expanded_predictions) * PREDICTION_NOISE_LEVEL
                        varied_predictions_batch = expanded_predictions + noise

                        # 5. Dekode seluruh batch dan tambahkan ke set prediksi utama
                        shot_predictions = self._decode_predicted_features_to_number(varied_predictions_batch)
                        all_predictions.update(shot_predictions)

                        # Hentikan jika target sudah tercapai lebih awal
                        if len(all_predictions) >= TARGET_UNIQUE_PREDICTIONS:
                            break

                is_win = str(target_logresult) in all_predictions

                result_details = "KALAH"
                if is_win:
                    result_details = f"MENANG (pada percobaan training ke-{retries + 1})"

                log_entry = (f"Periode: {target_periode}, "
                             f"Prediksi ({len(all_predictions)} unik), "
                             f"Aktual: {target_logresult}, Hasil: {result_details}")
                print(log_entry)
                if is_win:
                    print("="*80) # Cetak garis pemisah setelah menang
                self.training_log.append(log_entry)
                
                if not is_win:
                    # Jika model kalah, tingkatkan jumlah percobaan ulang.
                    # Ini akan menyebabkan loop 'while' di atas berjalan lagi untuk melatih ulang model.
                    retries += 1
            
            # Pindah ke periode selanjutnya setelah MENANG atau setelah mencoba maksimal
            current_period_idx += 1

        print(f"\nTraining sekuensial untuk {self.game_code} selesai.")
        self.save_final_models()

    def save_final_models(self):
        """Menyimpan semua model setelah training selesai."""
        print(f"Menyimpan model final untuk {self.game_code}...")

        # Buat direktori GameCode jika belum ada
        game_code_trained_files_dir = os.path.join(TRAINED_FILES_DIR, self.game_code)
        os.makedirs(game_code_trained_files_dir, exist_ok=True)
        for model_name, model in self.models.items():
            model_filename = f"{model_name}_model.pth"
            model_path = os.path.join(game_code_trained_files_dir, model_filename)
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
                print(f"Data untuk {game_code} tidak cukup ({len(df_full_log)} baris), memerlukan setidaknya {PIPELINE_WINDOW_SIZE + 1}. Melewatkan.")
                continue
            
            trainer = OnlineTrainingManager(game_code, df_full_log)
            trainer.run_online_training()

            # Setelah training selesai, update status
            print(f"Training untuk {game_code} selesai. Mengupdate status di database...")
            update_training_game_status(SQL_CONNECTION_STRING, game_code)

        except Exception as e:
            print(f"Terjadi error saat memproses {game_code}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n--- Semua Proses Training Selesai ---")

if __name__ == "__main__":
    main()
