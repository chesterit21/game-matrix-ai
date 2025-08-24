import pyodbc
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import time
from typing import List, Dict, Optional

# Path untuk menyimpan scaler dan encoder
SCALER_PATH = 'scaler.joblib'
ENCODER_LOG_RESULT_PATH = 'encoder_log_result.joblib'
ENCODER_PARTS_PATH = 'encoder_parts.joblib'
FEATURE_KEYS_PATH = 'feature_keys.json'

def _get_db_params(connection_string: str) -> dict:
    """
    Fungsi helper untuk mem-parsing connection string.
    """
    conn_params = {}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            conn_params[key.strip().lower()] = value.strip()
    
    return conn_params

def fetch_data_from_sql_server(connection_string: str, query: str, max_retries: int = 3) -> pd.DataFrame:
    """Mengambil data dari SQL Server menggunakan pyodbc dengan retry mechanism."""
    for attempt in range(max_retries):
        try:
            conn = pyodbc.connect(connection_string)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except pyodbc.Error as ex:
            print(f"Attempt {attempt + 1}/{max_retries}: Kesalahan database: {ex}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                raise
        except Exception as ex:
            print(f"Unexpected error: {ex}")
            raise

def test_connection(connection_string: str) -> bool:
    """Test koneksi database."""
    try:
        conn = pyodbc.connect(connection_string)
        conn.close()
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def save_predictions_to_db(connection_string: str, game_code: str, predictions: list[str]):
    """Menyimpan hasil prediksi final ke dalam tabel AINumberGame di database."""
    the_number_json = json.dumps(sorted(predictions))
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM AINumberGame WHERE GameCode = ?", (game_code,))
        cursor.execute("INSERT INTO AINumberGame (GameCode, TheNumber) VALUES (?, ?)", 
                      (game_code, the_number_json))
        conn.commit()
        print(f"Prediksi baru untuk GameCode '{game_code}' berhasil disimpan ke database.")
    except Exception as ex:
        print(f"Kesalahan database saat menyimpan prediksi: {ex}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def preprocess_data_for_embedding(df: pd.DataFrame, chunk_size: int, overlap: int) -> list[dict]:
    """Melakukan pra-pemrosesan data dan mengubahnya menjadi chunk untuk embedding."""
    if df.empty:
        return []
    
    processed_chunks = []
    try:
        df_processed = df.copy()
        
        # Handle date parsing
        if 'DateResultInGame' in df_processed.columns:
            df_processed['Periode_DT'] = pd.to_datetime(
                df_processed['DateResultInGame'], errors='coerce'
            )
        elif 'Periode' in df_processed.columns:
            # Jika Periode adalah timestamp
            df_processed['Periode_DT'] = pd.to_datetime(
                df_processed['Periode'], errors='coerce'
            )
        else:
            # Create dummy datetime if no date column
            df_processed['Periode_DT'] = pd.date_range(
                start='2023-01-01', periods=len(df_processed), freq='D'
            )
        
        df_processed.dropna(subset=['Periode_DT'], inplace=True)
        df_processed.sort_values(by='Periode_DT', inplace=True)
        df_processed.reset_index(drop=True, inplace=True)
        
        # Extract temporal features
        df_processed['hour'] = df_processed['Periode_DT'].dt.hour
        df_processed['day_of_week'] = df_processed['Periode_DT'].dt.dayofweek
        df_processed['month'] = df_processed['Periode_DT'].dt.month
        df_processed['year'] = df_processed['Periode_DT'].dt.year

        # Process LogResult
        if 'LogResult' in df_processed.columns:
            df_processed['LogResult'] = df_processed['LogResult'].astype(str).str.zfill(4)
            df_processed['log_result_front'] = pd.to_numeric(
                df_processed['LogResult'].str[0:2], errors='coerce'
            )
            df_processed['log_result_mid'] = pd.to_numeric(
                df_processed['LogResult'].str[1:3], errors='coerce'
            )
            df_processed['log_result_back'] = pd.to_numeric(
                df_processed['LogResult'].str[2:4], errors='coerce'
            )
            
            # Best Practice: Avoid 'inplace=True' on chained assignments to prevent FutureWarning.
            for col in ['log_result_front', 'log_result_mid', 'log_result_back']:
                df_processed[col] = df_processed[col].fillna(-1)

        # Process numerical columns
        numerical_cols = ['As', 'Kop', 'Kepala', 'Ekor']
        numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
        
        if numerical_cols:
            imputer_num = SimpleImputer(strategy='mean')
            df_processed[numerical_cols] = imputer_num.fit_transform(df_processed[numerical_cols])
            
            scaler = MinMaxScaler()
            df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
            joblib.dump(scaler, SCALER_PATH)

        # Prepare features dataframe
        columns_to_keep = [
            'hour', 'day_of_week', 'month', 'year',
            'log_result_front', 'log_result_mid', 'log_result_back',
            'As', 'Kop', 'Kepala', 'Ekor'
        ]
        columns_to_keep = [col for col in columns_to_keep if col in df_processed.columns]
        
        features_df = df_processed[columns_to_keep].copy()
        
        # Save feature keys
        all_feature_keys = sorted(list(features_df.columns))
        with open(FEATURE_KEYS_PATH, 'w') as f:
            json.dump(all_feature_keys, f)

        # Create chunks
        step_size = max(1, chunk_size - overlap)
        for i in range(0, len(features_df) - chunk_size + 1, step_size):
            chunk_data = features_df.iloc[i:i + chunk_size]
            chunk_data_sequence = chunk_data.to_dict(orient='records')
            
            document = {
                "chunk_id": f"chunk_{i}_{i + chunk_size - 1}",
                "features_sequence": chunk_data_sequence
            }
            processed_chunks.append(document)
            
        return processed_chunks
        
    except Exception as e:
        print(f"Kesalahan selama pra-pemrosesan data: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_documents_to_folder(documents: list[dict], output_dir: str):
    """Menyimpan dokumen yang diproses ke folder sebagai file JSON."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        for doc in documents:
            file_path = os.path.join(output_dir, f"{doc['chunk_id']}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Kesalahan I/O saat menyimpan dokumen: {e}")
        raise

def load_sample_data() -> pd.DataFrame:
    """Load sample data for testing when database is unavailable."""
    print("Using sample data for testing...")
    
    # Create sample data
    sample_data = []
    for i in range(1000):
        sample_data.append({
            'Id': i + 1,
            'GameCode': 'TEST01',
            'Periode': i + 1000,
            'LogResult': str(np.random.randint(0, 9999)).zfill(4),
            'As': np.random.randint(0, 9),
            'Kop': np.random.randint(0, 9),
            'Kepala': np.random.randint(0, 9),
            'Ekor': np.random.randint(0, 9),
            'DateResultInGame': f'2025-01-{(i % 30) + 1:02d}'
        })
    
    return pd.DataFrame(sample_data)