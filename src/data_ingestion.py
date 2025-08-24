# src/data_ingestion.py

import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

def _get_db_params(connection_string: str) -> dict:
    """
    Fungsi helper untuk mem-parsing connection string dan menyiapkan parameter.
    """
    conn_params = {k.strip().lower(): v.strip() for k, v in (part.split('=', 1) for part in connection_string.split(';') if '=' in part)}
    
    db_params = {
        'server': conn_params.get('server', '172.17.0.2'),
        'port': conn_params.get('port', '1433'),
        'user': conn_params.get('uid', conn_params.get('user', 'sa')),
        'password': conn_params.get('pwd', conn_params.get('password', '<YourStrong@Password123>')),
        'database': conn_params.get('database', 'GamesMatrix')
    }
    return db_params

def _get_sqlalchemy_engine(connection_string: str):
    """Membuat SQLAlchemy engine dari connection string dengan password yang di-encode."""
    db_params = _get_db_params(connection_string)    
    encoded_password = quote_plus(db_params['password'])
    
    # URL format: mssql+pymssql://user:password@host:port/database
    conn_url = (
        f"mssql+pymssql://{db_params['user']}:{encoded_password}@"
        f"{db_params['server']}:{db_params['port']}/{db_params['database']}"
    )
    return create_engine(conn_url)

def fetch_data_from_sql_server(connection_string: str, query: str) -> pd.DataFrame:
    """Mengambil data dari SQL Server menggunakan SQLAlchemy."""
    try:
        engine = _get_sqlalchemy_engine(connection_string)
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df
    except SQLAlchemyError as ex:
        print(f"Kesalahan database saat mengambil data: {ex}")
        raise

def save_predictions_to_db(connection_string: str, game_code: str, predictions_string: str):
    """Menyimpan hasil prediksi final ke dalam tabel AINumberGame di database menggunakan SQLAlchemy."""
    try:
        engine = _get_sqlalchemy_engine(connection_string)
        with engine.connect() as connection:
            with connection.begin() as transaction:
                try:
                    # Gunakan text() untuk statement SQL dan parameter binding untuk keamanan
                    del_stmt = text("DELETE FROM AINumberGame WHERE GameCode = :game_code")
                    connection.execute(del_stmt, {"game_code": game_code})
                    
                    ins_stmt = text("INSERT INTO AINumberGame (GameCode, TheNumber) VALUES (:game_code, :the_number)")
                    connection.execute(ins_stmt, {"game_code": game_code, "the_number": predictions_string})
                    
                    transaction.commit()
                    print(f"Prediksi baru untuk GameCode '{game_code}' berhasil disimpan ke database.")
                except Exception as e:
                    print(f"Gagal menyimpan prediksi, transaksi dibatalkan: {e}")
                    transaction.rollback()
                    raise
    except SQLAlchemyError as ex:
        print(f"Kesalahan database saat menyimpan prediksi: {ex}")
        raise

def update_training_game_status(connection_string: str, game_code: str):
    """Mengupdate status IsActive menjadi 0 untuk game_code tertentu di tabel TRAININGGAME."""
    try:
        engine = _get_sqlalchemy_engine(connection_string)
        with engine.connect() as connection:
            with connection.begin() as transaction:
                try:
                    stmt = text("UPDATE TRAININGGAME SET IsActive = 0 WHERE GameCode = :game_code")
                    connection.execute(stmt, {"game_code": game_code})
                    transaction.commit()
                    print(f"Status untuk GameCode '{game_code}' berhasil diupdate menjadi tidak aktif (IsActive = 0).")
                except Exception as e:
                    print(f"Gagal mengupdate status untuk {game_code}, transaksi dibatalkan: {e}")
                    transaction.rollback()
                    raise
    except SQLAlchemyError as ex:
        print(f"Kesalahan database saat mengupdate status: {ex}")
        raise


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur-fitur baru yang kaya dari data mentah untuk meningkatkan performa model.
    """
    df_eng = df.copy()
    if 'LogResult' not in df_eng.columns:
        return df_eng

    df_eng['LogResult_Str'] = df_eng['LogResult'].astype(str).str.zfill(4)

    # Fitur Target (untuk model klasifikasi/regresi) - memprediksi setiap digit
    for i in range(4):
        df_eng[f'digit_{i+1}'] = pd.to_numeric(df_eng['LogResult_Str'].str[i], errors='coerce')

    # Fitur Jump (Delta)
    df_eng['periode_jump'] = df_eng['Periode'].diff().fillna(0)
    numeric_result = pd.to_numeric(df_eng['LogResult'], errors='coerce')
    df_eng['result_jump'] = numeric_result.diff().fillna(0)

    # Fitur Statistik Rolling
    window_sizes = [5, 10, 20]
    for window in window_sizes:
        df_eng[f'result_roll_mean_{window}'] = df_eng['result_jump'].rolling(window=window).mean().fillna(0)
        df_eng[f'result_roll_std_{window}'] = df_eng['result_jump'].rolling(window=window).std().fillna(0)

    # Fitur Domain
    df_eng['is_even'] = (numeric_result % 2 == 0).astype(int)
    df_eng['is_small'] = (numeric_result < 5000).astype(int)
    
    df_eng = df_eng.drop(columns=['LogResult_Str'])
    df_eng = df_eng.fillna(0)

    return df_eng

def prepare_tabular_data(df: pd.DataFrame):
    """
    Mempersiapkan data dalam format tabular untuk model seperti XGBoost.
    Setiap baris adalah fitur dari satu timestep, targetnya adalah digit dari timestep berikutnya.
    """
    if df.shape[0] < 2:
        return pd.DataFrame(), pd.DataFrame()

    df_featured = feature_engineering(df)
    
    original_digit_cols = [f'digit_{i+1}' for i in range(4)]
    shifted_target_cols = [f'target_digit_{i+1}' for i in range(4)]

    # Ensure original digit columns exist before trying to shift them
    if not all(col in df_featured.columns for col in original_digit_cols):
        return pd.DataFrame(), pd.DataFrame()
    
    # Shift target columns to align with the previous timestep's features
    for i, col in enumerate(original_digit_cols):
        df_featured[shifted_target_cols[i]] = df_featured[col].shift(-1)
        
    # Drop rows with NaN targets (the last row)
    df_featured = df_featured.dropna(subset=shifted_target_cols)

    # Define feature columns by excluding identifiers and all target-related columns
    # Also exclude 'LogResult' if it still exists, just in case.
    cols_to_exclude = ['GameCode', 'DateResultInGame', 'Periode', 'Periode_DT', 'LogResult'] + original_digit_cols + shifted_target_cols
    feature_cols = [col for col in df_featured.columns if col not in cols_to_exclude]
    
    X = df_featured[feature_cols]
    y = df_featured[shifted_target_cols]
    
    return X, y

def preprocess_data_for_embedding(
    df: pd.DataFrame, 
    chunk_size: int, 
    overlap: int, 
    scaler_path: str, 
    feature_keys_path: str) -> list[dict]:
    """Melakukan pra-pemrosesan data dan mengubahnya menjadi chunk untuk embedding."""
    processed_chunks = []
    try:
        # Terapkan feature engineering
        df_engineered = feature_engineering(df)

        df_processed = df_engineered.copy()
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

        # Best Practice: Define the final feature set BEFORE scaling to ensure consistency.
        columns_to_drop = ['GameCode', 'LogResult', 'DateResultInGame', 'Periode', 'Periode_DT']
        features_df_unscaled = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
        all_feature_keys = sorted(list(features_df_unscaled.columns))

        # Impute and Scale ONLY the final features.
        # This ensures the scaler is trained on the exact same columns as the models.
        imputer_num = SimpleImputer(strategy='mean')
        features_df_imputed = pd.DataFrame(imputer_num.fit_transform(features_df_unscaled), columns=all_feature_keys)
        
        scaler = MinMaxScaler()
        features_df_scaled = pd.DataFrame(scaler.fit_transform(features_df_imputed), columns=all_feature_keys)
        
        # Save the correctly trained scaler and the corresponding feature keys.
        joblib.dump(scaler, scaler_path)
        with open(feature_keys_path, 'w') as f: json.dump(all_feature_keys, f)

        features_df = features_df_scaled
        step_size = chunk_size - overlap
        for i in range(0, len(features_df) - chunk_size + 1, step_size):
            chunk_data_sequence = features_df.iloc[i:i + chunk_size].to_dict(orient='records')
            document = {
                "chunk_id": f"chunk_{i}_{i + chunk_size - 1}",
                "features_sequence": chunk_data_sequence
            }
            processed_chunks.append(document)
        return processed_chunks
    except Exception as e:
        print(f"Kesalahan selama pra-pemrosesan data: {e}")
        raise

def save_documents_to_folder(documents: list[dict], output_dir: str):
    """Menyimpan dokumen yang diproses ke folder sebagai file JSON."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        for doc in documents:
            file_path = os.path.join(output_dir, f"{doc['chunk_id']}.json")
            with open(file_path, 'w') as f:
                json.dump(doc, f, indent=4)
    except IOError as e:
        print(f"Kesalahan I/O saat menyimpan dokumen: {e}")
        raise