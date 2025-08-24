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

def preprocess_data_for_embedding(
    df: pd.DataFrame, 
    chunk_size: int, 
    overlap: int, 
    scaler_path: str, 
    feature_keys_path: str) -> list[dict]:
    """Melakukan pra-pemrosesan data dan mengubahnya menjadi chunk untuk embedding."""
    processed_chunks = []
    try:
        df_processed = df.copy()
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
            imputer_str = SimpleImputer(strategy='most_frequent')
            df_processed[['LogResult']] = imputer_str.fit_transform(df_processed[['LogResult']])
            df_processed['LogResult'] = df_processed['LogResult'].astype(str)
            df_processed['log_result_front'] = pd.to_numeric(df_processed['LogResult'].str[0:2], errors='coerce')
            df_processed['log_result_mid'] = pd.to_numeric(df_processed['LogResult'].str[1:3], errors='coerce')
            df_processed['log_result_back'] = pd.to_numeric(df_processed['LogResult'].str[2:4], errors='coerce')
            categorical_cols = ['log_result_front', 'log_result_mid', 'log_result_back']
            for col in categorical_cols: df_processed[col] = df_processed[col].fillna(-1)

        numerical_cols = ['As', 'Kop', 'Kepala', 'Ekor', 'Id']
        imputer_num = SimpleImputer(strategy='mean')
        df_processed[numerical_cols] = imputer_num.fit_transform(df_processed[numerical_cols])
        scaler = MinMaxScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        joblib.dump(scaler, scaler_path)

        columns_to_drop = ['GameCode', 'LogResult', 'DateResultInGame', 'Periode', 'Periode_DT']
        features_df = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
        all_feature_keys = sorted(list(features_df.columns))
        with open(feature_keys_path, 'w') as f: json.dump(all_feature_keys, f)

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