# src/config.py
"""
Centralized configuration file for the Game Matrix AI project.
All key hyperparameters and settings are defined here to ensure
consistency across different scripts (e.g., train.py, main.py).
"""

# --- Database and File Paths ---
SQL_CONNECTION_STRING = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=172.17.0.2;port=1433;DATABASE=GamesMatrix;UID=sa;PWD=<YourStrong@Password123>;Encrypt=False;"
DOCS_FOLDER = "docs"
CHROMA_DB_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "game_logs_embeddings"
TRAINED_FILES_DIR = "trained_models"
ARTIFACTS_DIR = "artifacts"

# --- Data Processing & Model Architecture ---
# This is the most critical parameter. It defines the fixed number of historical
# data points (rows) used for training and for creating embeddings.
# It MUST be consistent across all scripts.
PIPELINE_WINDOW_SIZE =  210

# Overlap for chunking data for ChromaDB ingestion.
OVERLAP_SIZE = 2

# --- Model Architecture ---
# Embedding Model
EMBEDDING_DIM = 128
EMBEDDING_MODEL_TYPES = ["DWT_Autoencoder", "Time2Vec", "TransformerAutoencoder", "SemanticTransformer"] # Menambah "Mata" keempat

# Transformer Specific Config
TRANSFORMER_NHEAD = 4
TRANSFORMER_DIM_FEEDFORWARD = 256

# Prediction Model (LSTM)
LSTM_HIDDEN_DIM = 128
GRU_HIDDEN_DIM = 128 # Hidden dimension for the new GRU model
MLP_HIDDEN_DIMS = [256, 128] # Hidden dimensions for the new MLP model
VAE_LATENT_DIM = 64 # Latent dimension for the new VAE model
TCN_NUM_CHANNELS = [64, 128] # Jumlah channel untuk setiap layer TCN
TCN_KERNEL_SIZE = 3 # Ukuran kernel untuk konvolusi TCN
TCN_DROPOUT = 0.2 # Dropout rate untuk TCN
LSTM_NUM_LAYERS = 2

# --- Training Hyperparameters ---
ONLINE_TRAINING_EPOCHS = 5
LEARNING_RATE = 0.001
MAX_TRAINING_RETRIES = 5
RETRY_WINDOW_EXPANSION_STEP = 50

# --- Prediction Generation ---
TARGET_UNIQUE_PREDICTIONS = 8820
PREDICTION_NOISE_LEVEL = 0.05
NUM_PREDICTION_SHOTS = 25
PREDICTION_HORIZON = 9380 # For main.py final output
N_RETRIEVED_CHUNKS = 20