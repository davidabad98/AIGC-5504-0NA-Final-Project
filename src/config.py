import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Create directories if they don't exist
for dir_path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

# Data file paths
RAW_TRAIN_PATH = os.path.join(
    RAW_DATA_DIR, "customer_churn_dataset-training-master.csv"
)
RAW_TEST_PATH = os.path.join(RAW_DATA_DIR, "customer_churn_dataset-testing-master.csv")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_data.npz")
PROCESSED_VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val_data.npz")
PROCESSED_PREPROCESSOR_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessor.pkl")

# Data Preprocessing configuration (migrated from data_config.json)
DATA_CONFIG = {
    "raw_file_name": "customer_churn_dataset-training-master.csv",
    "processed_file_name": "processed_customer_churn-training.csv",
    "target_column": "Churn",
    "exclude_columns": ["CustomerID"],
    "remove_outliers": True,
    "outlier_method": "IQR",
    "outlier_threshold": 1.5,
    "validation_size": 0.15,
    "random_state": 42,
    "scaling_method": "standard",
    "categorical_encoding": "onehot",
}

# Generation configuration (migrated from generation_config.json)
# For checkpoint_file and vae_config_file we build absolute paths where applicable.
GENERATION_CONFIG = {
    "num_samples": 1000,
    "output_file": os.path.join(SYNTHETIC_DATA_DIR, "synthetic_customer_churn.csv"),
    # Convert the relative path to an absolute path using MODELS_DIR
    "checkpoint_file": os.path.join(MODELS_DIR, "vae_checkpoint.pth"),
}

# VAE model configuration (migrated from vae_config.json)
VAE_CONFIG = {
    "latent_dim": 10,
    "encoder_hidden_dims": [128, 64],
    "decoder_hidden_dims": [64, 128],
    "training": {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        # Checkpoint path using the absolute path from MODELS_DIR
        "checkpoint_path": os.path.join(MODELS_DIR, "vae_checkpoint.pth"),
    },
    # (Optional) If you decide to specify the input dimension directly here.
    # "input_dim": 12
}
