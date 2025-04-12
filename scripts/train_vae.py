import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset

from src.models.vae.vae_model import VAE
from src.training.vae_trainer import VAETrainer

def load_config(config_filename):
    """
    Loads a JSON configuration file.

    Args:
        config_filename (str): The configuration file name located in the configs directory.

    Returns:
        dict: Configuration parameters.
    """
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_filename
    with open(config_path, "r") as f:
        return json.load(f)

def load_data(filename):
    """
    Loads preprocessed data stored in .npz format.

    Args:
        filename (str): The file name (e.g., 'train_data.npz') in the processed data folder.

    Returns:
        np.array: Feature data.
    """
    processed_data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    data_path = processed_data_dir / filename
    data = np.load(data_path)
    # Use only the features since the VAE is unsupervised
    return data["features"]

def main():
    # Load VAE hyperparameters and training settings from configuration file.
    vae_config = load_config("vae_config.json")
    training_config = vae_config.get("training", {})

    # Load training and (optionally) validation data
    X_train = load_data("train_data.npz")
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)

    # Check for validation data
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    val_file = processed_dir / "val_data.npz"
    if val_file.exists():
        X_val = load_data("val_data.npz")
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_dataset = TensorDataset(val_tensor)
    else:
        val_dataset = None

    input_dim = train_tensor.shape[1]
    latent_dim = vae_config.get("latent_dim", 10)
    encoder_hidden_dims = vae_config.get("encoder_hidden_dims", [128, 64])
    decoder_hidden_dims = vae_config.get("decoder_hidden_dims", [64, 128])

    # Initialize VAE model
    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims
    )

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create optimizer
    learning_rate = training_config.get("learning_rate", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize and run trainer
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=training_config
    )
    trainer.train()

if __name__ == "__main__":
    main()
