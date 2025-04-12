import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, VAE_CONFIG
from src.models.vae.vae_model import VAE
from src.training.vae_trainer import VAETrainer
from torch.utils.data import TensorDataset


def load_data(file_path):
    """
    Loads preprocessed data stored in .npz format.

    Returns:
        np.array: Feature data.
    """
    data = np.load(file_path)
    # Use only the features since the VAE is unsupervised
    return data["features"]


def main():
    # Load VAE hyperparameters and training settings from configuration file.
    vae_config = VAE_CONFIG
    training_config = vae_config.get("training", {})

    # Load training and (optionally) validation data
    X_train = load_data(PROCESSED_TRAIN_PATH)
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)

    # Check for validation data

    X_val = load_data(PROCESSED_VAL_PATH)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    val_dataset = TensorDataset(val_tensor)

    input_dim = train_tensor.shape[1]
    latent_dim = vae_config.get("latent_dim", 10)
    encoder_hidden_dims = vae_config.get("encoder_hidden_dims", [128, 64])
    decoder_hidden_dims = vae_config.get("decoder_hidden_dims", [64, 128])

    # Initialize VAE model
    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
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
        config=training_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
