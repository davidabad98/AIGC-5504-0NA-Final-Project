import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GENERATION_CONFIG, PROCESSED_PREPROCESSOR_PATH, VAE_CONFIG
from src.data.postprocessing import inverse_transform
from src.data.preprocessing import ChurnDataPreprocessor
from src.models.vae.vae_model import VAE


def load_checkpoint(checkpoint_file, model, device):
    """
    Loads the model checkpoint.

    Args:
        checkpoint_file (str): Path to the checkpoint file.
        model (VAE): The VAE model instance.
        device (torch.device): The device to map the model.

    Returns:
        The model with loaded weights.
    """
    # checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_file
    checkpoint_path = checkpoint_file
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def generate_samples(model, num_samples, latent_dim, device):
    """
    Generates synthetic samples using the VAE decoder.

    Args:
        model (VAE): The trained VAE model.
        num_samples (int): Number of synthetic samples to generate.
        latent_dim (int): Dimension of the latent space.
        device (torch.device): Device to perform generation on.

    Returns:
        torch.Tensor: Generated synthetic samples.
    """
    model.eval()
    # Sample random latent vectors from the standard normal distribution
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        synthetic_data = model.decoder(z)
    return synthetic_data.cpu()


def main():
    # Load generation configuration
    gen_config = GENERATION_CONFIG
    num_samples = gen_config.get("num_samples", 1000)
    output_file = gen_config.get("output_file", "synthetic_customer_churn.csv")
    checkpoint_file = gen_config.get("checkpoint_file")

    # Load VAE configuration (hyperparameters must match training)
    vae_config = VAE_CONFIG
    latent_dim = vae_config.get("latent_dim", 10)
    # Note: For generation, we need the original input_dim. This should be consistent with the trained model.
    # Adjust the following line if you decide to include input_dim in your config.
    input_dim = vae_config.get("input_dim", 15)  # default example value

    encoder_hidden_dims = vae_config.get("encoder_hidden_dims", [128, 64])
    decoder_hidden_dims = vae_config.get("decoder_hidden_dims", [64, 128])

    # Initialize VAE model
    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
    )

    # Set device: using GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model weights from checkpoint
    model = load_checkpoint(checkpoint_file, model, device)

    # Generate synthetic samples (in the processed feature space)
    synthetic_processed = generate_samples(model, num_samples, latent_dim, device)
    synthetic_processed_np = synthetic_processed.numpy()

    # Load the saved preprocessor so that we can reverse-transform the synthetic features
    preprocessor = ChurnDataPreprocessor.load_preprocessor(PROCESSED_PREPROCESSOR_PATH)

    # Perform the inverse transformation to recover original data values
    df_original = inverse_transform(preprocessor, synthetic_processed_np)

    # Optionally, generate a default "Churn" column.
    # Since the target was dropped before preprocessing, you may need to
    # append it based on a chosen strategy. Here we assign a default value (e.g., 0).
    df_original["Churn"] = 0  # or sample based on desired distribution

    # Create a synthetic CustomerID column (sequential IDs starting at 1)
    df_original.insert(0, "CustomerID", range(1, len(df_original) + 1))

    # Reorder columns to match the desired final output.
    # Here we assume the original feature order (after CustomerID) is exactly:
    # Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
    # Subscription Type, Contract Length, Total Spend, Last Interaction, and then Churn.
    desired_order = [
        "CustomerID",
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction",
        "Churn",
    ]

    # In case inverse_transform returns a DataFrame with different column names,
    # you might need to rename them accordingly. For example:
    # df_original.rename(columns={
    #     "age_col_after_inv": "Age",
    #     "gender_col_after_inv": "Gender",
    #     ...
    # }, inplace=True)

    # Reindex to enforce desired column order
    df_final = df_original.reindex(columns=desired_order)

    # Save final DataFrame as CSV
    df_final.to_csv(output_file, index=False)
    print(
        f"Generated {num_samples} synthetic samples. Final CSV saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
