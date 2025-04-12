import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Variational Autoencoder Decoder Module.

    Decodes the latent vector back to the original input space.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_dim (int): Dimensionality of the output (should match the encoder input).
        hidden_dims (list, optional): List of hidden layer dimensions. Defaults to [64, 128].
    """
    def __init__(self, latent_dim, output_dim, hidden_dims=None):
        super(Decoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        modules = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(in_dim, h_dim))
            modules.append(nn.ReLU())
            in_dim = h_dim

        modules.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        """
        Forward pass through the decoder.

        Args:
            z (Tensor): Latent variable tensor.

        Returns:
            Tensor: Reconstructed input.
        """
        x_recon = self.decoder(z)
        return x_recon
