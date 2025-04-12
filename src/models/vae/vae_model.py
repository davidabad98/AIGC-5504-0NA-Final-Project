import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vae.encoder import Encoder
from src.models.vae.decoder import Decoder

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) that ties together an encoder and a decoder.

    Args:
        input_dim (int): Dimensionality of input features.
        latent_dim (int): Dimensionality of the latent space.
        encoder_hidden_dims (list, optional): Hidden dimensions for the encoder. Defaults to [128, 64].
        decoder_hidden_dims (list, optional): Hidden dimensions for the decoder. Defaults to [64, 128].
    """
    def __init__(self, input_dim, latent_dim, encoder_hidden_dims=None, decoder_hidden_dims=None):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, decoder_hidden_dims)

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample from N(mu, sigma^2).

        Args:
            mu (Tensor): Mean from the encoder.
            logvar (Tensor): Log variance from the encoder.

        Returns:
            Tensor: Reparameterized latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass: encode, reparameterize, and decode.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: Reconstructed input, latent mean, and log variance.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Compute the VAE loss composed of reconstruction loss and KL divergence.

        Args:
            recon_x (Tensor): Reconstructed input.
            x (Tensor): Original input.
            mu (Tensor): Mean from encoder.
            logvar (Tensor): Log variance from encoder.

        Returns:
            dict: Loss components including total loss, reconstruction loss, and KL divergence.
        """
        # Mean squared error as the reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + kl_div
        return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_div": kl_div}
