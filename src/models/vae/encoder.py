import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Variational Autoencoder Encoder Module.

    Encodes the input into a latent space represented by its mean and log variance.

    Args:
        input_dim (int): Dimensionality of the input features.
        latent_dim (int): Dimensionality of the latent space.
        hidden_dims (list, optional): List of hidden layer dimensions. Defaults to [128, 64].
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        modules = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(in_dim, h_dim))
            modules.append(nn.ReLU())
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        # Learnable parameters for the latent distribution
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: Tuple containing tensors for the latent mean and log variance.
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
