"""
Conditional Variational Autoencoder (cVAE) for Blood Transcription Module discovery.

Architecture:
  Encoder: X || c  -->  [hidden layers]  -->  mu, log_var  (latent Z)
  Decoder: Z || c  -->  [hidden layers]  -->  X_recon
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    """Maps (gene_expression || condition) to latent distribution parameters."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim + condition_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xc = torch.cat([x, c], dim=1)
        h = self.network(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Maps (latent Z || condition) back to reconstructed gene expression."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim + condition_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final reconstruction layer — no activation (regression)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, c], dim=1)
        return self.network(zc)


class ConditionalVAE(nn.Module):
    """
    Complete MI-Regularized Conditional VAE.

    Forward pass:
      1. Encode X||c  -> mu, logvar
      2. Reparameterize -> Z
      3. Decode Z||c   -> X_recon

    The decoder's final Linear layer weights W map latent dimensions (modules)
    to genes, which are extracted post-training for BTM discovery.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample Z from N(mu, sigma^2) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_recon: reconstructed expression
            mu:      latent mean
            logvar:  latent log-variance
            z:       sampled latent vector
        """
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar, z

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Encode to latent mean (deterministic projection for inference)."""
        mu, _ = self.encoder(x, c)
        return mu

    @classmethod
    def from_config(cls, model_cfg, condition_dim: int) -> "ConditionalVAE":
        """Factory method to build from a ModelConfig dataclass."""
        return cls(
            input_dim=model_cfg.input_dim,
            condition_dim=condition_dim,
            latent_dim=model_cfg.latent_dim,
            encoder_hidden_dims=model_cfg.encoder_hidden_dims,
            decoder_hidden_dims=model_cfg.decoder_hidden_dims,
            dropout=model_cfg.dropout,
            use_batch_norm=model_cfg.use_batch_norm,
        )
