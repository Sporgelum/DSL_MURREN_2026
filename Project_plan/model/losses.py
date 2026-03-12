"""
Loss functions for the MI-Regularized cVAE.

Total loss = Reconstruction + beta * KL Divergence - lambda * MI(X; Z)
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error reconstruction loss (per sample, summed over genes)."""
    return F.mse_loss(x_recon, x, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between q(z|x) = N(mu, sigma^2) and p(z) = N(0, I).

    D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    Averaged over the batch.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def compute_kl_weight(epoch: int, anneal_epochs: int, max_weight: float) -> float:
    """Linear KL annealing: ramp from 0 to max_weight over anneal_epochs."""
    if anneal_epochs <= 0:
        return max_weight
    return min(max_weight, max_weight * epoch / anneal_epochs)


def total_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mi_loss: torch.Tensor,
    kl_weight: float = 1.0,
    mi_weight: float = 0.1,
) -> torch.Tensor:
    """
    Combine all loss terms.

    loss = recon + kl_weight * KL + mi_weight * (-MI)
                                     ^-- mi_loss is already negated
    """
    recon = reconstruction_loss(x_recon, x)
    kl = kl_divergence(mu, logvar)
    return recon + kl_weight * kl + mi_weight * mi_loss
