"""
Enhanced loss functions for the MINE-improved cVAE.

Total loss = Reconstruction + β·KL + λ_MI·L_MI + λ_TC·L_TC

Where L_MI is now dimension-wise MI (not global) and L_TC is
the pairwise Total Correlation penalty for disentanglement.
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss."""
    return F.mse_loss(x_recon, x, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), averaged over batch."""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def compute_kl_weight(epoch: int, anneal_epochs: int, max_weight: float) -> float:
    """Linear KL annealing."""
    if anneal_epochs <= 0:
        return max_weight
    return min(max_weight, max_weight * epoch / anneal_epochs)


def total_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dimwise_mi_loss: torch.Tensor,
    tc_loss: torch.Tensor,
    kl_weight: float = 1.0,
    mi_weight: float = 0.1,
    tc_weight: float = 0.05,
) -> torch.Tensor:
    """
    Combined loss with all four terms.

    Args:
        x_recon:          Reconstructed expression
        x:                Original expression
        mu, logvar:       Latent distribution parameters
        dimwise_mi_loss:  Dimension-wise MI (already negated — minimize to maximize MI)
        tc_loss:          Pairwise TC penalty (positive — minimize to reduce redundancy)
        kl_weight:        β for KL term (annealed)
        mi_weight:        λ_MI for MI maximization
        tc_weight:        λ_TC for disentanglement

    Returns:
        Combined scalar loss
    """
    recon = reconstruction_loss(x_recon, x)
    kl = kl_divergence(mu, logvar)
    return recon + kl_weight * kl + mi_weight * dimwise_mi_loss + tc_weight * tc_loss
