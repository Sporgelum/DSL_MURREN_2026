"""
Mutual Information (MI) Estimator for regularizing the cVAE latent space.

Implements the MINE (Mutual Information Neural Estimation) approach to
maximize I(X; Z), preventing latent collapse and encouraging each latent
dimension to capture a distinct biological signal.
"""

import torch
import torch.nn as nn


class MINEEstimator(nn.Module):
    """
    MINE-based MI lower bound estimator.

    Estimates I(X; Z) by training a statistics network T(x, z) such that:
        I(X; Z) >= E[T(x, z)] - log(E[exp(T(x', z))])
    where x' is drawn from the marginal p(x) (achieved by shuffling).
    """

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        xz = torch.cat([x, z], dim=1)
        return self.network(xz)


class NWJEstimator(nn.Module):
    """
    Nguyen-Wainwright-Jordan (NWJ / f-GAN KL) MI estimator.

    Provides a tighter lower bound than MINE in some settings:
        I(X; Z) >= E[T(x, z)] - e^{-1} * E[exp(T(x', z))]
    """

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        xz = torch.cat([x, z], dim=1)
        return self.network(xz)


def compute_mi_loss(
    mi_estimator: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    estimator_type: str = "mine",
) -> torch.Tensor:
    """
    Compute the MI lower bound (negated, since we maximize MI = minimize -MI).

    Args:
        mi_estimator: The statistics network T(x, z)
        x: Input expression batch (n, input_dim)
        z: Latent batch (n, latent_dim)
        estimator_type: 'mine' or 'nwj'

    Returns:
        Negative MI estimate (to be minimized).
    """
    # Joint: T(x, z)  — samples from p(x, z)
    joint_scores = mi_estimator(x, z)

    # Marginal: T(x', z) — x' is shuffled to break the dependence
    perm = torch.randperm(x.size(0), device=x.device)
    x_shuffled = x[perm]
    marginal_scores = mi_estimator(x_shuffled, z)

    if estimator_type == "mine":
        # MINE: E_joint[T] - log(E_marginal[exp(T)])
        mi_estimate = joint_scores.mean() - torch.logsumexp(
            marginal_scores, dim=0
        ) + torch.log(torch.tensor(float(x.size(0)), device=x.device))
    elif estimator_type == "nwj":
        # NWJ: E_joint[T] - e^{-1} * E_marginal[exp(T)]
        mi_estimate = joint_scores.mean() - (
            torch.exp(marginal_scores - 1).mean()
        )
    else:
        raise ValueError(f"Unknown MI estimator type: {estimator_type}")

    # Negate because we want to *maximize* MI (minimize -MI)
    return -mi_estimate


def build_mi_estimator(mi_cfg, input_dim: int, latent_dim: int) -> nn.Module:
    """Factory: create MI estimator from config."""
    if mi_cfg.mi_estimator == "mine":
        return MINEEstimator(input_dim, latent_dim, mi_cfg.mi_hidden_dim)
    elif mi_cfg.mi_estimator == "nwj":
        return NWJEstimator(input_dim, latent_dim, mi_cfg.mi_hidden_dim)
    else:
        raise ValueError(f"Unknown MI estimator: {mi_cfg.mi_estimator}")
