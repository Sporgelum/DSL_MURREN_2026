"""
MINE-Enhanced Mutual Information Estimators for cVAE Regularization.

Improvements over the original mi_regularizer.py (see README.md for details):

  1. EMA bias correction (Paper §3.2)
     - Exponential moving average on the partition function denominator
     - Reduces gradient bias from mini-batch estimation

  2. Dimension-wise MI estimation (Improvement 3)
     - Per-dimension statistics networks: I(X; z_j) for each j
     - Ensures every latent dimension carries information

  3. Pairwise MI estimation for disentanglement (Improvement 4)
     - Estimates I(z_i; z_j) for random pairs
     - Total Correlation penalty pushes dimensions apart

  4. Deeper statistics network with noise injection (Improvement 6)
     - Two-stage: gene projector + joint estimator
     - Gaussian noise between layers (Paper §8.1.5 Table 15)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════

class DeepStatisticsNetwork(nn.Module):
    """
    Two-stage statistics network for high-dimensional gene expression.

    Stage 1 (gene projector): x_dim -> proj_dim  (compress gene space)
    Stage 2 (joint estimator): (proj_dim + z_dim) -> hidden -> 1

    Gaussian noise injection between layers improves generalization
    (Paper §8.1.5 Table 15).
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        proj_dim: int = 512,
        hidden_dim: int = 256,
        noise_std: float = 0.3,
    ):
        super().__init__()
        self.noise_std = noise_std

        # Stage 1: project gene space down
        self.gene_projector = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, proj_dim),
            nn.ELU(inplace=True),
        )

        # Stage 2: estimate T(projected_x, z)
        self.joint_estimator = nn.Sequential(
            nn.Linear(proj_dim + z_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Project gene expression
        h = self.gene_projector(x)

        # Add Gaussian noise during training (regularization)
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        # Joint estimation
        hz = torch.cat([h, z], dim=1)
        return self.joint_estimator(hz)


class LightStatisticsNetwork(nn.Module):
    """
    Lightweight statistics network for per-dimension MI estimation.

    Used when estimating I(X; z_j) for a single latent dimension (z_dim=1).
    Shares the gene projector to save memory.
    """

    def __init__(self, proj_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim + 1, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_proj: torch.Tensor, z_single: torch.Tensor) -> torch.Tensor:
        hz = torch.cat([h_proj, z_single], dim=1)
        return self.net(hz)


class PairwiseStatisticsNetwork(nn.Module):
    """
    Statistics network for estimating I(z_i; z_j) between latent pairs.
    Much smaller — operates on 2D input only.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        zz = torch.cat([z_i, z_j], dim=1)
        return self.net(zz)


# ═══════════════════════════════════════════════════════════════════════════════
# EMA-CORRECTED MINE ESTIMATION (Paper §3.2)
# ═══════════════════════════════════════════════════════════════════════════════

class EMAMINEEstimator(nn.Module):
    """
    MINE estimator with exponential moving average bias correction.

    The standard MINE gradient has bias because:
        ∇ log E[exp(T)] = E[∇T · exp(T)] / E[exp(T)]
    In mini-batch: E[A/B] ≠ E[A]/E[B]

    Fix: track E[exp(T)] with an EMA, use it as the denominator.

    This module wraps a statistics network and provides:
      - Forward: compute MI lower bound with EMA correction
      - The EMA state is maintained across batches within an epoch
    """

    def __init__(self, ema_alpha: float = 0.01):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.register_buffer("ema", torch.tensor(1.0))

    def compute_mine_loss(
        self,
        stats_net: nn.Module,
        x_or_h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute EMA-corrected MINE lower bound (negated for minimization).

        Args:
            stats_net: T(x, z) network
            x_or_h: Input or projected input (batch, dim)
            z: Latent (batch, z_dim) or single dimension (batch, 1)

        Returns:
            -I_hat: Negative MI estimate for gradient descent
        """
        # Joint: T(x, z) from p(x, z)
        joint_scores = stats_net(x_or_h, z)

        # Marginal: T(x', z) where x' breaks the dependence
        perm = torch.randperm(x_or_h.size(0), device=x_or_h.device)
        x_shuffled = x_or_h[perm]
        marginal_scores = stats_net(x_shuffled, z)

        # EMA-corrected MINE (Paper §3.2)
        et = marginal_scores.exp().mean()

        # Read EMA value BEFORE updating (avoids in-place version counter issue)
        ema_val = self.ema.detach().clone()

        if self.training:
            with torch.no_grad():
                self.ema.copy_(
                    (1 - self.ema_alpha) * self.ema + self.ema_alpha * et.detach()
                )

        # Use pre-update EMA in denominator for bias correction
        mi_estimate = joint_scores.mean() - (et / ema_val).log()

        return -mi_estimate


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION-WISE MI (Improvement 3)
# ═══════════════════════════════════════════════════════════════════════════════

class DimensionWiseMINE(nn.Module):
    """
    Estimates I(X; z_j) for each latent dimension j using lightweight
    per-dimension statistics networks that share a gene projector.

    Architecture:
      1. Shared gene projector: x (32K) -> h (512)
      2. Per-dimension networks: (h, z_j) -> scalar, for j = 1..D

    This ensures every dimension carries information about the input,
    directly preventing per-dimension latent collapse.
    """

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        proj_dim: int = 512,
        per_dim_hidden: int = 128,
        noise_std: float = 0.3,
        ema_alpha: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_std = noise_std

        # Shared gene projector
        self.gene_projector = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, proj_dim),
            nn.ELU(inplace=True),
        )

        # Per-dimension statistics networks
        self.dim_nets = nn.ModuleList([
            LightStatisticsNetwork(proj_dim, per_dim_hidden)
            for _ in range(latent_dim)
        ])

        # Per-dimension EMA trackers
        self.ema_alpha = ema_alpha
        self.register_buffer("emas", torch.ones(latent_dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute average dimension-wise MI (negated for minimization).

        Returns: -mean_j(I_hat(X; z_j))
        """
        # Project genes once (shared)
        h = self.gene_projector(x)
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        # Shuffle for marginal (shared across dimensions for consistency)
        perm = torch.randperm(x.size(0), device=x.device)
        h_shuffled = h[perm]

        mi_sum = torch.tensor(0.0, device=x.device)

        for j in range(self.latent_dim):
            z_j = z[:, j:j+1]  # (batch, 1)

            joint_score = self.dim_nets[j](h, z_j)
            marginal_score = self.dim_nets[j](h_shuffled, z_j)

            et = marginal_score.exp().mean()

            # Read EMA BEFORE update
            ema_j = self.emas[j].detach().clone()

            if self.training:
                with torch.no_grad():
                    self.emas[j] = (
                        (1 - self.ema_alpha) * self.emas[j]
                        + self.ema_alpha * et.detach()
                    )

            mi_j = joint_score.mean() - (et / ema_j).log()
            mi_sum = mi_sum + mi_j

        return -mi_sum / self.latent_dim


# ═══════════════════════════════════════════════════════════════════════════════
# PAIRWISE MI FOR DISENTANGLEMENT (Improvement 4)
# ═══════════════════════════════════════════════════════════════════════════════

class PairwiseMINE(nn.Module):
    """
    Estimates I(z_i; z_j) for random pairs of latent dimensions.

    Minimizing this drives the Total Correlation toward zero, ensuring
    each module captures unique (non-redundant) biological programs.

    Computational trick: we don't compute all D*(D-1)/2 pairs.
    Instead, each forward pass samples `n_pairs` random pairs.
    """

    def __init__(
        self,
        latent_dim: int,
        n_pairs: int = 32,
        hidden_dim: int = 64,
        ema_alpha: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_pairs = min(n_pairs, latent_dim * (latent_dim - 1) // 2)

        # Single shared pairwise network (dimension-agnostic)
        self.pair_net = PairwiseStatisticsNetwork(hidden_dim)
        self.ema_alpha = ema_alpha
        self.register_buffer("ema", torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute average pairwise MI for sampled pairs (positive = to minimize).

        Returns: mean I_hat(z_i; z_j) for sampled (i,j) pairs
        """
        device = z.device
        batch_size = z.size(0)

        # Sample random pairs of dimensions
        all_dims = torch.arange(self.latent_dim, device=device)
        # Generate n_pairs random (i, j) with i < j
        pairs = []
        for _ in range(self.n_pairs):
            idx = torch.randperm(self.latent_dim, device=device)[:2]
            pairs.append(idx.sort().values)
        pairs = torch.stack(pairs)  # (n_pairs, 2)

        tc_sum = torch.tensor(0.0, device=device)

        for k in range(self.n_pairs):
            i, j = pairs[k]
            z_i = z[:, i:i+1]  # (batch, 1)
            z_j = z[:, j:j+1]  # (batch, 1)

            joint_score = self.pair_net(z_i, z_j)

            perm = torch.randperm(batch_size, device=device)
            z_i_shuffled = z_i[perm]
            marginal_score = self.pair_net(z_i_shuffled, z_j)

            et = marginal_score.exp().mean()

            # Read EMA BEFORE update
            ema_val = self.ema.detach().clone()

            if self.training:
                with torch.no_grad():
                    self.ema.copy_(
                        (1 - self.ema_alpha) * self.ema
                        + self.ema_alpha * et.detach()
                    )

            mi_ij = joint_score.mean() - (et / ema_val).log()
            tc_sum = tc_sum + mi_ij

        return tc_sum / self.n_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MINE (EMA-corrected version of the original)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalMINE(nn.Module):
    """
    EMA-corrected global I(X; Z) estimator.

    Uses the deep statistics network (Improvement 6) + EMA bias correction
    (Improvement 1). Drop-in replacement for the original MINEEstimator.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        proj_dim: int = 512,
        hidden_dim: int = 256,
        noise_std: float = 0.3,
        ema_alpha: float = 0.01,
    ):
        super().__init__()
        self.stats_net = DeepStatisticsNetwork(x_dim, z_dim, proj_dim, hidden_dim, noise_std)
        self.ema_estimator = EMAMINEEstimator(ema_alpha)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns -I_hat(X; Z) for gradient descent."""
        return self.ema_estimator.compute_mine_loss(self.stats_net, x, z)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_mine_components(mine_cfg, input_dim: int, latent_dim: int):
    """
    Build all MINE components from config.

    Returns:
        global_mine:    GlobalMINE for I(X; Z) — backward compatible
        dimwise_mine:   DimensionWiseMINE for per-dim I(X; z_j)
        pairwise_mine:  PairwiseMINE for I(z_i; z_j) TC penalty
    """
    global_mine = GlobalMINE(
        x_dim=input_dim,
        z_dim=latent_dim,
        proj_dim=mine_cfg.proj_dim,
        hidden_dim=mine_cfg.hidden_dim,
        noise_std=mine_cfg.noise_std,
        ema_alpha=mine_cfg.ema_alpha,
    )

    dimwise_mine = DimensionWiseMINE(
        x_dim=input_dim,
        latent_dim=latent_dim,
        proj_dim=mine_cfg.proj_dim,
        per_dim_hidden=mine_cfg.per_dim_hidden,
        noise_std=mine_cfg.noise_std,
        ema_alpha=mine_cfg.ema_alpha,
    )

    pairwise_mine = PairwiseMINE(
        latent_dim=latent_dim,
        n_pairs=mine_cfg.tc_n_pairs,
        hidden_dim=mine_cfg.tc_hidden_dim,
        ema_alpha=mine_cfg.ema_alpha,
    )

    return global_mine, dimwise_mine, pairwise_mine
