"""
MINE-based nonlinear module extraction (Improvement 5).

After training the cVAE, use MINE to estimate I(x_g; z_d) for every
gene-dimension pair. This captures nonlinear dependencies that Pearson
correlation misses.

Algorithm:
  1. Pass all samples through the encoder → get z (N x D)
  2. For each dimension d:
     a. For each gene g (or batches of genes):
        - Train a small MINE network on (x_g, z_d) for a few epochs
        - Final MI estimate = I_hat(x_g; z_d)
  3. Assemble into G x D MI matrix

Optimization:
  - Instead of one network per gene, we train one network per dimension
    that takes (x_g_scalar, z_d_scalar) as input
  - This 2D→1 network is shared across genes, trained sequentially
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Optional


class GeneLatentMINE(nn.Module):
    """
    Small MINE for estimating I(x_g; z_d) where both are 1D scalars.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_scalar: torch.Tensor, z_scalar: torch.Tensor) -> torch.Tensor:
        """Both inputs: (N, 1)"""
        xz = torch.cat([x_scalar, z_scalar], dim=1)
        return self.net(xz)


def estimate_single_mi(
    x_g: torch.Tensor,
    z_d: torch.Tensor,
    hidden_dim: int = 128,
    n_epochs: int = 50,
    lr: float = 1e-3,
    ema_alpha: float = 0.01,
) -> float:
    """
    Estimate I(x_g; z_d) using MINE with EMA bias correction.

    Args:
        x_g: (N,) gene expression values across all samples
        z_d: (N,) latent dimension values across all samples
        hidden_dim: MINE network hidden dimension
        n_epochs: Training epochs for the estimator
        lr: Learning rate
        ema_alpha: EMA decay

    Returns:
        Estimated MI (nats)
    """
    device = x_g.device
    N = x_g.size(0)

    x_in = x_g.unsqueeze(1)  # (N, 1)
    z_in = z_d.unsqueeze(1)  # (N, 1)

    mine_net = GeneLatentMINE(hidden_dim).to(device)
    optimizer = torch.optim.Adam(mine_net.parameters(), lr=lr)
    ema = torch.tensor(1.0, device=device)

    best_mi = -float("inf")

    for epoch in range(n_epochs):
        mine_net.train()
        optimizer.zero_grad()

        # Joint
        joint = mine_net(x_in, z_in)

        # Marginal (shuffle x)
        perm = torch.randperm(N, device=device)
        marginal = mine_net(x_in[perm], z_in)

        et = marginal.exp().mean()
        ema = (1 - ema_alpha) * ema + ema_alpha * et.detach()

        mi_estimate = joint.mean() - (et / ema.detach()).log()
        loss = -mi_estimate  # Maximize MI
        loss.backward()
        optimizer.step()

        if mi_estimate.item() > best_mi:
            best_mi = mi_estimate.item()

    return max(best_mi, 0.0)  # MI >= 0


@torch.no_grad()
def get_latent_activations(
    model,
    dataloader,
    device: str = "cpu",
) -> tuple:
    """
    Pass all data through encoder to get latent activations.

    Returns:
        X: (N, G) all gene expression values
        Z: (N, D) all latent means
    """
    model.to(device)
    model.eval()

    all_x, all_z = [], []
    for batch in dataloader:
        x, c = batch[0].to(device), batch[1].to(device)
        mu, _ = model.encoder(x, c)
        all_x.append(x.cpu())
        all_z.append(mu.cpu())

    return torch.cat(all_x, dim=0), torch.cat(all_z, dim=0)


def mine_extraction(
    model,
    dataloader,
    gene_names: List[str],
    device: str = "cpu",
    hidden_dim: int = 128,
    n_epochs: int = 50,
    lr: float = 1e-3,
    top_n_genes_per_dim: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    MINE-based nonlinear module extraction.

    This is the 4th extraction method — captures nonlinear gene-latent
    dependencies that Pearson correlation misses.

    Strategy to keep it tractable:
      1. For each dimension d:
         a. Compute Pearson |r| for all genes (fast)
         b. Take top_n_genes_per_dim genes by |r|
         c. For those genes only, compute full MINE estimate
         d. Set MI=0 for remaining genes (below detection threshold)

    This focuses MINE computation where it matters most and avoids
    G×D×n_epochs forward passes for all 32K genes.

    Args:
        model: Trained cVAE
        dataloader: Full dataset loader
        gene_names: List of gene identifiers
        device: Compute device
        hidden_dim: MINE network hidden dim
        n_epochs: MINE training epochs per gene-dim pair
        lr: MINE learning rate
        top_n_genes_per_dim: Genes to evaluate per dimension
        verbose: Print progress

    Returns:
        DataFrame (genes × dimensions) with MI estimates
    """
    if verbose:
        print("\n  MINE-based nonlinear extraction...")

    X, Z = get_latent_activations(model, dataloader, device)
    N, G = X.shape
    D = Z.shape[1]

    # Initialize result matrix
    mi_matrix = np.zeros((G, D), dtype=np.float32)

    X_dev = X.to(device)
    Z_dev = Z.to(device)

    for d in range(D):
        z_d = Z_dev[:, d]

        # Step 1: Fast Pearson pre-screen
        z_centered = z_d - z_d.mean()
        z_std = z_centered.std()
        if z_std < 1e-8:
            if verbose and d % 16 == 0:
                print(f"    Dim {d:03d}: collapsed (std={z_std:.2e}), skipping")
            continue

        # Pearson r for all genes at once
        X_centered = X_dev - X_dev.mean(dim=0, keepdim=True)
        X_std = X_centered.std(dim=0)
        valid_mask = X_std > 1e-8
        pearson_r = torch.zeros(G, device=device)
        pearson_r[valid_mask] = (
            (X_centered[:, valid_mask] * z_centered.unsqueeze(1)).mean(dim=0)
            / (X_std[valid_mask] * z_std)
        )

        # Step 2: Select top genes by |r| for detailed MINE estimation
        abs_r = pearson_r.abs()
        top_k = min(top_n_genes_per_dim, (abs_r > 0).sum().item())
        if top_k == 0:
            continue

        _, top_indices = abs_r.topk(top_k)

        # Step 3: Full MINE for selected genes
        for gene_idx in top_indices:
            x_g = X_dev[:, gene_idx]
            mi_val = estimate_single_mi(
                x_g, z_d,
                hidden_dim=hidden_dim,
                n_epochs=n_epochs,
                lr=lr,
            )
            mi_matrix[gene_idx.item(), d] = mi_val

        if verbose and d % 16 == 0:
            max_mi = mi_matrix[:, d].max()
            n_nonzero = (mi_matrix[:, d] > 0).sum()
            print(f"    Dim {d:03d}: {n_nonzero} genes evaluated, max MI={max_mi:.4f}")

    # Build DataFrame
    dim_names = [f"Dim_{d:03d}" for d in range(D)]
    mi_df = pd.DataFrame(mi_matrix, index=gene_names, columns=dim_names)

    if verbose:
        n_active = (mi_df.max(axis=0) > 0.01).sum()
        print(f"\n  MINE extraction complete: {n_active}/{D} dimensions with MI > 0.01")

    return mi_df
