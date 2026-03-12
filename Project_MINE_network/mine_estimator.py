"""
Batched MINE (Mutual Information Neural Estimation) for gene-gene pairs.

Core idea
---------
Instead of computing MI via histogram binning (which discretises the data and
loses information), we train a small neural network T(x_i, x_j) to estimate
MI directly on the continuous expression values.

Efficiency trick
----------------
Training one network per gene pair is far too slow for millions of candidates.
Instead, we train B independent networks simultaneously using batched tensor
operations.  Each "network" is a separate set of weights in a (B, H, D) tensor,
and torch.bmm parallelises all B forward/backward passes in one GPU kernel.

Reference: Belghazi et al., "Mutual Information Neural Estimation", ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class BatchedMINE(nn.Module):
    """
    B independent statistics networks T_k(a, b) -> scalar, k = 1..B.

    Each network: 2 -> hidden_dim -> hidden_dim -> 1  (ELU activations).
    Represented as batched weight tensors for fully parallel training.
    """

    def __init__(self, batch_size: int, hidden_dim: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # Xavier-scale initialization
        s1 = (2.0 / (2 + hidden_dim)) ** 0.5
        s2 = (2.0 / (2 * hidden_dim)) ** 0.5
        s3 = (2.0 / (hidden_dim + 1)) ** 0.5

        self.W1 = nn.Parameter(torch.randn(batch_size, hidden_dim, 2) * s1)
        self.b1 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(batch_size, hidden_dim, hidden_dim) * s2)
        self.b2 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(batch_size, 1, hidden_dim) * s3)
        self.b3 = nn.Parameter(torch.zeros(batch_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N_samples, 2) -> (B, N_samples) scores.
        """
        # Layer 1
        h = torch.bmm(x, self.W1.transpose(1, 2)) + self.b1.unsqueeze(1)
        h = F.elu(h)
        # Layer 2
        h = torch.bmm(h, self.W2.transpose(1, 2)) + self.b2.unsqueeze(1)
        h = F.elu(h)
        # Layer 3 -> scalar
        out = torch.bmm(h, self.W3.transpose(1, 2)) + self.b3.unsqueeze(1)
        return out.squeeze(-1)  # (B, N)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch MI estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mi_batch(
    gene_i_data: torch.Tensor,
    gene_j_data: torch.Tensor,
    hidden_dim: int = 64,
    n_epochs: int = 200,
    lr: float = 1e-3,
    ema_alpha: float = 0.01,
    grad_clip: float = 1.0,
    n_eval_shuffles: int = 5,
) -> np.ndarray:
    """
    Estimate MI for B gene pairs simultaneously.

    Parameters
    ----------
    gene_i_data : (B, N_samples) float tensor — expression of gene i per pair
    gene_j_data : (B, N_samples) float tensor — expression of gene j per pair
    hidden_dim  : int — width of T network hidden layers
    n_epochs    : int — SGD training epochs
    lr          : float — learning rate
    ema_alpha   : float — EMA decay for bias correction
    grad_clip   : float — max gradient norm
    n_eval_shuffles : int — marginal shuffles averaged for final MI

    Returns
    -------
    mi_estimates : (B,) numpy array of MI in nats (clamped >= 0)
    """
    B, N = gene_i_data.shape
    device = gene_i_data.device

    model = BatchedMINE(B, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = torch.ones(B, device=device)

    for _ in range(n_epochs):
        optimizer.zero_grad()

        # Joint: matched (gene_i_sample_k, gene_j_sample_k)
        joint = torch.stack([gene_i_data, gene_j_data], dim=2)  # (B, N, 2)

        # Marginal: shuffle gene_j independently per pair
        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        gj_shuf = torch.gather(gene_j_data, 1, perm)
        marginal = torch.stack([gene_i_data, gj_shuf], dim=2)

        T_joint = model(joint)        # (B, N)
        T_marginal = model(marginal)  # (B, N)

        # DV bound with EMA bias correction (Paper §3.2)
        joint_mean = T_joint.mean(dim=1)       # (B,)
        et = T_marginal.exp().mean(dim=1)      # (B,)

        ema_snap = ema.detach().clone()
        with torch.no_grad():
            ema.copy_((1 - ema_alpha) * ema + ema_alpha * et.detach())

        mi = joint_mean - (et / ema_snap).log()
        loss = -mi.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # Final evaluation: average over several marginal shuffles for stability
    with torch.no_grad():
        mi_acc = torch.zeros(B, device=device)
        for _ in range(n_eval_shuffles):
            joint = torch.stack([gene_i_data, gene_j_data], dim=2)
            perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
            gj_shuf = torch.gather(gene_j_data, 1, perm)
            marginal = torch.stack([gene_i_data, gj_shuf], dim=2)
            T_j = model(joint)
            T_m = model(marginal)
            mi_acc += T_j.mean(dim=1) - T_m.exp().mean(dim=1).log()
        mi_acc /= n_eval_shuffles

    return mi_acc.clamp(min=0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# High-level: estimate MI for a list of candidate gene pairs
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mi_for_pairs(
    expr_matrix: np.ndarray,
    pair_indices: np.ndarray,
    mine_cfg,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """
    Estimate MI for all candidate pairs using batched MINE.

    Parameters
    ----------
    expr_matrix  : (n_genes, n_samples) float32 array — Z-scored expression
    pair_indices : (n_pairs, 2) int array — gene index pairs
    mine_cfg     : MINEConfig dataclass
    device       : torch.device
    verbose      : bool

    Returns
    -------
    mi_values : (n_pairs,) float array — MI estimates in nats
    """
    n_pairs = len(pair_indices)
    batch_size = mine_cfg.batch_pairs
    n_batches = (n_pairs + batch_size - 1) // batch_size
    mi_all = np.zeros(n_pairs, dtype=np.float32)

    expr_t = torch.from_numpy(expr_matrix).float().to(device)

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_pairs)
        actual_B = end - start

        idx_i = pair_indices[start:end, 0]
        idx_j = pair_indices[start:end, 1]

        gi = expr_t[idx_i]  # (actual_B, N_samples)
        gj = expr_t[idx_j]

        mi_batch = estimate_mi_batch(
            gi, gj,
            hidden_dim=mine_cfg.hidden_dim,
            n_epochs=mine_cfg.n_epochs,
            lr=mine_cfg.learning_rate,
            ema_alpha=mine_cfg.ema_alpha,
            grad_clip=mine_cfg.gradient_clip,
            n_eval_shuffles=mine_cfg.n_eval_shuffles,
        )
        mi_all[start:end] = mi_batch[:actual_B]

        if verbose and (b + 1) % max(1, n_batches // 10) == 0:
            pct = (b + 1) / n_batches * 100
            print(f"  MINE progress: {b+1}/{n_batches} batches ({pct:.0f}%)")

    return mi_all


# ═══════════════════════════════════════════════════════════════════════════════
# Null distribution via MINE on permuted pairs
# ═══════════════════════════════════════════════════════════════════════════════

def build_mine_null(
    expr_matrix: np.ndarray,
    mine_cfg,
    n_permutations: int = 10_000,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> np.ndarray:
    """
    Build the empirical null distribution of MINE MI under independence.

    For each null trial: pick two random genes, permute one, estimate MI.
    Under Z-scored data the null is approximately gene-pair-agnostic.

    Returns
    -------
    null_mi : (n_permutations,) float array
    """
    rng = np.random.default_rng(seed)
    n_genes, n_samples = expr_matrix.shape

    # Sample random gene pairs for the null
    gi_idx = rng.integers(0, n_genes, size=n_permutations)
    gj_idx = rng.integers(0, n_genes, size=n_permutations)

    # For each null pair, we permute gene j before MINE estimation
    expr_t = torch.from_numpy(expr_matrix).float().to(device)
    batch_size = mine_cfg.batch_pairs
    n_batches = (n_permutations + batch_size - 1) // batch_size
    null_mi = np.zeros(n_permutations, dtype=np.float32)

    if verbose:
        print(f"  Building MINE null distribution: {n_permutations} permutations "
              f"in {n_batches} batches...")

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_permutations)
        B = e - s

        gi = expr_t[gi_idx[s:e]]  # (B, N_samples)
        gj = expr_t[gj_idx[s:e]]

        # Permute gene_j samples (break dependence for null)
        perm = torch.argsort(torch.rand(B, n_samples, device=device), dim=1)
        gj_perm = torch.gather(gj, 1, perm)

        mi_batch = estimate_mi_batch(
            gi, gj_perm,
            hidden_dim=mine_cfg.hidden_dim,
            n_epochs=mine_cfg.n_epochs,
            lr=mine_cfg.learning_rate,
            ema_alpha=mine_cfg.ema_alpha,
            grad_clip=mine_cfg.gradient_clip,
            n_eval_shuffles=mine_cfg.n_eval_shuffles,
        )
        null_mi[s:e] = mi_batch[:B]

        if verbose and (b + 1) % max(1, n_batches // 5) == 0:
            print(f"  Null progress: {b+1}/{n_batches} batches")

    if verbose:
        print(f"  Null MI: mean={null_mi.mean():.4f}, std={null_mi.std():.4f}, "
              f"99.9th={np.percentile(null_mi, 99.9):.4f}")

    return null_mi
