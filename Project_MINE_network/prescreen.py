"""
Fast correlation pre-screening to reduce candidate gene pairs.

With 32K genes there are ~537 million unique pairs.  Training a MINE network
per pair is far too slow, so we first compute pairwise Pearson (or Spearman)
correlation on the continuous expression data and keep only pairs with
|r| > threshold.  MINE then refines these candidates.

This is sound because:
  - Pairs with near-zero linear correlation AND near-zero nonlinear MI are
    overwhelmingly the majority.
  - MINE's value-add is for pairs with moderate |r| that mask stronger
    nonlinear dependencies.  Very low |r| pairs almost always have low MI too.
  - Threshold of 0.3 is conservative; most significant MI edges have |r| > 0.3.
"""

import numpy as np
from joblib import Parallel, delayed
import time


def _pearson_row(i, X, n_genes):
    """Pearson |r| between gene i and genes i+1..n (upper triangle only)."""
    xi = X[i]
    xi_centered = xi - xi.mean()
    xi_norm = np.sqrt(np.dot(xi_centered, xi_centered))
    if xi_norm == 0:
        return np.zeros(n_genes - i - 1, dtype=np.float32)

    results = np.empty(n_genes - i - 1, dtype=np.float32)
    for j_offset in range(n_genes - i - 1):
        j = i + 1 + j_offset
        xj = X[j]
        xj_centered = xj - xj.mean()
        xj_norm = np.sqrt(np.dot(xj_centered, xj_centered))
        if xj_norm == 0:
            results[j_offset] = 0.0
        else:
            results[j_offset] = abs(np.dot(xi_centered, xj_centered) / (xi_norm * xj_norm))
    return results


def prescreen_pairs(
    expr_matrix: np.ndarray,
    method: str = "pearson",
    threshold: float = 0.3,
    max_pairs: int = 500_000,
    n_jobs: int = -1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Return gene index pairs (i, j) with |correlation| > threshold.

    Parameters
    ----------
    expr_matrix : (n_genes, n_samples) float array
    method      : "pearson" or "spearman"
    threshold   : minimum |r| to keep
    max_pairs   : hard cap — if exceeded, raise threshold dynamically
    n_jobs      : CPU cores for parallelism
    verbose     : print progress

    Returns
    -------
    pair_indices : (n_pairs, 2) int array, each row is (gene_i, gene_j)
    """
    n_genes, n_samples = expr_matrix.shape

    if method == "spearman":
        # Rank-transform each gene, then Pearson on ranks = Spearman
        from scipy.stats import rankdata
        expr_matrix = np.array([rankdata(expr_matrix[g]) for g in range(n_genes)])

    if verbose:
        print(f"  Pre-screening {n_genes} genes ({method}, |r| > {threshold})...")

    t0 = time.time()

    # Parallel row-wise computation
    row_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_pearson_row)(i, expr_matrix, n_genes)
        for i in range(n_genes - 1)
    )

    # Collect pairs above threshold
    pairs = []
    for i, r_vals in enumerate(row_results):
        above = np.where(r_vals > threshold)[0]
        for offset in above:
            j = i + 1 + offset
            pairs.append((i, j))

    pair_indices = np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Pre-screen: {len(pair_indices):,} pairs above |r| > {threshold} "
              f"(from {n_genes * (n_genes - 1) // 2:,} total) in {elapsed:.1f}s")

    # Dynamic threshold raise if too many pairs
    if len(pair_indices) > max_pairs and len(pair_indices) > 0:
        # Collect all |r| values for the selected pairs and pick a higher cutoff
        r_values = []
        for i, r_vals in enumerate(row_results):
            above = np.where(r_vals > threshold)[0]
            r_values.extend(r_vals[above])
        r_values = np.array(r_values)
        new_threshold = np.sort(r_values)[-max_pairs]
        mask = r_values >= new_threshold
        pair_indices = pair_indices[mask]
        if verbose:
            print(f"  Capped to {max_pairs:,} pairs "
                  f"(raised threshold to |r| > {new_threshold:.3f})")

    return pair_indices
