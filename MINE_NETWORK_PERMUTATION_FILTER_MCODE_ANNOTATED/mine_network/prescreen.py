"""
Fast correlation pre-screening to reduce candidate gene pairs.
==============================================================

With G genes there are G(G-1)/2 unique pairs.  For 32 000 genes that is
~537 million pairs.  Training a MINE network per pair would take weeks,
so we optionally pre-filter using fast linear correlation.

Justification (Section 4c)
---------------------------
Most gene pairs have near-zero linear correlation *and* near-zero MI.
MINE's advantage is for pairs that have moderate Pearson |r| but harbour
stronger nonlinear dependencies.  Filtering at |r| > 0.3 retains these
interesting pairs while eliminating the vast majority of truly independent
ones.

The pre-screen is **optional** (``PrescreenConfig.enabled``).  When disabled
(small gene sets or targeted analyses), all unique pairs are returned.

Implementation
--------------
Row-wise Pearson correlation is computed in parallel via ``joblib``.
For each gene i, |r(i, j)| is computed for all j > i (upper triangle only).
Pairs exceeding the threshold are collected.  If the total exceeds
``max_pairs``, the threshold is dynamically raised to cap the count.
"""

import numpy as np
from joblib import Parallel, delayed
import time


def _pearson_row(i: int, X: np.ndarray, n_genes: int) -> np.ndarray:
    """
    Compute |Pearson r| between gene i and genes i+1 … n-1.

    Parameters
    ----------
    i : int
        Row index of the query gene.
    X : np.ndarray, shape (n_genes, n_samples)
        Expression matrix (Z-scored or raw).
    n_genes : int
        Total number of genes.

    Returns
    -------
    np.ndarray, shape (n_genes - i - 1,)
        Absolute Pearson correlation with each subsequent gene.
    """
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
            results[j_offset] = abs(
                np.dot(xi_centered, xj_centered) / (xi_norm * xj_norm)
            )
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
    Return gene-index pairs (i, j) with |correlation| > threshold.

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_samples)
        Expression matrix (typically Z-scored).
    method : str
        ``"pearson"`` or ``"spearman"`` (rank-transform then Pearson).
    threshold : float
        Minimum |r| to keep a pair.
    max_pairs : int
        Hard cap.  If exceeded, the threshold is raised dynamically.
    n_jobs : int
        CPU cores for parallel computation (-1 = all).
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray, shape (n_pairs, 2), dtype int32
        Each row is (gene_i_index, gene_j_index) with i < j.
    """
    n_genes, n_samples = expr_matrix.shape

    if method == "spearman":
        from scipy.stats import rankdata
        expr_matrix = np.array(
            [rankdata(expr_matrix[g]) for g in range(n_genes)]
        )

    if verbose:
        total = n_genes * (n_genes - 1) // 2
        print(f"  Pre-screening {n_genes:,} genes ({method}, |r| > {threshold}, "
              f"{total:,} total pairs)...")

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

    pair_indices = (
        np.array(pairs, dtype=np.int32) if pairs
        else np.empty((0, 2), dtype=np.int32)
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"  Pre-screen: {len(pair_indices):,} pairs above |r| > {threshold} "
              f"in {elapsed:.1f}s")

    # Dynamic threshold raise if too many pairs
    if len(pair_indices) > max_pairs and len(pair_indices) > 0:
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


def all_pairs(n_genes: int) -> np.ndarray:
    """
    Generate all unique (i, j) pairs with i < j.

    Used when pre-screening is disabled.

    Parameters
    ----------
    n_genes : int
        Number of genes.

    Returns
    -------
    np.ndarray, shape (n_genes*(n_genes-1)//2, 2), dtype int32
    """
    rows, cols = np.triu_indices(n_genes, k=1)
    return np.column_stack([rows, cols]).astype(np.int32)
