from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, scores.shape[-1])
    return np.argsort(-scores)[:k]


def jaccard_at_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int = 50) -> float:
    """Jaccard overlap of top-k genes from two score vectors."""
    a = set(top_k_indices(scores_a, k).tolist())
    b = set(top_k_indices(scores_b, k).tolist())
    union = a.union(b)
    if not union:
        return 1.0
    return len(a.intersection(b)) / len(union)


def _rankdata_desc(x: np.ndarray) -> np.ndarray:
    """Simple descending rank transform (0 is highest rank)."""
    order = np.argsort(-x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman_rank_correlation(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Spearman correlation computed from descending ranks."""
    if scores_a.shape != scores_b.shape:
        raise ValueError("score vectors must have the same shape")

    ra = _rankdata_desc(scores_a)
    rb = _rankdata_desc(scores_b)

    ra_mean = ra.mean()
    rb_mean = rb.mean()
    cov = np.mean((ra - ra_mean) * (rb - rb_mean))
    std = ra.std() * rb.std()
    if std == 0.0:
        return 0.0
    return float(cov / std)


def consensus_gene_ranking(score_runs: Sequence[np.ndarray]) -> np.ndarray:
    """Average absolute importance across runs for robust ranking."""
    if not score_runs:
        raise ValueError("score_runs cannot be empty")

    stacked = np.stack([np.abs(s) for s in score_runs], axis=0)
    return np.mean(stacked, axis=0)


def mean_pairwise_jaccard(score_runs: Sequence[np.ndarray], k: int = 50) -> float:
    if len(score_runs) < 2:
        return 1.0

    vals = []
    for i in range(len(score_runs)):
        for j in range(i + 1, len(score_runs)):
            vals.append(jaccard_at_k(score_runs[i], score_runs[j], k=k))
    return float(np.mean(vals)) if vals else 1.0


def mean_pairwise_spearman(score_runs: Sequence[np.ndarray]) -> float:
    if len(score_runs) < 2:
        return 1.0

    vals = []
    for i in range(len(score_runs)):
        for j in range(i + 1, len(score_runs)):
            vals.append(spearman_rank_correlation(score_runs[i], score_runs[j]))
    return float(np.mean(vals)) if vals else 1.0
