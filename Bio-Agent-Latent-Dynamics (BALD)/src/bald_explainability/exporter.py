from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def ranked_genes_dataframe(
    group_consensus_scores: Dict[str, np.ndarray],
    gene_names: Sequence[str],
    top_k: int = 100,
    method: str = "model_attribution",
) -> pd.DataFrame:
    """Build a tidy table of top-ranked genes per group."""
    n_genes = len(gene_names)
    rows: List[dict] = []

    for group_name in sorted(group_consensus_scores.keys()):
        scores = np.asarray(group_consensus_scores[group_name])
        if scores.ndim != 1:
            raise ValueError("Each group score vector must be rank-1")
        if scores.shape[0] != n_genes:
            raise ValueError("gene_names length must match score vector length")

        k = min(top_k, n_genes)
        top_idx = np.argsort(-scores)[:k]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "group": str(group_name),
                    "gene": str(gene_names[idx]),
                    "gene_index": int(idx),
                    "score": float(scores[idx]),
                    "rank": int(rank),
                    "method": method,
                }
            )

    return pd.DataFrame(rows)


def export_ranked_genes_csv(
    group_consensus_scores: Dict[str, np.ndarray],
    gene_names: Sequence[str],
    output_csv: Path,
    top_k: int = 100,
    method: str = "model_attribution",
    extra_columns: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Export top-ranked genes per group to CSV and return the DataFrame."""
    df = ranked_genes_dataframe(
        group_consensus_scores=group_consensus_scores,
        gene_names=gene_names,
        top_k=top_k,
        method=method,
    )

    if extra_columns:
        for key, value in extra_columns.items():
            df[key] = value

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
