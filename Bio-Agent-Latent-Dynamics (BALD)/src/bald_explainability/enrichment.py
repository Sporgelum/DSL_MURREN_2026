from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def load_gmt(gmt_paths: Sequence[Path]) -> Dict[str, Set[str]]:
    """Load GMT files as {term -> gene set}."""
    out: Dict[str, Set[str]] = {}
    for path in gmt_paths:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                term = parts[0].strip()
                genes = {g.strip() for g in parts[2:] if g.strip()}
                if term and genes:
                    out[term] = genes
    return out


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def enrich_gene_lists(
    ranked_genes: pd.DataFrame,
    gene_sets: Dict[str, Set[str]],
    background_genes: Sequence[str],
    group_col: str = "group",
    gene_col: str = "gene",
    top_n_per_group: int = 200,
    min_overlap: int = 3,
) -> pd.DataFrame:
    """Run hypergeometric enrichment for each group in ranked genes."""
    universe = set(background_genes)
    M = len(universe)
    if M == 0:
        return pd.DataFrame()

    rows: List[dict] = []
    for group in sorted(ranked_genes[group_col].astype(str).unique()):
        grp_df = ranked_genes[ranked_genes[group_col].astype(str) == group]
        grp_df = grp_df.sort_values("rank").head(top_n_per_group)
        query = set(grp_df[gene_col].astype(str).tolist()) & universe
        N = len(query)
        if N == 0:
            continue

        for term, gs in gene_sets.items():
            gs_u = gs & universe
            K = len(gs_u)
            if K == 0:
                continue
            overlap_genes = sorted(query & gs_u)
            k = len(overlap_genes)
            if k < min_overlap:
                continue

            pval = float(hypergeom.sf(k - 1, M, K, N))
            rows.append(
                {
                    "group": group,
                    "term": term,
                    "query_size": N,
                    "gene_set_size": K,
                    "overlap": k,
                    "p_value": pval,
                    "overlap_genes": ";".join(overlap_genes[:100]),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["fdr_bh"] = _bh_fdr(df["p_value"].values.astype(float))
    return df.sort_values(["group", "fdr_bh", "p_value", "overlap"], ascending=[True, True, True, False])
