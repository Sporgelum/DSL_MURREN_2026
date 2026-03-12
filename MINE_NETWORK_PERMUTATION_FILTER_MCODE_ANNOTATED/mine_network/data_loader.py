"""
Data loading — expression matrices, metadata, and study discovery.
==================================================================

Responsibilities
----------------
1. **Load the logCPM expression matrix** (genes × samples, tab-separated).
   - Auto-detects TSV/CSV from extension (defaults to tab).
   - Returns a ``pandas.DataFrame`` with gene names as row index, sample
     (Run / SRR) IDs as column names.

2. **Load sample metadata** from a tab-separated table.
   - Must contain at least two columns:
       ``Run``        — matches expression‐matrix column names (SRR IDs).
       ``BioProject`` — study / cohort identifier (PRJ IDs).
   - Additional columns (tissue, condition, etc.) are preserved for
     downstream annotation but not required by the pipeline.

3. **Discover studies** automatically from the ``BioProject`` column.
   - Each unique ``BioProject`` ID becomes one independent study.
   - Only Run IDs present in *both* the expression matrix and the metadata
     are used; extra rows are silently ignored.
   - Studies with fewer than ``min_samples`` samples are dropped (default 3).
   - Returns a list of study dicts, each with the expression sub-matrix
     and gene names, ready for per-study processing.

4. **Z-score expression** — standardise each gene to mean 0, std 1.
   - This is the continuous alternative to KBinsDiscretizer (quantile
     binning) used in the histogram-MI pipeline.
   - After Z-scoring every gene ≈ N(0, 1), so the permutation null
     is approximately gene-pair-agnostic (Section 4c of the user design).

Design notes
------------
- Separator detection: if the file extension ends with ``.tsv`` or ``.txt``
  the separator is ``\\t``; otherwise ``\\t`` is still the default because
  the user's data is tab-separated even with ``.csv`` extension.
- Gene names are always taken from the first column (row index).
- No genes are filtered at this stage — that is the responsibility of
  upstream QC or the pre-screening step.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_expression(counts_path: str) -> pd.DataFrame:
    """
    Load the full logCPM expression matrix.

    Parameters
    ----------
    counts_path : str
        Path to the expression file (genes × samples, tab-separated).
        The first column is treated as the gene name index.

    Returns
    -------
    pd.DataFrame
        Rows = genes, columns = sample Run IDs.
    """
    expr = pd.read_csv(counts_path, sep="\t", index_col=0)
    print(f"[INFO] Loaded expression matrix: {counts_path}")
    print(f"[INFO] Shape: {expr.shape[0]:,} genes × {expr.shape[1]:,} samples")
    return expr


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load sample metadata.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata file (tab-separated).
        Must contain columns ``Run`` (SRR IDs) and ``BioProject`` (PRJ IDs).

    Returns
    -------
    pd.DataFrame
        Full metadata table.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    md = pd.read_csv(metadata_path, sep="\t")
    print(f"[INFO] Loaded metadata: {metadata_path}")
    print(f"[INFO] Metadata shape: {md.shape}")
    for col in ("Run", "BioProject"):
        if col not in md.columns:
            raise ValueError(
                f"Metadata is missing required column '{col}'. "
                f"Available columns: {list(md.columns)}"
            )
    return md


# ═══════════════════════════════════════════════════════════════════════════════
# Study discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_studies(
    expr_full: pd.DataFrame,
    metadata: pd.DataFrame,
    min_samples: int = 3,
) -> list:
    """
    Auto-discover studies from the BioProject column of the metadata.

    Each unique ``BioProject`` value becomes one independent study.
    Only ``Run`` IDs present in the expression matrix are included.
    Studies with fewer than ``min_samples`` samples are skipped.

    Parameters
    ----------
    expr_full : pd.DataFrame
        Full expression matrix (genes × all samples).
    metadata : pd.DataFrame
        Must contain ``Run`` and ``BioProject`` columns.
    min_samples : int
        Minimum number of samples to include a study.

    Returns
    -------
    list[dict]
        Each dict has keys:

        - ``name`` : str — BioProject ID (filesystem-safe).
        - ``expr`` : pd.DataFrame — sub-matrix (genes × study_samples).
        - ``gene_names`` : list[str] — gene name list.
    """
    available_runs = set(expr_full.columns)
    md_matched = metadata[metadata["Run"].isin(available_runs)].copy()

    n_unmatched = len(metadata) - len(md_matched)
    if n_unmatched:
        print(f"[WARN] {n_unmatched} metadata rows without matching "
              f"expression columns (ignored).")

    studies = []
    for bioproj, group in md_matched.groupby("BioProject"):
        runs = group["Run"].tolist()
        if len(runs) < min_samples:
            print(f"[WARN] {bioproj}: {len(runs)} samples < "
                  f"{min_samples} — skipping")
            continue
        sub = expr_full[runs]
        safe_name = str(bioproj).replace(" ", "_").replace("/", "-")
        studies.append({
            "name": safe_name,
            "expr": sub,
            "gene_names": sub.index.tolist(),
        })
        print(f"[INFO] Study: {safe_name} ({len(runs)} samples)")

    print(f"[INFO] Total studies discovered: {len(studies)}")
    return studies


# ═══════════════════════════════════════════════════════════════════════════════
# Z-scoring
# ═══════════════════════════════════════════════════════════════════════════════

def zscore_expression(expr_data: pd.DataFrame) -> np.ndarray:
    """
    Z-score each gene across samples: (x - μ) / σ.

    After this transform every gene's marginal ≈ N(0, 1), which makes the
    permutation null distribution approximately gene-pair-agnostic.  This
    is the continuous-data analogue of quantile binning.

    Parameters
    ----------
    expr_data : pd.DataFrame
        Expression sub-matrix (genes × study_samples), raw logCPM.

    Returns
    -------
    np.ndarray, shape (n_genes, n_samples), dtype float32
        Z-scored expression values.
    """
    X = expr_data.values.astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero for constant genes
    return (X - mu) / std
