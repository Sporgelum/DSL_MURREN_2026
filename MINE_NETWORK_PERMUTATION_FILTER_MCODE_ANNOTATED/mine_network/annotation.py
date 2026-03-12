"""
Biological annotation — gene-set enrichment for MCODE modules.
===============================================================

After MCODE identifies dense modules of co-expressed genes, the next step
is biological interpretation.  This module performs **hypergeometric
enrichment testing** (equivalent to Fisher's exact test) of each module
against user-provided gene-set collections.

Supported input format
-----------------------
Gene-set files in **GMT** (Gene Matrix Transposed) format, the standard
used by MSigDB, KEGG, Reactome, and GO.  Each line:

    <set_name>\\t<description>\\t<gene1>\\t<gene2>\\t...

Common sources:
  - MSigDB Hallmark (H), curated pathways (C2:CP), GO (C5)
  - KEGG pathways
  - Reactome
  - Custom gene-set files (e.g. blood transcription modules from Li et al.)

For non-human organisms (e.g. pig), gene names should be mapped to the
same identifiers used in the expression matrix before enrichment.

Statistical method
-------------------
For each (module, gene_set) combination:

1. Define:
   - K = module size
   - M = gene-set size
   - N = background (universe) size
   - x = |module ∩ gene_set| (overlap)

2. Compute p-value from the hypergeometric distribution:
   p = P(X ≥ x) = 1 − CDF(x − 1; N, M, K)

3. Apply Benjamini–Hochberg FDR correction across all tested combinations.

4. Report enrichments with adjusted p < ``fdr_threshold``.

Output
------
Per-module enrichment table (TSV):
  Module, GeneSet, Description, Overlap, ModuleSize, SetSize, pValue,
  pAdjusted, OverlappingGenes

This enables direct biological interpretation of the MCODE modules as
pathways, functional categories, or known gene signatures.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


# ═══════════════════════════════════════════════════════════════════════════════
# GMT loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_gmt(gmt_path: str) -> dict:
    """
    Load a GMT gene-set file.

    Parameters
    ----------
    gmt_path : str
        Path to the GMT file.

    Returns
    -------
    dict[str, dict]
        Keys are gene-set names.  Values are dicts with:
        - ``"description"`` : str
        - ``"genes"`` : set[str]
    """
    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            desc = parts[1]
            genes = set(parts[2:])
            gene_sets[name] = {"description": desc, "genes": genes}
    print(f"[INFO] Loaded {len(gene_sets)} gene sets from {gmt_path}")
    return gene_sets


def load_multiple_gmt(gmt_paths: list) -> dict:
    """
    Load and merge multiple GMT files.

    If the same gene-set name appears in multiple files, the later one
    overwrites the earlier.

    Parameters
    ----------
    gmt_paths : list[str]
        Paths to GMT files.

    Returns
    -------
    dict[str, dict]
        Merged gene-set collection.
    """
    merged = {}
    for path in gmt_paths:
        merged.update(load_gmt(path))
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Hypergeometric enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def hypergeometric_test(
    module_genes: set,
    geneset_genes: set,
    background_size: int,
) -> tuple:
    """
    One-sided hypergeometric test for over-representation.

    Parameters
    ----------
    module_genes : set[str]
        Genes in the MCODE module.
    geneset_genes : set[str]
        Genes in the reference gene set.
    background_size : int
        Total number of genes in the universe.

    Returns
    -------
    overlap : int
        Number of genes in common.
    p_value : float
        P(X ≥ overlap) under the hypergeometric distribution.
    overlap_genes : list[str]
        Names of the overlapping genes.
    """
    overlap_set = module_genes & geneset_genes
    x = len(overlap_set)
    if x == 0:
        return 0, 1.0, []

    # N = population, M = successes in population, n = drawn
    N = background_size
    M = len(geneset_genes)
    n = len(module_genes)

    # P(X >= x) = 1 - CDF(x - 1)
    p_value = hypergeom.sf(x - 1, N, M, n)
    return x, float(p_value), sorted(overlap_set)


def _bh_correct(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray, shape (m,)
        Raw p-values.

    Returns
    -------
    np.ndarray, shape (m,)
        BH-adjusted p-values.
    """
    m = len(p_values)
    if m == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, m + 1)
    adjusted = sorted_p * m / ranks
    # Enforce monotonicity (from last to first)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    # Restore original order
    result = np.empty(m)
    result[order] = adjusted
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Module annotation driver
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_modules(
    modules: dict,
    gene_sets: dict,
    background_genes: set,
    fdr_threshold: float = 0.05,
    min_overlap: int = 2,
) -> pd.DataFrame:
    """
    Annotate all MCODE modules against a gene-set collection.

    For each (module, gene_set) combination with overlap ≥ ``min_overlap``,
    performs a hypergeometric enrichment test.  P-values are BH-corrected
    across *all* tests jointly.

    Parameters
    ----------
    modules : dict[int, list[str]]
        ``{module_id: [gene_name, ...]}`` from MCODE.
    gene_sets : dict[str, dict]
        From ``load_gmt``.  Each value has keys ``"description"`` and
        ``"genes"`` (set).
    background_genes : set[str]
        The gene universe (typically all genes in the expression matrix).
    fdr_threshold : float
        Maximum adjusted p-value to include in the output.
    min_overlap : int
        Minimum overlap size to test.

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns:
        ``Module``, ``GeneSet``, ``Description``, ``Overlap``,
        ``ModuleSize``, ``SetSize``, ``pValue``, ``pAdjusted``,
        ``OverlappingGenes``.
        Only rows with ``pAdjusted < fdr_threshold`` are included.
        Sorted by ``pAdjusted``.
    """
    bg_size = len(background_genes)
    if bg_size == 0:
        print("[WARN] Empty background gene set — skipping annotation.")
        return pd.DataFrame()

    records = []
    for mid, mod_genes_list in modules.items():
        mod_genes = set(mod_genes_list) & background_genes
        if len(mod_genes) < min_overlap:
            continue

        for gs_name, gs_info in gene_sets.items():
            gs_genes = gs_info["genes"] & background_genes
            if len(gs_genes) == 0:
                continue

            overlap, pval, overlap_genes = hypergeometric_test(
                mod_genes, gs_genes, bg_size,
            )
            if overlap < min_overlap:
                continue

            records.append({
                "Module": f"M{mid}",
                "GeneSet": gs_name,
                "Description": gs_info["description"],
                "Overlap": overlap,
                "ModuleSize": len(mod_genes),
                "SetSize": len(gs_genes),
                "pValue": pval,
                "OverlappingGenes": ";".join(overlap_genes),
            })

    if not records:
        print("[INFO] No enrichments found with the given gene sets.")
        return pd.DataFrame(columns=[
            "Module", "GeneSet", "Description", "Overlap",
            "ModuleSize", "SetSize", "pValue", "pAdjusted",
            "OverlappingGenes",
        ])

    df = pd.DataFrame(records)

    # BH correction across all tests
    df["pAdjusted"] = _bh_correct(df["pValue"].values)

    # Filter and sort
    df = df[df["pAdjusted"] < fdr_threshold].copy()
    df.sort_values("pAdjusted", inplace=True)

    print(f"[INFO] Enrichment: {len(df)} significant annotations "
          f"(FDR < {fdr_threshold})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: save annotation results
# ═══════════════════════════════════════════════════════════════════════════════

def save_annotations(
    annotation_df: pd.DataFrame,
    output_dir: str,
    filename: str = "module_annotations.tsv",
) -> None:
    """
    Save annotation results to a TSV file.

    Also saves a per-module summary (top 5 gene sets per module).

    Parameters
    ----------
    annotation_df : pd.DataFrame
        From ``annotate_modules``.
    output_dir : str
        Output directory.
    filename : str
        Name of the main annotation file.
    """
    if annotation_df.empty:
        print("[INFO] No annotations to save.")
        return

    path = os.path.join(output_dir, filename)
    annotation_df.to_csv(path, sep="\t", index=False)
    print(f"[SAVED] {path}")

    # Per-module summary (top 5)
    summary_rows = []
    for module_id, group in annotation_df.groupby("Module"):
        top5 = group.nsmallest(5, "pAdjusted")
        for _, row in top5.iterrows():
            summary_rows.append({
                "Module": row["Module"],
                "TopGeneSet": row["GeneSet"],
                "Description": row["Description"],
                "Overlap": row["Overlap"],
                "pAdjusted": row["pAdjusted"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "module_annotation_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"[SAVED] {summary_path}")
