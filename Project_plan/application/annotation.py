"""
GSEA annotation — cross-reference discovered modules with biological databases.

Maps extracted BTM gene sets against MSigDB / GSEA databases to assign
functional labels (e.g., "Module 5 = Early Interferon Response").

Protocol Section 4 — Annotation step.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_reference_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """
    Load a reference .gmt file (e.g., from MSigDB).

    Returns:
        pathways: dict {pathway_name: [gene_list]}
    """
    pathways = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = parts[2:]
    return pathways


def compute_overlap(
    module_genes: List[str],
    pathway_genes: List[str],
) -> Tuple[int, float, List[str]]:
    """
    Compute overlap between a module and a reference pathway.

    Returns:
        n_overlap:   number of shared genes
        jaccard:     Jaccard similarity
        shared:      list of shared gene names
    """
    set_m = set(module_genes)
    set_p = set(pathway_genes)
    shared = set_m & set_p
    union = set_m | set_p
    jaccard = len(shared) / len(union) if union else 0.0
    return len(shared), jaccard, sorted(shared)


def annotate_modules(
    btm_modules: Dict[int, pd.DataFrame],
    reference_pathways: Dict[str, List[str]],
    min_overlap: int = 3,
    top_k: int = 5,
) -> Dict[int, pd.DataFrame]:
    """
    For each module, find the top-k most overlapping reference pathways.

    Args:
        btm_modules:         module_index -> DataFrame with 'gene' column
        reference_pathways:  pathway_name -> gene list
        min_overlap:         minimum genes to consider a hit
        top_k:               return top K annotations per module

    Returns:
        annotations: module_index -> DataFrame with columns
                     ['pathway', 'overlap', 'jaccard', 'shared_genes']
    """
    annotations = {}

    for j, module_df in btm_modules.items():
        module_genes = module_df["gene"].tolist()
        results = []

        for pw_name, pw_genes in reference_pathways.items():
            n_overlap, jaccard, shared = compute_overlap(module_genes, pw_genes)
            if n_overlap >= min_overlap:
                results.append({
                    "pathway": pw_name,
                    "overlap": n_overlap,
                    "jaccard": jaccard,
                    "shared_genes": ";".join(shared),
                })

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("jaccard", ascending=False).head(top_k)
            annotations[j] = df.reset_index(drop=True)

    return annotations


def save_annotations(
    annotations: Dict[int, pd.DataFrame],
    output_dir: str,
) -> str:
    """Save all module annotations to a single CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for j, df in annotations.items():
        for _, row in df.iterrows():
            rows.append({"module": j, **row.to_dict()})

    combined = pd.DataFrame(rows)
    fpath = out / "module_annotations.csv"
    combined.to_csv(fpath, index=False)
    print(f"Saved annotations for {len(annotations)} modules to {fpath}")
    return str(fpath)
