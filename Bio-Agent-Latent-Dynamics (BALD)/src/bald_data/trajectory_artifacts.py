from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def _pca_scores(x: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Compute PCA scores using SVD, no sklearn dependency."""
    x0 = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x0, full_matrices=False)
    k = min(n_components, u.shape[1])
    return u[:, :k] * s[:k]


def _top_gene_table(
    expr_samples_by_genes: pd.DataFrame,
    meta: pd.DataFrame,
    top_k: int = 100,
) -> pd.DataFrame:
    """Simple robust gene ranking baseline by day contrast.

    This is an artifact starter for BALD-Explorer before model-based attributions
    are fully trained. It ranks genes by absolute mean shift from Day0.
    """
    out_rows: List[Dict[str, object]] = []

    x = expr_samples_by_genes
    if "day_order" not in meta.columns:
        raise ValueError("metadata must contain day_order")

    if "BioProject" not in meta.columns:
        raise ValueError("metadata must contain BioProject")

    global_day0_idx = meta.index[meta["day_order"] == 0].tolist()
    global_day0 = x.loc[global_day0_idx] if global_day0_idx else None

    for project in sorted(meta["BioProject"].astype(str).unique()):
        p_idx = meta.index[meta["BioProject"].astype(str) == project]
        p_meta = meta.loc[p_idx]
        p_x = x.loc[p_idx]

        day0_idx = p_meta.index[p_meta["day_order"] == 0].tolist()
        ref = p_x.loc[day0_idx] if day0_idx else global_day0
        if ref is None or ref.empty:
            continue

        ref_mean = ref.mean(axis=0)

        for day in sorted(d for d in p_meta["day_order"].unique().tolist() if d != 99):
            day_idx = p_meta.index[p_meta["day_order"] == day].tolist()
            if not day_idx:
                continue
            day_mean = p_x.loc[day_idx].mean(axis=0)
            score = (day_mean - ref_mean).abs().sort_values(ascending=False)
            top = score.head(top_k)

            for rank, (gene, val) in enumerate(top.items(), start=1):
                out_rows.append(
                    {
                        "group": project,
                        "day_order": int(day),
                        "day_label": f"Day{int(day)}",
                        "gene": str(gene),
                        "score": float(val),
                        "rank": rank,
                        "method": "mean_abs_shift_from_day0",
                    }
                )

    return pd.DataFrame(out_rows)


def build_trajectory_artifacts(
    expr_samples_by_genes: pd.DataFrame,
    meta: pd.DataFrame,
    output_dir: Path,
    top_k_genes: int = 100,
) -> Dict[str, Path]:
    """Create starter artifacts for BALD-Explorer.

    Output files:
    - latent_points.csv
    - trajectory_summary.csv
    - top_genes_by_group.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    x = expr_samples_by_genes.values.astype(np.float32)
    x = _zscore_columns(x)
    pcs = _pca_scores(x, n_components=3)

    points = meta.copy()
    points = points.reset_index().rename(columns={"index": "Run"})
    points["pc1"] = pcs[:, 0] if pcs.shape[1] > 0 else 0.0
    points["pc2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0
    points["pc3"] = pcs[:, 2] if pcs.shape[1] > 2 else 0.0

    keep_cols = [
        col
        for col in [
            "Run",
            "BioProject",
            "SampleTimepoint",
            "day_order",
            "day_label",
            "pc1",
            "pc2",
            "pc3",
        ]
        if col in points.columns
    ]
    points = points[keep_cols]

    traj = (
        points.groupby(["BioProject", "day_order", "day_label"], as_index=False)[["pc1", "pc2", "pc3"]]
        .mean()
        .sort_values(["BioProject", "day_order"])
    )

    top_genes = _top_gene_table(expr_samples_by_genes, meta, top_k=top_k_genes)

    points_path = output_dir / "latent_points.csv"
    traj_path = output_dir / "trajectory_summary.csv"
    genes_path = output_dir / "top_genes_by_group.csv"

    points.to_csv(points_path, index=False)
    traj.to_csv(traj_path, index=False)
    top_genes.to_csv(genes_path, index=False)

    return {
        "latent_points": points_path,
        "trajectory_summary": traj_path,
        "top_genes_by_group": genes_path,
    }
