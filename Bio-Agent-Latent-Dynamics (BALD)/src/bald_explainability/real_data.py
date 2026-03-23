from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.bald_explainability.enrichment import enrich_gene_lists, load_gmt
from src.bald_explainability.pipeline import ExplainabilityEngine
from src.bald_models import CheckpointLoadConfig, load_checkpoint_model


@dataclass
class RealAttributionConfig:
    checkpoint_path: Path
    output_dir: Path
    methods: Sequence[str]
    top_k: int = 100
    group_by_day: bool = True
    device: str = "cpu"
    model_hidden_dim: int = 256
    model_output_dim: int = 16
    shap_feature_cap: int = 300
    shap_max_samples: int = 96
    shap_nsamples: int = 128
    feature_list_path: Optional[Path] = None
    gmt_paths: Optional[Sequence[Path]] = None


def _select_features_for_model(
    expr: pd.DataFrame,
    gene_names: Sequence[str],
    feature_list_path: Optional[Path],
) -> Tuple[pd.DataFrame, List[str]]:
    if feature_list_path is None:
        return expr, list(gene_names)

    with feature_list_path.open("r", encoding="utf-8", errors="ignore") as f:
        expected = [ln.strip() for ln in f if ln.strip()]

    keep = [g for g in expected if g in expr.columns]
    if not keep:
        raise ValueError("No overlap between feature-list genes and expression genes")
    return expr[keep].copy(), keep


def _build_group_labels(meta: pd.DataFrame, group_by_day: bool) -> List[str]:
    if group_by_day and "day_label" in meta.columns:
        return [f"{p}__{d}" for p, d in zip(meta["BioProject"].astype(str), meta["day_label"].astype(str))]
    return meta["BioProject"].astype(str).tolist()


def _postprocess_ranked_genes(ranked: pd.DataFrame) -> pd.DataFrame:
    if "group" not in ranked.columns:
        return ranked

    if ranked["group"].astype(str).str.contains("__").any():
        split = ranked["group"].astype(str).str.split("__", n=1, expand=True)
        ranked["BioProject"] = split[0]
        ranked["day_label"] = split[1]
        ranked["day_order"] = ranked["day_label"].str.replace("Day", "", regex=False).replace("Unknown", np.nan)
    return ranked


def _make_publication_tables(
    ranked_ig: Optional[pd.DataFrame],
    ranked_shap: Optional[pd.DataFrame],
    trajectory_summary: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    if ranked_ig is not None and ranked_shap is not None and not ranked_ig.empty and not ranked_shap.empty:
        merged = ranked_ig.merge(
            ranked_shap,
            on=[c for c in ["group", "gene", "BioProject", "day_label", "day_order"] if c in ranked_ig.columns and c in ranked_shap.columns],
            how="outer",
            suffixes=("_ig", "_shap"),
        )
        if "rank_ig" in merged.columns and "rank_shap" in merged.columns:
            merged["consensus_rank_mean"] = merged[["rank_ig", "rank_shap"]].mean(axis=1)
        out["publication_top_genes"] = merged.sort_values([c for c in ["group", "consensus_rank_mean", "rank_ig", "rank_shap"] if c in merged.columns])

    if not trajectory_summary.empty:
        traj = trajectory_summary.copy()
        if {"BioProject", "day_label", "pc1", "pc2"}.issubset(traj.columns):
            out["publication_trajectory_table"] = traj[[c for c in ["BioProject", "day_label", "day_order", "pc1", "pc2", "pc3"] if c in traj.columns]].sort_values([c for c in ["BioProject", "day_order"] if c in traj.columns])

    return out


def run_model_attributions(
    expr_samples_by_genes: pd.DataFrame,
    meta: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    config: RealAttributionConfig,
) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    expr_for_model, selected_genes = _select_features_for_model(
        expr=expr_samples_by_genes,
        gene_names=expr_samples_by_genes.columns.tolist(),
        feature_list_path=config.feature_list_path,
    )

    model = load_checkpoint_model(
        CheckpointLoadConfig(
            checkpoint_path=config.checkpoint_path,
            input_dim=expr_for_model.shape[1],
            output_dim=config.model_output_dim,
            hidden_dim=config.model_hidden_dim,
            device=config.device,
        )
    )

    x_full = torch.as_tensor(expr_for_model.values.astype(np.float32), dtype=torch.float32, device=torch.device(config.device))
    groups = _build_group_labels(meta=meta, group_by_day=config.group_by_day)

    artifact_paths: Dict[str, Path] = {}
    ranked_by_method: Dict[str, pd.DataFrame] = {}

    for method in config.methods:
        if method == "shap":
            variances = np.var(expr_for_model.values.astype(np.float32), axis=0)
            cap = min(config.shap_feature_cap, expr_for_model.shape[1])
            feat_idx = np.argsort(-variances)[:cap]
            sample_cap = min(config.shap_max_samples, x_full.shape[0])
            x_method = x_full[:sample_cap, feat_idx]
            groups_method = groups[:sample_cap]
            genes_method = [selected_genes[i] for i in feat_idx]
        else:
            x_method = x_full
            groups_method = groups
            genes_method = selected_genes

        engine = ExplainabilityEngine(
            model=model,
            method=method,
            top_k=config.top_k,
            shap_nsamples=config.shap_nsamples,
        )

        result = engine.explain(x_method, groups_method)
        out_csv = config.output_dir / f"top_genes_by_group_{method}.csv"
        ranked_df = result.to_ranked_gene_csv(
            gene_names=genes_method,
            output_csv=out_csv,
            top_k=config.top_k,
            method=method,
        )
        ranked_df = _postprocess_ranked_genes(ranked_df)
        ranked_df.to_csv(out_csv, index=False)

        stability_path = config.output_dir / f"stability_{method}.json"
        stability_path.write_text(pd.Series(result.stability).to_json(indent=2), encoding="utf-8")

        artifact_paths[f"top_genes_{method}"] = out_csv
        artifact_paths[f"stability_{method}"] = stability_path
        ranked_by_method[method] = ranked_df

    if config.gmt_paths:
        gene_sets = load_gmt(config.gmt_paths)
        if gene_sets and "integrated_gradients" in ranked_by_method:
            enr_df = enrich_gene_lists(
                ranked_genes=ranked_by_method["integrated_gradients"],
                gene_sets=gene_sets,
                background_genes=selected_genes,
                top_n_per_group=max(config.top_k, 200),
                min_overlap=2,
            )
            if not enr_df.empty:
                enr_path = config.output_dir / "pathway_enrichment_ig.csv"
                enr_df.to_csv(enr_path, index=False)
                artifact_paths["pathway_enrichment_ig"] = enr_path

                pub_path = config.output_dir / "publication_pathway_table.csv"
                enr_df.sort_values(["group", "fdr_bh", "p_value"]).groupby("group", as_index=False).head(10).to_csv(pub_path, index=False)
                artifact_paths["publication_pathway_table"] = pub_path

    pub_tables = _make_publication_tables(
        ranked_ig=ranked_by_method.get("integrated_gradients"),
        ranked_shap=ranked_by_method.get("shap"),
        trajectory_summary=trajectory_summary,
    )
    for name, df in pub_tables.items():
        if df.empty:
            continue
        path = config.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        artifact_paths[name] = path

    return artifact_paths
