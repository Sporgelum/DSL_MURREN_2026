from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .attribution import gradient_x_input_attribution, integrated_gradients_attribution, shap_attribution
from .exporter import export_ranked_genes_csv
from .stability import consensus_gene_ranking, mean_pairwise_jaccard, mean_pairwise_spearman


@dataclass
class ExplainabilityResult:
    sample_scores: np.ndarray
    group_consensus_scores: Dict[str, np.ndarray]
    top_gene_indices_per_group: Dict[str, np.ndarray]
    stability: Dict[str, float]

    def to_ranked_gene_csv(
        self,
        gene_names: Sequence[str],
        output_csv: Path,
        top_k: int = 100,
        method: str = "model_attribution",
    ):
        """Export ranked genes per group into CSV."""
        return export_ranked_genes_csv(
            group_consensus_scores=self.group_consensus_scores,
            gene_names=gene_names,
            output_csv=output_csv,
            top_k=top_k,
            method=method,
        )


class ExplainabilityEngine:
    """Minimal explainability pipeline for BALD trajectory models.

    Expected model behavior:
    - input: expression vector at Day 0 or concatenated temporal feature vector
    - output: trajectory target (for example latent delta or Day 7 latent)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        method: str = "integrated_gradients",
        top_k: int = 50,
        target_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shap_background: Optional[torch.Tensor] = None,
        shap_nsamples: int = 128,
    ) -> None:
        if method not in {"integrated_gradients", "gradient_x_input", "shap"}:
            raise ValueError("method must be 'integrated_gradients', 'gradient_x_input', or 'shap'")
        self.model = model
        self.method = method
        self.top_k = top_k
        self.target_selector = target_selector
        self.shap_background = shap_background
        self.shap_nsamples = shap_nsamples

    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "integrated_gradients":
            return integrated_gradients_attribution(
                self.model,
                x,
                steps=32,
                target_selector=self.target_selector,
            )
        if self.method == "shap":
            return shap_attribution(
                self.model,
                x,
                background=self.shap_background,
                nsamples=self.shap_nsamples,
                target_selector=self.target_selector,
            )
        return gradient_x_input_attribution(
            self.model,
            x,
            target_selector=self.target_selector,
        )

    def explain(
        self,
        x: torch.Tensor,
        groups: Sequence[str],
        repeated_score_runs: Optional[Sequence[np.ndarray]] = None,
    ) -> ExplainabilityResult:
        """Run attribution and aggregate results by group.

        Args:
            x: Tensor [n_samples, n_genes].
            groups: Group label per sample, same length as n_samples.
            repeated_score_runs: Optional repeated global score vectors for stability.
        """
        if x.ndim != 2:
            raise ValueError("x must be rank-2 [samples, genes]")
        if len(groups) != x.shape[0]:
            raise ValueError("groups length must equal number of samples")

        attributions = self.attribute(x).detach().cpu().numpy()
        sample_scores = np.abs(attributions)

        group_consensus_scores: Dict[str, np.ndarray] = {}
        top_gene_indices_per_group: Dict[str, np.ndarray] = {}

        groups_np = np.array(groups)
        for group_name in sorted(set(groups)):
            idx = np.where(groups_np == group_name)[0]
            group_scores = sample_scores[idx].mean(axis=0)
            group_consensus_scores[group_name] = group_scores
            k = min(self.top_k, group_scores.shape[0])
            top_gene_indices_per_group[group_name] = np.argsort(-group_scores)[:k]

        if repeated_score_runs:
            stability = {
                "mean_pairwise_jaccard_topk": mean_pairwise_jaccard(repeated_score_runs, k=self.top_k),
                "mean_pairwise_spearman": mean_pairwise_spearman(repeated_score_runs),
            }
        else:
            global_scores = sample_scores.mean(axis=0)
            stability = {
                "mean_pairwise_jaccard_topk": 1.0,
                "mean_pairwise_spearman": 1.0,
                "global_consensus_available": float(global_scores.mean()),
            }

        return ExplainabilityResult(
            sample_scores=sample_scores,
            group_consensus_scores=group_consensus_scores,
            top_gene_indices_per_group=top_gene_indices_per_group,
            stability=stability,
        )

    @staticmethod
    def build_consensus_from_runs(score_runs: Sequence[np.ndarray]) -> np.ndarray:
        return consensus_gene_ranking(score_runs)
