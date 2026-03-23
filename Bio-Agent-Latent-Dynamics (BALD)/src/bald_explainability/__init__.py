"""BALD explainability toolkit.

Minimal, publication-oriented utilities to:
- score gene importance for trajectory predictions,
- aggregate rankings across cohorts,
- measure stability across folds and random seeds.
"""

from .attribution import gradient_x_input_attribution, integrated_gradients_attribution, shap_attribution
from .enrichment import enrich_gene_lists, load_gmt
from .exporter import export_ranked_genes_csv, ranked_genes_dataframe
from .pipeline import ExplainabilityEngine, ExplainabilityResult
from .real_data import RealAttributionConfig, run_model_attributions
from .stability import jaccard_at_k, spearman_rank_correlation

__all__ = [
    "gradient_x_input_attribution",
    "integrated_gradients_attribution",
    "shap_attribution",
    "load_gmt",
    "enrich_gene_lists",
    "ranked_genes_dataframe",
    "export_ranked_genes_csv",
    "ExplainabilityEngine",
    "ExplainabilityResult",
    "RealAttributionConfig",
    "run_model_attributions",
    "jaccard_at_k",
    "spearman_rank_correlation",
]
