"""
Decoder Weight Extraction for Blood Transcription Module (BTM) discovery.

After training, the decoder's final linear layer maps latent dimensions (modules)
to genes. We extract, rank, and threshold these weights to define gene modules.

Protocol Section 4 — Results Extraction.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple

from model.cvae import ConditionalVAE


def extract_decoder_weights(model: ConditionalVAE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the weight matrix W from the decoder's final linear layer.

    The final layer maps hidden_dim -> output_dim (n_genes).
    W has shape (n_genes, last_hidden_dim). We need to trace the full
    effective mapping from latent_dim -> n_genes through all decoder layers.

    For simplicity and interpretability, we extract the last Linear layer's
    weights, which capture the direct gene-level contribution.

    Returns:
        weights: (n_genes, last_hidden_dim) weight matrix
        biases:  (n_genes,) bias vector
    """
    # Get the last Linear layer in the decoder
    last_linear = None
    for module in reversed(list(model.decoder.network.modules())):
        if isinstance(module, torch.nn.Linear):
            last_linear = module
            break

    if last_linear is None:
        raise RuntimeError("No Linear layer found in decoder")

    weights = last_linear.weight.detach().cpu().numpy()  # (n_genes, hidden_dim)
    biases = last_linear.bias.detach().cpu().numpy()      # (n_genes,)
    return weights, biases


def compute_effective_weights(model: ConditionalVAE) -> np.ndarray:
    """
    Compute the effective weight matrix from latent space to gene space
    by multiplying through all decoder linear layers (ignoring nonlinearities).

    This gives an approximate linear mapping: gene_j = sum_k W_eff[j, k] * z_k

    Returns:
        W_eff: (n_genes, latent_dim + condition_dim) effective weight matrix
    """
    linear_layers = [
        m for m in model.decoder.network.modules()
        if isinstance(m, torch.nn.Linear)
    ]

    W = None
    for layer in linear_layers:
        w = layer.weight.detach().cpu().numpy()
        if W is None:
            W = w
        else:
            W = w @ W  # chain multiply

    return W  # shape: (n_genes, latent_dim + condition_dim)


def rank_genes_per_module(
    W_eff: np.ndarray,
    gene_names: List[str],
    latent_dim: int,
) -> Dict[int, pd.DataFrame]:
    """
    For each latent dimension (module), rank genes by absolute contribution.

    Only uses the first `latent_dim` columns (ignoring condition dimensions).

    Returns:
        modules: dict mapping module_index -> DataFrame with columns
                 ['gene', 'weight', 'abs_weight', 'zscore']
    """
    # Use only latent dimensions (exclude condition columns)
    W_latent = W_eff[:, :latent_dim]

    modules = {}
    for j in range(latent_dim):
        col = W_latent[:, j]
        abs_col = np.abs(col)
        mean_w = abs_col.mean()
        std_w = abs_col.std()
        zscores = (abs_col - mean_w) / (std_w + 1e-8)

        df = pd.DataFrame({
            "gene": gene_names,
            "weight": col,
            "abs_weight": abs_col,
            "zscore": zscores,
        })
        df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
        modules[j] = df

    return modules


def select_module_genes(
    module_df: pd.DataFrame,
    zscore_threshold: float = 2.5,
    top_n: Optional[int] = None,
    min_size: int = 10,
    max_size: int = 500,
) -> pd.DataFrame:
    """
    Select genes belonging to a module using Z-score or top-N.

    Selection logic:
      - If top_n is set: take the top N genes.
      - Otherwise: take genes with Z-score > threshold.
      - Clamp to [min_size, max_size].
    """
    if top_n is not None:
        selected = module_df.head(top_n)
    else:
        selected = module_df[module_df["zscore"] > zscore_threshold]

    # Enforce size constraints
    if len(selected) < min_size:
        selected = module_df.head(min_size)
    elif len(selected) > max_size:
        selected = selected.head(max_size)

    return selected.reset_index(drop=True)


def extract_all_modules(
    model: ConditionalVAE,
    gene_names: List[str],
    extraction_cfg,
) -> Dict[int, pd.DataFrame]:
    """
    End-to-end module extraction from a trained model.

    Returns:
        btm_modules: dict mapping module_index -> DataFrame of selected genes
    """
    W_eff = compute_effective_weights(model)
    ranked = rank_genes_per_module(W_eff, gene_names, model.latent_dim)

    btm_modules = {}
    for j, module_df in ranked.items():
        selected = select_module_genes(
            module_df,
            zscore_threshold=extraction_cfg.zscore_threshold,
            top_n=extraction_cfg.top_n_genes,
            min_size=extraction_cfg.min_module_size,
            max_size=extraction_cfg.max_module_size,
        )
        if len(selected) >= extraction_cfg.min_module_size:
            btm_modules[j] = selected

    print(f"Extracted {len(btm_modules)} modules "
          f"(from {model.latent_dim} latent dimensions)")
    return btm_modules
