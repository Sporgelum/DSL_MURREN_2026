"""
Latent Space Interpretation — Which genes are compressed into each dimension?

The latent matrix Z has shape (n_samples, latent_dim):
  - Rows    = samples (patients/timepoints)
  - Columns = compressed features (latent dimensions / "modules")

Each column is a LEARNED COMBINATION of original genes. This script extracts
exactly which genes contribute to each latent dimension using three methods:

  Method 1 — Decoder Weights (analytical):
      Multiply through all decoder linear layers: W_eff = W_L * W_{L-1} * ... * W_1
      Column j of W_eff tells you how latent dim j maps to each gene.
      ⚠ Approximation: ignores ReLU nonlinearities between layers.

  Method 2 — Encoder Weights (analytical):
      Multiply through encoder layers + fc_mu: W_enc = W_mu * W_{L} * ... * W_1
      Row j of W_enc tells you how each gene feeds into latent dim j.
      ⚠ Same linear approximation caveat.

  Method 3 — Empirical Correlations (data-driven, most accurate):
      Pass all training data through the encoder to get Z.
      For each (gene_i, latent_dim_j), compute Pearson correlation.
      This captures the ACTUAL nonlinear relationship the network learned.
      ✓ No approximation — uses the real activations.

All three methods produce a (n_genes × latent_dim) matrix of "gene loadings"
that can be ranked, thresholded, and exported per dimension.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from model.cvae import ConditionalVAE


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1: Decoder Effective Weights (Z → Genes)
# ═══════════════════════════════════════════════════════════════════════════════

def decoder_effective_weights(model: ConditionalVAE) -> np.ndarray:
    """
    Chain-multiply all decoder Linear layers to approximate Z → Gene mapping.

    Returns:
        W_dec: shape (n_genes, latent_dim + condition_dim)
               Column j = how latent dim j contributes to each gene
    """
    linear_layers = [
        m for m in model.decoder.network.modules()
        if isinstance(m, nn.Linear)
    ]
    W = None
    for layer in linear_layers:
        w = layer.weight.detach().cpu().numpy()
        W = w if W is None else w @ W
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2: Encoder Effective Weights (Genes → Z)
# ═══════════════════════════════════════════════════════════════════════════════

def encoder_effective_weights(model: ConditionalVAE) -> np.ndarray:
    """
    Chain-multiply all encoder Linear layers + fc_mu to approximate Gene → Z.

    Returns:
        W_enc: shape (latent_dim, input_dim + condition_dim)
               Row j = how each gene feeds into latent dim j
    """
    linear_layers = [
        m for m in model.encoder.network.modules()
        if isinstance(m, nn.Linear)
    ]
    W = None
    for layer in linear_layers:
        w = layer.weight.detach().cpu().numpy()
        W = w if W is None else w @ W

    # Final projection to mu
    w_mu = model.encoder.fc_mu.weight.detach().cpu().numpy()
    W = w_mu if W is None else w_mu @ W
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 3: Empirical Correlations (data-driven, most accurate)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def empirical_gene_latent_correlations(
    model: ConditionalVAE,
    dataloader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlation between each gene and each latent dimension
    using actual data passed through the trained encoder.

    This is the most accurate method because it captures the full nonlinear
    mapping the network learned, not a linear approximation.

    Returns:
        correlations: shape (n_genes, latent_dim) — Pearson r values
        latent_all:   shape (n_samples, latent_dim) — all Z values (for reuse)
    """
    model.eval()
    model.to(device)

    all_x = []
    all_z = []

    for x_batch, c_batch in dataloader:
        x_batch = x_batch.to(device)
        c_batch = c_batch.to(device)
        mu, _ = model.encoder(x_batch, c_batch)
        all_x.append(x_batch.cpu().numpy())
        all_z.append(mu.cpu().numpy())

    X = np.vstack(all_x)  # (n_samples, n_genes)
    Z = np.vstack(all_z)  # (n_samples, latent_dim)

    n_genes = X.shape[1]
    latent_dim = Z.shape[1]

    # Standardize for Pearson correlation
    X_std = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    Z_std = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)

    # Correlation matrix: (n_genes, latent_dim)
    correlations = (X_std.T @ Z_std) / X.shape[0]

    return correlations, Z


# ═══════════════════════════════════════════════════════════════════════════════
# Unified gene loading extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_gene_loadings(
    model: ConditionalVAE,
    gene_names: List[str],
    dataloader=None,
    device: str = "cpu",
) -> Dict[str, pd.DataFrame]:
    """
    Extract gene loadings for every latent dimension using all three methods.

    Returns:
        loadings: dict with keys 'decoder', 'encoder', 'empirical' (if dataloader given)
                  Each value is a DataFrame of shape (n_genes, latent_dim)
                  with gene names as index and 'Dim_000', 'Dim_001', ... as columns.
    """
    latent_dim = model.latent_dim
    dim_names = [f"Dim_{j:03d}" for j in range(latent_dim)]
    loadings = {}

    # Method 1: Decoder weights
    W_dec = decoder_effective_weights(model)
    W_dec_latent = W_dec[:, :latent_dim]  # drop condition columns
    loadings["decoder"] = pd.DataFrame(
        W_dec_latent, index=gene_names, columns=dim_names
    )

    # Method 2: Encoder weights
    W_enc = encoder_effective_weights(model)
    W_enc_genes = W_enc[:, :len(gene_names)]  # drop condition columns
    # Transpose so shape is (n_genes, latent_dim) — same orientation as decoder
    loadings["encoder"] = pd.DataFrame(
        W_enc_genes.T, index=gene_names, columns=dim_names
    )

    # Method 3: Empirical correlations (if data available)
    if dataloader is not None:
        corr, _ = empirical_gene_latent_correlations(model, dataloader, device)
        loadings["empirical"] = pd.DataFrame(
            corr, index=gene_names, columns=dim_names
        )

    return loadings


def top_genes_per_dimension(
    loading_df: pd.DataFrame,
    top_n: int = 20,
    by: str = "absolute",
) -> Dict[str, pd.DataFrame]:
    """
    For each latent dimension, return the top contributing genes.

    Args:
        loading_df: (n_genes, latent_dim) DataFrame from extract_gene_loadings
        top_n:      number of top genes per dimension
        by:         'absolute' (rank by |loading|) or 'signed' (keep sign, rank by magnitude)

    Returns:
        per_dim: dict mapping "Dim_000" -> DataFrame with ['gene', 'loading', 'rank']
    """
    per_dim = {}
    for col in loading_df.columns:
        vals = loading_df[col].copy()
        if by == "absolute":
            ranked = vals.abs().sort_values(ascending=False)
        else:
            ranked = vals.reindex(vals.abs().sort_values(ascending=False).index)

        top = ranked.head(top_n)
        df = pd.DataFrame({
            "gene": top.index,
            "loading": loading_df[col].loc[top.index].values,
            "abs_loading": top.values if by == "absolute" else vals.abs().loc[top.index].values,
            "rank": range(1, len(top) + 1),
        })
        per_dim[col] = df

    return per_dim


# ═══════════════════════════════════════════════════════════════════════════════
# Export and reporting
# ═══════════════════════════════════════════════════════════════════════════════

def export_loadings(
    loadings: Dict[str, pd.DataFrame],
    output_dir: str = "results/interpretation",
):
    """Save all loading matrices and per-dimension gene lists to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for method_name, loading_df in loadings.items():
        # Full matrix
        fpath = out / f"gene_loadings_{method_name}.csv"
        loading_df.to_csv(fpath)
        print(f"  Saved {method_name} loadings: {fpath} ({loading_df.shape})")

        # Per-dimension top genes
        per_dim = top_genes_per_dimension(loading_df, top_n=30)
        dim_dir = out / method_name
        dim_dir.mkdir(exist_ok=True)

        all_tops = []
        for dim_name, df in per_dim.items():
            df.to_csv(dim_dir / f"{dim_name}_top_genes.csv", index=False)
            df_with_dim = df.copy()
            df_with_dim.insert(0, "dimension", dim_name)
            all_tops.append(df_with_dim)

        # Combined summary
        combined = pd.concat(all_tops, ignore_index=True)
        combined.to_csv(out / f"top_genes_per_dim_{method_name}.csv", index=False)
        print(f"  Saved per-dim top genes: {out / f'top_genes_per_dim_{method_name}.csv'}")


def compare_methods(
    loadings: Dict[str, pd.DataFrame],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    For each dimension, compare the top gene sets across methods
    to see how consistent they are (Jaccard overlap).
    """
    methods = list(loadings.keys())
    if len(methods) < 2:
        return pd.DataFrame()

    top_sets = {}
    for method in methods:
        per_dim = top_genes_per_dimension(loadings[method], top_n=top_n)
        top_sets[method] = {
            dim: set(df["gene"].tolist()) for dim, df in per_dim.items()
        }

    rows = []
    dims = list(top_sets[methods[0]].keys())
    for dim in dims:
        row = {"dimension": dim}
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                s1 = top_sets[m1][dim]
                s2 = top_sets[m2][dim]
                jaccard = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
                overlap = len(s1 & s2)
                row[f"overlap_{m1}_vs_{m2}"] = overlap
                row[f"jaccard_{m1}_vs_{m2}"] = round(jaccard, 3)
        rows.append(row)

    return pd.DataFrame(rows)


def validate_against_ground_truth(
    loadings: pd.DataFrame,
    ground_truth_path: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Compare discovered gene loadings against known ground truth modules.

    For each latent dim, check which ground-truth module it best matches.
    """
    gt = pd.read_csv(ground_truth_path)
    gt_modules = gt.groupby("module")["gene"].apply(set).to_dict()

    per_dim = top_genes_per_dimension(loadings, top_n=top_n)

    rows = []
    for dim_name, dim_df in per_dim.items():
        discovered = set(dim_df["gene"].tolist())
        best_module = -1
        best_jaccard = 0
        best_overlap = 0

        for mod_id, mod_genes in gt_modules.items():
            overlap = len(discovered & mod_genes)
            jaccard = overlap / len(discovered | mod_genes) if discovered | mod_genes else 0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_module = mod_id
                best_overlap = overlap

        rows.append({
            "dimension": dim_name,
            "best_gt_module": best_module,
            "overlap": best_overlap,
            "jaccard": round(best_jaccard, 3),
            "gt_module_size": len(gt_modules.get(best_module, set())),
            "discovered_size": len(discovered),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dimension_loadings(
    loading_df: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    top_n: int = 15,
    save_dir: Optional[str] = None,
):
    """Bar plot of top gene loadings for selected dimensions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    if dimensions is None:
        # Pick dimensions with highest variance (most informative)
        var = loading_df.var()
        dimensions = var.nlargest(min(6, len(var))).index.tolist()

    n_dims = len(dimensions)
    fig, axes = plt.subplots(2, (n_dims + 1) // 2, figsize=(6 * ((n_dims + 1) // 2), 10))
    axes = axes.flatten() if n_dims > 1 else [axes]

    for idx, dim in enumerate(dimensions):
        if idx >= len(axes):
            break
        ax = axes[idx]
        vals = loading_df[dim]
        top = vals.abs().nlargest(top_n)
        signed_vals = vals.loc[top.index]

        colors = ["#d32f2f" if v > 0 else "#1976d2" for v in signed_vals]
        ax.barh(range(top_n), signed_vals.values[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top.index[::-1], fontsize=7)
        ax.set_title(f"{dim}", fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)

    # Hide unused subplots
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Gene Loadings per Latent Dimension (red=positive, blue=negative)", y=1.01)
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_dir}/dimension_loadings.png", dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {save_dir}/dimension_loadings.png")
    plt.show()


def plot_loading_heatmap(
    loading_df: pd.DataFrame,
    top_n_genes_per_dim: int = 10,
    save_path: Optional[str] = None,
):
    """Heatmap showing the top genes across all dimensions at once."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping")
        return

    # Collect top genes across all dimensions
    top_genes = set()
    for col in loading_df.columns:
        top = loading_df[col].abs().nlargest(top_n_genes_per_dim).index
        top_genes.update(top)

    subset = loading_df.loc[sorted(top_genes)]

    plt.figure(figsize=(max(12, len(loading_df.columns) * 0.5), max(8, len(subset) * 0.2)))
    sns.heatmap(
        subset, cmap="RdBu_r", center=0,
        xticklabels=True, yticklabels=True,
        cbar_kws={"label": "Gene Loading"},
    )
    plt.title("Gene × Latent Dimension Loading Matrix")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Gene")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved heatmap: {save_path}")
    plt.show()
