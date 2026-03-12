"""
Utility functions for the BTM discovery pipeline.

Includes:
  - Training history visualization
  - Latent space visualization (UMAP/t-SNE)
  - Module heatmap plotting
  - Reproducibility helpers
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device string to a torch.device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def save_history(history: Dict[str, list], output_path: str):
    """Save training history to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    out.write_text(json.dumps(serializable, indent=2))
    print(f"Saved training history to {out}")


def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None):
    """Plot training and validation loss curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Reconstruction loss
    axes[0].plot(history["train_recon"], label="Train")
    axes[0].plot(history["val_recon"], label="Val")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # KL divergence
    axes[1].plot(history["train_kl"], label="Train KL")
    axes[1].plot(history["kl_weight"], label="KL Weight (β)", linestyle="--")
    axes[1].set_title("KL Divergence")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Total loss
    axes[2].plot(history["train_total"], label="Train Total")
    axes[2].plot(history["val_total"], label="Val Total")
    axes[2].set_title("Total Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to {save_path}")
    plt.show()


def plot_latent_space(
    latent_codes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "umap",
    save_path: Optional[str] = None,
):
    """
    Visualize the latent space in 2D using UMAP or t-SNE.

    Args:
        latent_codes: (n_samples, latent_dim) array
        labels:       optional categorical labels for coloring
        method:       'umap' or 'tsne'
        save_path:    optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            print("umap-learn not installed — falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    embedding = reducer.fit_transform(latent_codes)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            mask = labels == lab
            plt.scatter(embedding[mask, 0], embedding[mask, 1], label=lab, s=10, alpha=0.6)
        plt.legend(markerscale=3)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.6)

    plt.title(f"Latent Space ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved latent plot to {save_path}")
    plt.show()


def plot_module_heatmap(
    activity_df,
    save_path: Optional[str] = None,
    top_n_modules: int = 30,
):
    """Plot a heatmap of module activity scores across samples."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping plot")
        return

    # Use top N most variable modules
    variances = activity_df.var()
    top_modules = variances.nlargest(top_n_modules).index
    subset = activity_df[top_modules]

    plt.figure(figsize=(12, 8))
    sns.heatmap(subset.T, cmap="RdBu_r", center=0, xticklabels=False)
    plt.title(f"Top {top_n_modules} Most Variable Modules")
    plt.ylabel("Module")
    plt.xlabel("Sample")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
