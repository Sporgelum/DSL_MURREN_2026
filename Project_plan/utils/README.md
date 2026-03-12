# Utils Module — Visualization & Helpers

## Goal
Provide shared utility functions for reproducibility, visualization, and diagnostics throughout the BTM discovery pipeline.

## Main Idea
Centralize common operations that multiple pipeline stages need: setting random seeds, plotting training curves, visualizing the latent space, and generating module activity heatmaps. These are kept separate from the core logic to maintain clean module boundaries.

## Functions

| Function | Purpose |
|---|---|
| `set_seed(seed)` | Fix random seeds across Python, NumPy, and PyTorch for reproducibility |
| `get_device(preference)` | Resolve `"auto"` / `"cuda"` / `"cpu"` to a `torch.device` |
| `save_history(history, path)` | Save training metrics dict to JSON |
| `plot_training_history(history)` | 3-panel plot: reconstruction, KL, total loss curves |
| `plot_latent_space(codes, labels)` | 2D UMAP or t-SNE visualization of the latent space |
| `plot_module_heatmap(activity_df)` | Seaborn heatmap of module activity scores across samples |
| `count_parameters(model)` | Count trainable parameters for model diagnostics |

## Inputs
| Input | Description |
|---|---|
| `history` dict | Per-epoch metric lists from training |
| `latent_codes` array | Encoded samples from the latent space |
| `activity_df` DataFrame | Module activity scores from projection |

## Outputs
| Output | Description |
|---|---|
| Saved plots (`.png`) | Training curves, latent space, heatmaps |
| `training_history.json` | Serialized metrics for offline analysis |

## Dependencies
- `matplotlib` — core plotting
- `seaborn` — heatmaps
- `umap-learn` (optional) — UMAP visualization
- `scikit-learn` — t-SNE fallback
