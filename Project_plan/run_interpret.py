"""
Interpret Latent Dimensions — Which genes are in each compressed feature?

Loads the trained model and runs all three interpretation methods:
  1. Decoder weights  (analytical — approximate)
  2. Encoder weights  (analytical — approximate)
  3. Empirical correlations (data-driven — exact)

Exports:
  - Full gene × dimension loading matrices (CSV)
  - Top 30 genes per dimension (CSV per method)
  - Method comparison (how consistent are the three approaches?)
  - Ground truth validation (if available)
  - Visualizations (bar plots + heatmap)

Usage:
    cd Project_plan
    python run_interpret.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from config import PipelineConfig
from data.data_loader import prepare_data
from model.cvae import ConditionalVAE
from model.mi_regularizer import build_mi_estimator
from extraction.interpret_latent import (
    extract_gene_loadings,
    top_genes_per_dimension,
    export_loadings,
    compare_methods,
    validate_against_ground_truth,
    plot_dimension_loadings,
    plot_loading_heatmap,
)


def main():
    print("=" * 65)
    print("  Latent Space Interpretation")
    print("  'Which genes are compressed into each dimension?'")
    print("=" * 65)

    # ── Config ──
    cfg = PipelineConfig()
    cfg.training.batch_size = 64

    # ── Load data (need the DataLoader for empirical correlations) ──
    print("\n[1] Loading data...")
    train_loader, val_loader, test_loader, gene_names, condition_dim = prepare_data(cfg)
    print(f"    {len(gene_names)} genes, {condition_dim} condition dims")

    # ── Rebuild model with correct dimensions ──
    cfg.model.input_dim = len(gene_names)
    cfg.model.condition_dim = condition_dim
    cfg.model.latent_dim = 16       # Must match what was used in training
    cfg.model.encoder_hidden_dims = [256, 128]
    cfg.model.decoder_hidden_dims = [128, 256]

    model = ConditionalVAE.from_config(cfg.model, condition_dim)

    # ── Load trained weights ──
    ckpt_path = Path("checkpoints/best_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"    ERROR: No checkpoint at {ckpt_path}")
        print(f"    Run 'python run_test.py' first to train a model.")
        return

    print(f"[2] Loading trained model from {ckpt_path}...")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"    Loaded (trained for {state['epoch']} epochs)")

    # ── Extract gene loadings with all three methods ──
    print("\n[3] Extracting gene loadings (3 methods)...")

    # Combine train + val for more data
    from torch.utils.data import DataLoader, ConcatDataset
    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)

    loadings = extract_gene_loadings(
        model, gene_names,
        dataloader=combined_loader,
        device="cpu",
    )

    for method, df in loadings.items():
        print(f"\n    === {method.upper()} METHOD ===")
        print(f"    Shape: {df.shape}  (genes × latent dimensions)")

        # Show top 5 genes for first 3 dimensions
        per_dim = top_genes_per_dimension(df, top_n=5)
        for dim_name in list(per_dim.keys())[:3]:
            genes = per_dim[dim_name]["gene"].tolist()
            loads = per_dim[dim_name]["loading"].round(3).tolist()
            print(f"      {dim_name}: {list(zip(genes, loads))}")
        print(f"      ... ({len(per_dim)} dimensions total)")

    # ── Export everything ──
    print("\n[4] Exporting to results/interpretation/...")
    export_loadings(loadings, output_dir="results/interpretation")

    # ── Compare methods ──
    print("\n[5] Comparing methods (top-20 gene overlap)...")
    comparison = compare_methods(loadings, top_n=20)
    comparison.to_csv("results/interpretation/method_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # ── Validate against ground truth ──
    gt_path = Path("data/ground_truth_modules.csv")
    if gt_path.exists():
        print("\n[6] Validating against ground truth...")
        for method in loadings:
            validation = validate_against_ground_truth(
                loadings[method], str(gt_path), top_n=30
            )
            validation.to_csv(
                f"results/interpretation/gt_validation_{method}.csv", index=False
            )
            matched = validation[validation["jaccard"] > 0.1]
            print(f"\n    {method.upper()}: {len(matched)}/{len(validation)} dims "
                  f"match a ground truth module (Jaccard > 0.1)")
            if len(matched) > 0:
                for _, row in matched.iterrows():
                    print(f"      {row['dimension']} → GT Module {row['best_gt_module']} "
                          f"(overlap={row['overlap']}, J={row['jaccard']})")
    else:
        print("\n[6] No ground truth file — skipping validation")

    # ── Visualize ──
    print("\n[7] Generating plots...")
    try:
        # Use empirical if available, else decoder
        vis_method = "empirical" if "empirical" in loadings else "decoder"
        plot_dimension_loadings(
            loadings[vis_method],
            top_n=15,
            save_dir="results/interpretation/plots",
        )
        plot_loading_heatmap(
            loadings[vis_method],
            top_n_genes_per_dim=8,
            save_path="results/interpretation/plots/loading_heatmap.png",
        )
    except Exception as e:
        print(f"    Plotting error (non-fatal): {e}")

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  INTERPRETATION COMPLETE")
    print("=" * 65)
    print(f"""
  Output files in results/interpretation/:
    gene_loadings_decoder.csv     — Full (genes × dims) decoder weight matrix
    gene_loadings_encoder.csv     — Full (genes × dims) encoder weight matrix
    gene_loadings_empirical.csv   — Full (genes × dims) correlation matrix
    top_genes_per_dim_*.csv       — Top 30 genes per dimension (each method)
    method_comparison.csv         — Cross-method consistency
    gt_validation_*.csv           — Ground truth matching (if available)
    plots/                        — Bar plots and heatmaps

  How to read the results:
    Open 'gene_loadings_empirical.csv' — each column (Dim_000, Dim_001, ...)
    is a latent dimension. The values are Pearson correlations between that
    gene's expression and that dimension's activation across all samples.

    High positive loading = gene is strongly ACTIVATED when this dimension fires
    High negative loading = gene is strongly SUPPRESSED when this dimension fires
    Near zero             = gene is irrelevant to this dimension
""")


if __name__ == "__main__":
    main()
