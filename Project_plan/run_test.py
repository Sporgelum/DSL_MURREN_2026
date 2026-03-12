"""
Step-by-step test script — run each pipeline stage independently.

This script uses smaller synthetic data (500 genes instead of 20,000)
so you can test the full pipeline quickly on CPU.

Usage:
    cd Project_plan
    python run_test.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd


def step0_install_check():
    """Verify all dependencies are available."""
    print("=" * 60)
    print("STEP 0: Checking dependencies")
    print("=" * 60)
    deps = {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
    }
    all_ok = True
    for module, pip_name in deps.items():
        try:
            __import__(module)
            print(f"  [OK] {pip_name}")
        except ImportError:
            print(f"  [MISSING] {pip_name}  -->  pip install {pip_name}")
            all_ok = False

    # Optional
    for module, pip_name in [("seaborn", "seaborn"), ("umap", "umap-learn")]:
        try:
            __import__(module)
            print(f"  [OK] {pip_name} (optional)")
        except ImportError:
            print(f"  [SKIP] {pip_name} (optional, install for extra plots)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    return all_ok


def step1_generate_data():
    """Generate synthetic test data."""
    print("=" * 60)
    print("STEP 1: Generating synthetic RNA-seq data")
    print("=" * 60)
    from generate_synthetic_data import generate_synthetic_data

    df, true_modules = generate_synthetic_data(
        n_samples=600,
        n_genes=500,
        n_true_modules=10,
        genes_per_module=30,
        n_studies=3,
        output_dir="data/raw",
        seed=42,
    )
    print(f"\n  Shape: {df.shape}")
    print(f"  Vaccine types: {df['vaccine_type'].unique().tolist()}")
    print(f"  Time points: {df['time_point'].unique().tolist()}")
    print(f"  Ground truth modules saved to data/ground_truth_modules.csv")
    print()
    return df, true_modules


def step2_load_and_preprocess():
    """Load, preprocess, and build DataLoaders."""
    print("=" * 60)
    print("STEP 2: Loading and preprocessing data")
    print("=" * 60)
    from config import PipelineConfig
    from data.data_loader import prepare_data

    cfg = PipelineConfig()
    # Override for smaller synthetic test data
    cfg.training.batch_size = 64

    train_loader, val_loader, test_loader, gene_names, condition_dim = prepare_data(cfg)

    print(f"  Genes: {len(gene_names)}")
    print(f"  Condition dim (one-hot): {condition_dim}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Peek at one batch
    x_batch, c_batch = next(iter(train_loader))
    print(f"  Batch shapes: expression={x_batch.shape}, condition={c_batch.shape}")
    print(f"  Expression range: [{x_batch.min():.2f}, {x_batch.max():.2f}]")
    print()
    return train_loader, val_loader, test_loader, gene_names, condition_dim


def step3_build_model(n_genes, condition_dim):
    """Construct the cVAE and MI estimator."""
    print("=" * 60)
    print("STEP 3: Building model")
    print("=" * 60)
    from config import PipelineConfig
    from model.cvae import ConditionalVAE
    from model.mi_regularizer import build_mi_estimator
    from utils.utils import count_parameters

    cfg = PipelineConfig()
    # Adjust architecture for our smaller test dataset
    cfg.model.input_dim = n_genes
    cfg.model.condition_dim = condition_dim
    cfg.model.latent_dim = 16  # Fewer modules for fast testing
    cfg.model.encoder_hidden_dims = [256, 128]
    cfg.model.decoder_hidden_dims = [128, 256]

    model = ConditionalVAE.from_config(cfg.model, condition_dim)
    mi_est = build_mi_estimator(cfg.mi, n_genes, cfg.model.latent_dim)

    print(f"  cVAE parameters: {count_parameters(model):,}")
    print(f"  MI estimator parameters: {count_parameters(mi_est):,}")
    print(f"  Latent dim (modules): {cfg.model.latent_dim}")

    # Test forward pass
    device = torch.device("cpu")
    x_test = torch.randn(4, n_genes)
    c_test = torch.randn(4, condition_dim)
    x_recon, mu, logvar, z = model(x_test, c_test)
    print(f"\n  Forward pass test:")
    print(f"    Input:  {x_test.shape}")
    print(f"    Recon:  {x_recon.shape}")
    print(f"    Latent: {z.shape}")
    print(f"    Mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    print()
    return model, mi_est, cfg


def step4_train(model, mi_est, train_loader, val_loader, cfg):
    """Train the model (short run for testing)."""
    print("=" * 60)
    print("STEP 4: Training (short test run)")
    print("=" * 60)
    from training.trainer import Trainer

    # Quick test: only a few epochs
    cfg.training.epochs = 20
    cfg.training.early_stopping_patience = 10
    cfg.training.kl_anneal_epochs = 5
    cfg.training.batch_size = 64
    cfg.training.device = "auto"

    trainer = Trainer(model, mi_est, train_loader, val_loader, cfg)
    history = trainer.fit()

    print(f"\n  Final train loss: {history['train_total'][-1]:.4f}")
    print(f"  Final val loss: {history['val_total'][-1]:.4f}")
    print(f"  Epochs run: {len(history['train_total'])}")

    # Save history
    from utils.utils import save_history
    save_history(history, "logs/training_history.json")

    # Try plotting (will skip if no display)
    try:
        from utils.utils import plot_training_history
        plot_training_history(history, save_path="logs/training_curves.png")
        print("  Training curves saved to logs/training_curves.png")
    except Exception as e:
        print(f"  (Plotting skipped: {e})")

    print()
    return history, trainer


def step5_extract_modules(model, gene_names, cfg):
    """Extract BTM modules from the trained decoder."""
    print("=" * 60)
    print("STEP 5: Extracting modules")
    print("=" * 60)
    from extraction.weight_extractor import extract_all_modules

    cfg.extraction.zscore_threshold = 2.0  # Slightly less strict for small data
    btm_modules = extract_all_modules(model, gene_names, cfg.extraction)

    print(f"\n  Discovered {len(btm_modules)} modules:")
    for j, mod_df in sorted(btm_modules.items()):
        top_genes = ", ".join(mod_df["gene"].head(5).tolist())
        print(f"    Module {j:02d}: {len(mod_df)} genes — [{top_genes}]")

    # Compare with ground truth
    gt_path = Path("data/ground_truth_modules.csv")
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        print(f"\n  Ground truth: {gt['module'].nunique()} modules")
        print("  (Compare discovered modules with ground truth for validation)")

    print()
    return btm_modules


def step6_export_and_apply(btm_modules, cfg):
    """Export GMT and test projection."""
    print("=" * 60)
    print("STEP 6: Exporting GMT & testing projection")
    print("=" * 60)
    from application.gmt_export import modules_to_gmt, load_gmt

    gmt_path = modules_to_gmt(btm_modules, cfg.application.gmt_output_path)
    print(f"  GMT written to: {gmt_path}")

    # Verify GMT is readable
    loaded = load_gmt(gmt_path)
    print(f"  Verified: {len(loaded)} modules loaded back from GMT")
    for name, genes in list(loaded.items())[:3]:
        print(f"    {name}: {len(genes)} genes")

    print()
    return gmt_path


def main():
    print("\n" + "#" * 60)
    print("#  BTM Discovery Pipeline — Test Run")
    print("#" * 60 + "\n")

    # Step 0: Check deps
    if not step0_install_check():
        print("Install missing dependencies first: pip install -r requirements.txt")
        return

    # Step 1: Generate data
    df, true_modules = step1_generate_data()

    # Step 2: Load & preprocess
    train_loader, val_loader, test_loader, gene_names, condition_dim = step2_load_and_preprocess()

    # Step 3: Build model
    model, mi_est, cfg = step3_build_model(len(gene_names), condition_dim)

    # Step 4: Train (short run)
    history, trainer = step4_train(model, mi_est, train_loader, val_loader, cfg)

    # Step 5: Extract modules
    btm_modules = step5_extract_modules(model, gene_names, cfg)

    # Step 6: Export & apply
    gmt_path = step6_export_and_apply(btm_modules, cfg)

    print("=" * 60)
    print("ALL STEPS COMPLETE!")
    print("=" * 60)
    print(f"""
What was produced:
  - data/raw/Study_*.csv      : Synthetic expression data
  - data/ground_truth_modules.csv : Known module structure
  - checkpoints/               : Model weights
  - logs/training_history.json : Training metrics
  - logs/training_curves.png   : Loss plots
  - {gmt_path}   : Discovered modules (GMT)

Next steps:
  1. Inspect the training curves in logs/
  2. Compare discovered modules vs ground truth
  3. Tune config.py parameters (latent_dim, thresholds, etc.)
  4. Replace synthetic data with real RNA-seq CSVs in data/raw/
  5. Add a reference GMT from MSigDB to data/reference/ for annotation
""")


if __name__ == "__main__":
    main()
