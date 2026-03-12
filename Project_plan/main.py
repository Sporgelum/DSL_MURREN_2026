"""
Main orchestrator for the BTM Discovery Pipeline.

Executes the full protocol:
  1. Load and preprocess data
  2. Build the MI-Regularized cVAE
  3. Train the model
  4. Extract decoder weights → BTM gene modules
  5. Export modules as .gmt + annotate with GSEA databases
  6. (Optional) Project new data for digital inference

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PipelineConfig
from data.data_loader import prepare_data
from model.cvae import ConditionalVAE
from model.mi_regularizer import build_mi_estimator
from training.trainer import Trainer
from extraction.weight_extractor import extract_all_modules
from application.gmt_export import modules_to_gmt
from application.annotation import load_reference_gmt, annotate_modules, save_annotations
from utils.utils import (
    set_seed,
    save_history,
    plot_training_history,
    count_parameters,
)


def main():
    # ── Configuration ──
    cfg = PipelineConfig()
    set_seed(cfg.data.random_seed)
    print("=" * 60)
    print("BTM Discovery Pipeline — MI-Regularized cVAE")
    print("=" * 60)

    # ── Step 1: Data Loading ──
    print("\n[1/5] Loading and preprocessing data...")
    train_loader, val_loader, test_loader, gene_names, condition_dim = prepare_data(cfg)
    cfg.model.input_dim = len(gene_names)
    cfg.model.condition_dim = condition_dim
    print(f"  Genes: {len(gene_names)}, Condition dim: {condition_dim}")
    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # ── Step 2: Build Model ──
    print("\n[2/5] Building MI-Regularized cVAE...")
    model = ConditionalVAE.from_config(cfg.model, condition_dim)
    mi_estimator = build_mi_estimator(cfg.mi, cfg.model.input_dim, cfg.model.latent_dim)
    print(f"  cVAE parameters: {count_parameters(model):,}")
    print(f"  MI estimator parameters: {count_parameters(mi_estimator):,}")
    print(f"  Latent dimensions (modules): {cfg.model.latent_dim}")

    # ── Step 3: Train ──
    print("\n[3/5] Training...")
    trainer = Trainer(model, mi_estimator, train_loader, val_loader, cfg)
    history = trainer.fit()

    # Save and plot history
    save_history(history, f"{cfg.training.log_dir}/training_history.json")
    plot_training_history(history, save_path=f"{cfg.training.log_dir}/training_curves.png")

    # Load best checkpoint for extraction
    trainer.load_checkpoint(f"{cfg.training.checkpoint_dir}/best_checkpoint.pt")
    print("  Loaded best checkpoint for extraction.")

    # ── Step 4: Extract Modules ──
    print("\n[4/5] Extracting Blood Transcription Modules...")
    btm_modules = extract_all_modules(model, gene_names, cfg.extraction)

    for j, mod_df in sorted(btm_modules.items()):
        top3 = ", ".join(mod_df["gene"].head(3).tolist())
        print(f"  Module {j:03d}: {len(mod_df)} genes (top: {top3})")

    # ── Step 5: Export & Annotate ──
    print("\n[5/5] Exporting .gmt and annotating modules...")
    gmt_path = modules_to_gmt(btm_modules, cfg.application.gmt_output_path)
    print(f"  GMT saved to: {gmt_path}")

    # Annotation (if reference GMT is available)
    reference_gmt_path = Path("data/reference/msigdb_immunologic.gmt")
    if reference_gmt_path.exists():
        ref_pathways = load_reference_gmt(str(reference_gmt_path))
        annotations = annotate_modules(btm_modules, ref_pathways)
        save_annotations(annotations, "results/annotations")
        print(f"  Annotated {len(annotations)} modules against {len(ref_pathways)} pathways")
    else:
        print(f"  Skipping annotation — reference GMT not found at {reference_gmt_path}")
        print(f"  Place an MSigDB .gmt file there for automatic annotation.")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Modules: {len(btm_modules)}")
    print(f"  GMT file: {gmt_path}")
    print(f"  Checkpoints: {cfg.training.checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
