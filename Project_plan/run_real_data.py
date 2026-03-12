"""
Run the full BTM pipeline on REAL RNA-seq data.

Data: logCPM counts (32,763 genes × 613 samples) + metadata
      Tab-separated files in counts_and_metadata/

Conditions used for the cVAE:
  - BioProject   (study / batch correction — 26 studies)
  - SampleStyle   (control vs disease groups — 8 types)
  - SampleTissue  (blood vs pbmc — 3 types)

Pipeline:
  1. Load and align logCPM + metadata (match by Run ID)
  2. Build cVAE with architecture scaled for 32K genes
  3. Train with MI regularization
  4. Extract modules via all 3 methods (decoder, encoder, empirical)
  5. Compare methods: how many modules, gene overlap, consistency
  6. Export gene loadings per dimension

Usage:
    cd Project_plan
    python run_real_data.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import PipelineConfig
from model.cvae import ConditionalVAE
from model.mi_regularizer import build_mi_estimator
from model.losses import reconstruction_loss, kl_divergence, compute_kl_weight, total_loss
from extraction.interpret_latent import (
    extract_gene_loadings,
    top_genes_per_dimension,
    export_loadings,
    compare_methods,
    plot_dimension_loadings,
    plot_loading_heatmap,
)
from utils.utils import set_seed, count_parameters, save_history, plot_training_history


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_data():
    """Load logCPM + metadata, align by Run ID, one-hot encode conditions."""
    print("\n[1/6] Loading real data...")

    # Load expression (genes as rows, samples as columns → transpose)
    expr = pd.read_csv(
        "counts_and_metadata/logCPM_matrix_filtered_samples.csv",
        sep="\t", index_col=0,
    )
    print(f"  logCPM raw: {expr.shape[0]} genes × {expr.shape[1]} samples")

    # Load metadata
    meta = pd.read_csv(
        "counts_and_metadata/metadata_with_sample_annotations.csv",
        sep="\t",
    )
    print(f"  Metadata: {meta.shape[0]} rows × {meta.shape[1]} columns")

    # Keep only metadata rows whose Run ID is in the logCPM columns
    meta = meta[meta["Run"].isin(expr.columns)].copy()
    meta = meta.drop_duplicates(subset="Run").set_index("Run")
    print(f"  Matched samples: {len(meta)}")

    # Align: keep only overlapping samples, in same order
    common_samples = sorted(set(expr.columns) & set(meta.index))
    expr = expr[common_samples]
    meta = meta.loc[common_samples]
    print(f"  Aligned: {expr.shape[1]} samples")

    # Transpose: now (samples × genes)
    X = expr.T
    gene_names = list(expr.index)
    print(f"  Expression matrix: {X.shape} (samples × genes)")

    # Data is already logCPM → just Z-score normalize across samples
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))
    print(f"  Normalized: mean≈{X_scaled.mean():.4f}, std≈{X_scaled.std():.4f}")

    # One-hot encode conditions: BioProject, SampleStyle, SampleTissue
    condition_cols = ["BioProject", "SampleStyle", "SampleTissue"]
    encoders = {}
    one_hot_parts = []
    for col in condition_cols:
        le = LabelEncoder()
        vals = meta[col].astype(str).values
        int_enc = le.fit_transform(vals)
        n_cls = len(le.classes_)
        oh = np.zeros((len(int_enc), n_cls), dtype=np.float32)
        oh[np.arange(len(int_enc)), int_enc] = 1.0
        one_hot_parts.append(oh)
        encoders[col] = le
        print(f"  {col}: {n_cls} categories")

    conditions = np.hstack(one_hot_parts)
    condition_dim = conditions.shape[1]
    print(f"  Total condition dim: {condition_dim}")

    return X_scaled, conditions, gene_names, condition_dim, meta, scaler, encoders


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_dataloaders(X, conditions, batch_size=64, seed=42):
    x_tensor = torch.FloatTensor(X)
    c_tensor = torch.FloatTensor(conditions)
    dataset = TensorDataset(x_tensor, c_tensor)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n[2/6] DataLoaders ready")
    print(f"  Train: {n_train} samples ({len(train_loader)} batches)")
    print(f"  Val:   {n_val} samples ({len(val_loader)} batches)")
    return train_loader, val_loader, full_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(n_genes, condition_dim, latent_dim=128):
    """Build cVAE scaled for real gene counts."""
    cfg = PipelineConfig()
    cfg.model.input_dim = n_genes
    cfg.model.condition_dim = condition_dim
    cfg.model.latent_dim = latent_dim
    # Architecture for ~32K genes
    cfg.model.encoder_hidden_dims = [2048, 512, 256]
    cfg.model.decoder_hidden_dims = [256, 512, 2048]
    cfg.model.dropout = 0.2

    model = ConditionalVAE.from_config(cfg.model, condition_dim)
    mi_est = build_mi_estimator(cfg.mi, n_genes, latent_dim)

    print(f"\n[3/6] Model built")
    print(f"  Input dim:  {n_genes} genes")
    print(f"  Latent dim: {latent_dim} (= number of modules)")
    print(f"  Condition:  {condition_dim}")
    print(f"  cVAE params:  {count_parameters(model):,}")
    print(f"  MI est params: {count_parameters(mi_est):,}")
    print(f"  Architecture: {cfg.model.encoder_hidden_dims} → z{latent_dim} → {cfg.model.decoder_hidden_dims}")

    return model, mi_est, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING (inline for control over logging)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, mi_est, train_loader, val_loader, cfg, epochs=50, device="auto"):
    from model.mi_regularizer import compute_mi_loss

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"\n[4/6] Training on {device} for {epochs} epochs...")

    model.to(device)
    mi_est.to(device)

    opt_vae = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    opt_mi = torch.optim.Adam(mi_est.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vae, T_max=epochs)

    kl_anneal = 30
    mi_weight = cfg.mi.mi_weight
    patience = 15
    best_val = float("inf")
    patience_ctr = 0

    history = {k: [] for k in [
        "train_recon", "train_kl", "train_mi", "train_total",
        "val_recon", "val_kl", "val_total", "kl_weight",
    ]}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        mi_est.train()
        kl_w = compute_kl_weight(epoch, kl_anneal, cfg.training.kl_weight)

        ep_recon, ep_kl, ep_mi, ep_total, nb = 0, 0, 0, 0, 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)

            # Update MI estimator
            opt_mi.zero_grad()
            with torch.no_grad():
                _, _, _, z_det = model(x, c)
            mi_loss_val = compute_mi_loss(mi_est, x, z_det, cfg.mi.mi_estimator)
            mi_loss_val.backward()
            opt_mi.step()

            # Update cVAE
            opt_vae.zero_grad()
            x_rec, mu, logvar, z = model(x, c)
            mi_for_vae = compute_mi_loss(mi_est, x, z, cfg.mi.mi_estimator)
            loss = total_loss(x_rec, x, mu, logvar, mi_for_vae, kl_w, mi_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_vae.step()

            with torch.no_grad():
                ep_recon += reconstruction_loss(x_rec, x).item()
                ep_kl += kl_divergence(mu, logvar).item()
                ep_mi += mi_for_vae.item()
                ep_total += loss.item()
            nb += 1

        scheduler.step()

        # Validation
        model.eval()
        mi_est.eval()
        v_recon, v_kl, v_total, vnb = 0, 0, 0, 0
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.to(device), c.to(device)
                x_rec, mu, logvar, z = model(x, c)
                v_recon += reconstruction_loss(x_rec, x).item()
                v_kl += kl_divergence(mu, logvar).item()
                v_total += (reconstruction_loss(x_rec, x) + kl_divergence(mu, logvar)).item()
                vnb += 1

        # Record
        history["train_recon"].append(ep_recon / nb)
        history["train_kl"].append(ep_kl / nb)
        history["train_mi"].append(ep_mi / nb)
        history["train_total"].append(ep_total / nb)
        history["val_recon"].append(v_recon / vnb)
        history["val_kl"].append(v_kl / vnb)
        history["val_total"].append(v_total / vnb)
        history["kl_weight"].append(kl_w)

        # Early stopping
        is_best = (v_total / vnb) < best_val
        if is_best:
            best_val = v_total / vnb
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "mi_estimator_state": mi_est.state_dict(),
            }, "checkpoints/best_real_checkpoint.pt")
        else:
            patience_ctr += 1

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"  Ep {epoch:03d} | recon={ep_recon/nb:.4f} kl={ep_kl/nb:.4f} "
                  f"mi={ep_mi/nb:.4f} | val={v_total/vnb:.4f} | β={kl_w:.2f} "
                  f"| pat={patience_ctr}/{patience} | {elapsed:.0f}s"
                  f"{' ★' if is_best else ''}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Reload best
    state = torch.load("checkpoints/best_real_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    print(f"  Best model from epoch {state['epoch']} (val={best_val:.4f})")

    return history, model


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTERPRETATION — All three methods
# ═══════════════════════════════════════════════════════════════════════════════

def interpret(model, gene_names, full_loader, device="cpu"):
    print(f"\n[5/6] Extracting gene loadings (3 methods)...")
    model.to(device)
    model.eval()

    loadings = extract_gene_loadings(model, gene_names, dataloader=full_loader, device=device)

    for method, df in loadings.items():
        per_dim = top_genes_per_dimension(df, top_n=20)
        n_active_dims = sum(1 for col in df.columns if df[col].abs().max() > 0.01)
        print(f"\n  === {method.upper()} ===")
        print(f"  Shape: {df.shape}")
        print(f"  Active dimensions (max|loading| > 0.01): {n_active_dims}/{df.shape[1]}")
        # Show top 3 dims by variance (most informative)
        var = df.var().sort_values(ascending=False)
        for dim in var.index[:3]:
            top5 = top_genes_per_dimension(df[[dim]], top_n=5)[dim]
            genes = top5["gene"].tolist()
            loads = [f"{v:.4f}" for v in top5["loading"].tolist()]
            print(f"    {dim} (var={var[dim]:.6f}): {list(zip(genes, loads))}")

    return loadings


# ═══════════════════════════════════════════════════════════════════════════════
# 6. METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_and_export(loadings, output_dir="results/real_data"):
    print(f"\n[6/6] Comparing methods and exporting...")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Export full loading matrices + per-dim top genes
    export_loadings(loadings, output_dir=str(out / "interpretation"))

    # Method comparison
    for top_n in [20, 50]:
        comp = compare_methods(loadings, top_n=top_n)
        comp.to_csv(out / f"method_comparison_top{top_n}.csv", index=False)
        print(f"\n  Method comparison (top-{top_n} genes per dim):")

        # Summarize
        for col in comp.columns:
            if col.startswith("jaccard_"):
                mean_j = comp[col].mean()
                max_j = comp[col].max()
                n_good = (comp[col] > 0.1).sum()
                print(f"    {col}: mean J={mean_j:.3f}, max J={max_j:.3f}, "
                      f"dims with J>0.1: {n_good}/{len(comp)}")

    # Module count summary: how many "meaningful" modules per method
    print(f"\n  Module counts (dims where top gene has |loading| > threshold):")
    for method, df in loadings.items():
        max_abs = df.abs().max()
        for thresh_name, thresh in [("strict (>0.3)", 0.3), ("moderate (>0.1)", 0.1), ("loose (>0.05)", 0.05)]:
            n_modules = (max_abs > thresh).sum()
            print(f"    {method:10s} {thresh_name}: {n_modules} modules")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BTM Pipeline — REAL DATA (logCPM, 32K genes × 613 samples)")
    print("=" * 65)

    set_seed(42)

    # 1. Load
    X, conditions, gene_names, condition_dim, meta, scaler, encoders = load_real_data()

    # 2. DataLoaders
    train_loader, val_loader, full_loader = make_dataloaders(X, conditions, batch_size=64)

    # 3. Build model
    LATENT_DIM = 128  # ← number of modules to discover
    model, mi_est, cfg = build_model(len(gene_names), condition_dim, latent_dim=LATENT_DIM)

    # 4. Train
    history, model = train_model(model, mi_est, train_loader, val_loader, cfg, epochs=80)

    # Save & plot history
    save_history(history, "logs/real_training_history.json")
    try:
        plot_training_history(history, save_path="logs/real_training_curves.png")
    except Exception:
        pass

    # 5. Interpret: extract gene loadings from all 3 methods
    device = "cpu"
    loadings = interpret(model, gene_names, full_loader, device=device)

    # 6. Compare methods & export
    out_dir = compare_and_export(loadings)

    # Plots
    try:
        vis = "empirical" if "empirical" in loadings else "decoder"
        plot_dimension_loadings(loadings[vis], top_n=15, save_dir=str(out_dir / "plots"))
        plot_loading_heatmap(loadings[vis], top_n_genes_per_dim=8,
                            save_path=str(out_dir / "plots/loading_heatmap.png"))
    except Exception as e:
        print(f"  Plotting note: {e}")

    print("\n" + "=" * 65)
    print("  DONE — All results in results/real_data/")
    print("=" * 65)
    print(f"""
  Key output files:
    results/real_data/interpretation/
      gene_loadings_decoder.csv     — Decoder weight matrix (genes × dims)
      gene_loadings_encoder.csv     — Encoder weight matrix (genes × dims)
      gene_loadings_empirical.csv   — Correlation matrix  (genes × dims)
      top_genes_per_dim_*.csv       — Top 30 genes per dimension per method
    results/real_data/
      method_comparison_top20.csv   — Cross-method Jaccard overlap
      method_comparison_top50.csv
    logs/
      real_training_history.json
      real_training_curves.png
    checkpoints/
      best_real_checkpoint.pt
""")


if __name__ == "__main__":
    main()
