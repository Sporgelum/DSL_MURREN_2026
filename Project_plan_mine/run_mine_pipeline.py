"""
MINE-Enhanced BTM Pipeline — Main Entry Point.

Uses the same real data (logCPM + metadata) from project_plan/counts_and_metadata/
but with six improvements from the MINE paper (Belghazi et al., ICML 2018).

Improvements over run_real_data.py:
  1. EMA bias correction for MINE gradients (Paper §3.2)
  2. Adaptive gradient clipping (Paper §8.1.1)
  3. Dimension-wise MI maximization (per-module anti-collapse)
  4. Pairwise MI minimization (Total Correlation → disentanglement)
  5. MINE-based nonlinear module extraction (post-training)
  6. Deeper statistics network with noise injection (Paper §8.1.5)

Usage:
    cd Project_plan_mine
    python run_mine_pipeline.py
"""

import sys
import time
import importlib.util
from pathlib import Path

PROJECT_MINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_MINE_ROOT.parent / "Project_plan"

# ── Helper: load a module from an explicit file path (avoids namespace collisions) ──
def _import_from(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Imports from project_plan_mine (this folder) ──
sys.path.insert(0, str(PROJECT_MINE_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import PipelineConfig
from model.mine_estimator import build_mine_components
from model.losses import (
    reconstruction_loss, kl_divergence, compute_kl_weight, total_loss,
)
from utils.adaptive_clip import two_pass_backward, compute_grad_norm
from extraction.mine_extraction import mine_extraction, get_latent_activations

# ── Imports from project_plan (loaded by filepath to avoid package name collision) ──
_cvae_mod = _import_from(PROJECT_ROOT / "model" / "cvae.py", "pp_cvae")
ConditionalVAE = _cvae_mod.ConditionalVAE

# Register so that interpret_latent.py's `from model.cvae import ConditionalVAE` resolves
sys.modules["model.cvae"] = _cvae_mod

_interp_mod = _import_from(PROJECT_ROOT / "extraction" / "interpret_latent.py", "pp_interp")
extract_gene_loadings = _interp_mod.extract_gene_loadings
top_genes_per_dimension = _interp_mod.top_genes_per_dimension
export_loadings = _interp_mod.export_loadings
compare_methods = _interp_mod.compare_methods

_utils_mod = _import_from(PROJECT_ROOT / "utils" / "utils.py", "pp_utils")
set_seed = _utils_mod.set_seed
count_parameters = _utils_mod.count_parameters
save_history = _utils_mod.save_history
plot_training_history = _utils_mod.plot_training_history


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (identical to run_real_data.py — reads from project_plan)
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_data():
    """Load logCPM + metadata from project_plan/counts_and_metadata/."""
    print("\n[1/7] Loading real data...")

    data_dir = PROJECT_ROOT / "counts_and_metadata"

    expr = pd.read_csv(
        data_dir / "logCPM_matrix_filtered_samples.csv",
        sep="\t", index_col=0,
    )
    print(f"  logCPM: {expr.shape[0]} genes × {expr.shape[1]} samples")

    meta = pd.read_csv(
        data_dir / "metadata_with_sample_annotations.csv",
        sep="\t",
    )
    meta = meta[meta["Run"].isin(expr.columns)].copy()
    meta = meta.drop_duplicates(subset="Run").set_index("Run")

    common_samples = sorted(set(expr.columns) & set(meta.index))
    expr = expr[common_samples]
    meta = meta.loc[common_samples]
    print(f"  Matched: {len(common_samples)} samples")

    X = expr.T
    gene_names = list(expr.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))

    condition_cols = ["BioProject", "SampleStyle", "SampleTissue"]
    one_hot_parts = []
    for col in condition_cols:
        le = LabelEncoder()
        vals = meta[col].astype(str).values
        int_enc = le.fit_transform(vals)
        n_cls = len(le.classes_)
        oh = np.zeros((len(int_enc), n_cls), dtype=np.float32)
        oh[np.arange(len(int_enc)), int_enc] = 1.0
        one_hot_parts.append(oh)
        print(f"  {col}: {n_cls} categories")

    conditions = np.hstack(one_hot_parts)
    condition_dim = conditions.shape[1]
    print(f"  Condition dim: {condition_dim}")

    return X_scaled, conditions, gene_names, condition_dim


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATALOADERS
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

    print(f"\n[2/7] DataLoaders: train={n_train}, val={n_val}")
    return train_loader, val_loader, full_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL + MINE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(n_genes, condition_dim, latent_dim=128):
    """Build cVAE + all three MINE components."""
    cfg = PipelineConfig()
    cfg.model.input_dim = n_genes
    cfg.model.condition_dim = condition_dim
    cfg.model.latent_dim = latent_dim

    model = ConditionalVAE.from_config(cfg.model, condition_dim)
    global_mine, dimwise_mine, pairwise_mine = build_mine_components(
        cfg.mine, n_genes, latent_dim
    )

    print(f"\n[3/7] Model + MINE components built")
    print(f"  cVAE:            {count_parameters(model):>12,} params")
    print(f"  Global MINE:     {count_parameters(global_mine):>12,} params")
    print(f"  Dim-wise MINE:   {count_parameters(dimwise_mine):>12,} params")
    print(f"  Pairwise MINE:   {count_parameters(pairwise_mine):>12,} params")
    print(f"  Architecture:    {cfg.model.encoder_hidden_dims} → z{latent_dim}")
    print(f"  MI mode:         {'dimension-wise' if cfg.mine.use_dimwise else 'global'}")
    print(f"  TC penalty:      λ_TC={cfg.mine.tc_weight}, {cfg.mine.tc_n_pairs} pairs")
    print(f"  Adaptive clip:   {cfg.mine.use_adaptive_clip}")
    print(f"  EMA α:           {cfg.mine.ema_alpha}")

    return model, global_mine, dimwise_mine, pairwise_mine, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING — Enhanced with all 6 improvements
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model, global_mine, dimwise_mine, pairwise_mine,
    train_loader, val_loader, cfg,
    epochs=100, device="auto",
):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"\n[4/7] Training on {device} for {epochs} epochs...")

    model.to(device)
    global_mine.to(device)
    dimwise_mine.to(device)
    pairwise_mine.to(device)

    # Separate optimizers: lower LR for MINE networks (more stable)
    opt_vae = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    mi_params = (
        list(global_mine.parameters())
        + list(dimwise_mine.parameters())
        + list(pairwise_mine.parameters())
    )
    opt_mi = torch.optim.Adam(mi_params, lr=cfg.training.mi_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vae, T_max=epochs)

    kl_anneal = cfg.training.kl_anneal_epochs
    mi_weight = cfg.mine.mi_weight
    tc_weight = cfg.mine.tc_weight
    use_dimwise = cfg.mine.use_dimwise
    use_adaptive_clip = cfg.mine.use_adaptive_clip
    patience = cfg.training.early_stopping_patience
    best_val = float("inf")
    patience_ctr = 0

    # Output directory
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    history = {k: [] for k in [
        "train_recon", "train_kl", "train_mi", "train_tc", "train_total",
        "val_total", "kl_weight", "mi_grad_scale",
    ]}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        global_mine.train()
        dimwise_mine.train()
        pairwise_mine.train()

        kl_w = compute_kl_weight(epoch, kl_anneal, cfg.training.kl_weight)

        ep_recon, ep_kl, ep_mi, ep_tc, ep_total = 0, 0, 0, 0, 0
        ep_scale = 0
        nb = 0

        for x, c in train_loader:
            x, c = x.to(device), c.to(device)

            # ─── Step 1: Update MI estimator networks ───
            opt_mi.zero_grad()
            with torch.no_grad():
                _, _, _, z_det = model(x, c)

            if use_dimwise:
                mi_loss_for_est = dimwise_mine(x, z_det)
            else:
                mi_loss_for_est = global_mine(x, z_det)

            tc_loss_for_est = pairwise_mine(z_det)
            est_loss = mi_loss_for_est + tc_weight * tc_loss_for_est
            est_loss.backward()
            torch.nn.utils.clip_grad_norm_(mi_params, 1.0)
            opt_mi.step()

            # ─── Step 2: Update cVAE ───
            opt_vae.zero_grad()
            x_rec, mu, logvar, z = model(x, c)

            # Compute losses
            recon = reconstruction_loss(x_rec, x)
            kl = kl_divergence(mu, logvar)
            vae_loss = recon + kl_w * kl

            if use_dimwise:
                mi_loss = dimwise_mine(x, z)
            else:
                mi_loss = global_mine(x, z)

            tc_loss = pairwise_mine(z)

            if use_adaptive_clip:
                # Two-pass backward with adaptive clipping (Paper §8.1.1)
                vae_norm, info_norm, scale = two_pass_backward(
                    model, vae_loss, mi_loss, mi_weight,
                    tc_loss, tc_weight,
                    max_grad_norm=cfg.training.gradient_clip,
                )
                ep_scale += scale
            else:
                # Standard combined backward
                loss = vae_loss + mi_weight * mi_loss + tc_weight * tc_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.gradient_clip
                )
                scale = 1.0
                ep_scale += scale

            opt_vae.step()

            with torch.no_grad():
                total = vae_loss + mi_weight * mi_loss + tc_weight * tc_loss
                ep_recon += recon.item()
                ep_kl += kl.item()
                ep_mi += mi_loss.item()
                ep_tc += tc_loss.item()
                ep_total += total.item()
            nb += 1

        scheduler.step()

        # ─── Validation ───
        model.eval()
        v_total, vnb = 0, 0
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.to(device), c.to(device)
                x_rec, mu, logvar, z = model(x, c)
                v_total += (
                    reconstruction_loss(x_rec, x) + kl_divergence(mu, logvar)
                ).item()
                vnb += 1

        # Record history
        history["train_recon"].append(ep_recon / nb)
        history["train_kl"].append(ep_kl / nb)
        history["train_mi"].append(ep_mi / nb)
        history["train_tc"].append(ep_tc / nb)
        history["train_total"].append(ep_total / nb)
        history["val_total"].append(v_total / vnb)
        history["kl_weight"].append(kl_w)
        history["mi_grad_scale"].append(ep_scale / nb)

        # Early stopping
        is_best = (v_total / vnb) < best_val
        if is_best:
            best_val = v_total / vnb
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "global_mine_state": global_mine.state_dict(),
                "dimwise_mine_state": dimwise_mine.state_dict(),
                "pairwise_mine_state": pairwise_mine.state_dict(),
            }, str(ckpt_dir / "best_mine_checkpoint.pt"))
        else:
            patience_ctr += 1

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(
                f"  Ep {epoch:03d} | recon={ep_recon/nb:.4f} kl={ep_kl/nb:.4f} "
                f"mi={ep_mi/nb:.4f} tc={ep_tc/nb:.4f} | val={v_total/vnb:.4f} "
                f"| β={kl_w:.2f} scale={ep_scale/nb:.3f} "
                f"| pat={patience_ctr}/{patience} | {elapsed:.0f}s"
                f"{'  ★' if is_best else ''}"
            )

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Reload best
    state = torch.load(
        str(ckpt_dir / "best_mine_checkpoint.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(state["model_state"])
    print(f"  Best model: epoch {state['epoch']} (val={best_val:.4f})")

    return history, model


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLASSICAL EXTRACTION (same 3 methods as before)
# ═══════════════════════════════════════════════════════════════════════════════

def classical_extraction(model, gene_names, full_loader, device="cpu"):
    print(f"\n[5/7] Classical extraction (decoder, encoder, empirical)...")
    model.to(device)
    model.eval()

    loadings = extract_gene_loadings(model, gene_names, dataloader=full_loader, device=device)

    for method, df in loadings.items():
        max_abs = df.abs().max().max()
        n_active = (df.abs().max() > 0.05).sum()
        print(f"  {method:10s}: max|loading|={max_abs:.4f}, dims>0.05: {n_active}/{df.shape[1]}")

    return loadings


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MINE-BASED NONLINEAR EXTRACTION (Improvement 5)
# ═══════════════════════════════════════════════════════════════════════════════

def mine_based_extraction(model, full_loader, gene_names, cfg, device="cpu"):
    print(f"\n[6/7] MINE-based nonlinear extraction (Improvement 5)...")
    print(f"  This estimates I(x_g; z_d) for top genes per dimension")
    print(f"  Using {cfg.extraction.mine_extraction_epochs} MINE epochs per pair")

    mi_loadings = mine_extraction(
        model=model,
        dataloader=full_loader,
        gene_names=gene_names,
        device=device,
        hidden_dim=cfg.extraction.mine_extraction_hidden,
        n_epochs=cfg.extraction.mine_extraction_epochs,
        lr=cfg.extraction.mine_extraction_lr,
        top_n_genes_per_dim=200,
        verbose=True,
    )

    max_mi = mi_loadings.max().max()
    n_active = (mi_loadings.max() > 0.01).sum()
    print(f"  Max MI: {max_mi:.4f} nats")
    print(f"  Active dimensions: {n_active}/{mi_loadings.shape[1]}")

    return mi_loadings


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMPARE & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def compare_and_export(loadings, mi_loadings, output_dir):
    print(f"\n[7/7] Exporting results to {output_dir}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Export classical loadings
    export_loadings(loadings, output_dir=str(out / "interpretation"))

    # Export MINE loadings
    mi_loadings.to_csv(out / "interpretation" / "gene_loadings_mine.csv")

    # Top genes from MINE method
    mine_top_dir = out / "interpretation" / "mine"
    mine_top_dir.mkdir(parents=True, exist_ok=True)

    all_top_rows = []
    for dim in mi_loadings.columns:
        dim_vals = mi_loadings[dim].sort_values(ascending=False)
        top30 = dim_vals.head(30)
        top_df = pd.DataFrame({
            "gene": top30.index,
            "mi_estimate": top30.values,
            "dimension": dim,
        })
        top_df.to_csv(mine_top_dir / f"{dim}_top_genes.csv", index=False)
        all_top_rows.append(top_df)

    if all_top_rows:
        pd.concat(all_top_rows, ignore_index=True).to_csv(
            out / "interpretation" / "top_genes_per_dim_mine.csv", index=False
        )

    # Method comparison: now 4 methods
    all_loadings = dict(loadings)
    all_loadings["mine"] = mi_loadings

    for top_n in [20, 50]:
        comp = compare_methods(all_loadings, top_n=top_n)
        comp.to_csv(out / f"method_comparison_4way_top{top_n}.csv", index=False)

    # Summary stats
    print(f"\n  === Module Quality Summary ===")
    print(f"  {'Method':<12} {'Max Loading':<14} {'Dims>0.05':<12} {'Dims>0.1':<10}")
    for method, df in all_loadings.items():
        max_abs = df.abs().max().max()
        n05 = (df.abs().max() > 0.05).sum()
        n10 = (df.abs().max() > 0.1).sum()
        print(f"  {method:<12} {max_abs:<14.4f} {n05:<12} {n10:<10}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MINE-Enhanced BTM Pipeline")
    print("  6 improvements from Belghazi et al. (ICML 2018)")
    print("=" * 70)

    set_seed(42)

    # 1. Load data (from project_plan/counts_and_metadata/)
    X, conditions, gene_names, condition_dim = load_real_data()

    # 2. DataLoaders
    train_loader, val_loader, full_loader = make_dataloaders(X, conditions, batch_size=64)

    # 3. Build model + MINE components
    LATENT_DIM = 128
    model, global_mine, dimwise_mine, pairwise_mine, cfg = build_model(
        len(gene_names), condition_dim, latent_dim=LATENT_DIM
    )

    # 4. Train with all 6 improvements
    history, model = train_model(
        model, global_mine, dimwise_mine, pairwise_mine,
        train_loader, val_loader, cfg,
        epochs=cfg.training.epochs,
    )

    # Save history
    save_history(history, str(PROJECT_ROOT / "logs" / "mine_training_history.json"))
    try:
        plot_training_history(history, save_path=str(PROJECT_ROOT / "logs" / "mine_training_curves.png"))
    except Exception:
        pass

    # 5. Classical extraction (3 methods)
    device = "cpu"
    loadings = classical_extraction(model, gene_names, full_loader, device=device)

    # 6. MINE extraction (4th method)
    mi_loadings = mine_based_extraction(model, full_loader, gene_names, cfg, device=device)

    # 7. Compare & export
    output_dir = PROJECT_ROOT / "results" / "mine_enhanced"
    compare_and_export(loadings, mi_loadings, str(output_dir))

    print("\n" + "=" * 70)
    print("  DONE — Results in results/mine_enhanced/")
    print("=" * 70)
    print(f"""
  Key improvements applied:
    ✓ EMA bias correction for MINE gradients (Paper §3.2)
    ✓ Adaptive gradient clipping (Paper §8.1.1)
    ✓ Dimension-wise MI maximization (per-module anti-collapse)
    ✓ Pairwise MI minimization (Total Correlation penalty)
    ✓ MINE-based nonlinear extraction (4th method)
    ✓ Deeper statistics network with noise injection

  Output files:
    results/mine_enhanced/interpretation/
      gene_loadings_decoder.csv
      gene_loadings_encoder.csv
      gene_loadings_empirical.csv
      gene_loadings_mine.csv          ← NEW: nonlinear MI estimates
      top_genes_per_dim_mine.csv      ← NEW: top genes by MI
    results/mine_enhanced/
      method_comparison_4way_top20.csv ← NEW: 4-way comparison
      method_comparison_4way_top50.csv
    checkpoints/
      best_mine_checkpoint.pt
    logs/
      mine_training_history.json
      mine_training_curves.png
""")


if __name__ == "__main__":
    main()
