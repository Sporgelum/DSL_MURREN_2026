"""
MINE-based Gene Network Inference — Main Pipeline.

Drop-in replacement for generate_net_python_pval.py.
Instead of histogram MI on discretised data, uses neural MI estimation (MINE)
on continuous Z-scored expression.

Pipeline:
  1. Load logCPM expression + metadata
  2. Auto-discover studies by BioProject
  3. Per study:
     a. Z-score expression (no binning)
     b. Pre-screen gene pairs by Pearson |r| (fast, reduces candidates)
     c. Estimate MI for candidate pairs via batched MINE
     d. Build MINE permutation null
     e. Filter edges by empirical p-value
     f. Save per-study results
  4. Build multi-study consensus master network
  5. Detect modules with MCODE
  6. Save master results

Usage:
    cd Project_MINE_network
    python run_mine_network.py

    # Or with a specific Python:
    & "path/to/python.exe" run_mine_network.py
"""

import os
import sys
import gc
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Add this directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PipelineConfig
from mine_estimator import estimate_mi_for_pairs, build_mine_null
from prescreen import prescreen_pairs
from mcode import mcode
from utils import (
    TeeLogger, Timer, format_time,
    load_expression, load_metadata, discover_studies, zscore_expression,
    filter_edges_by_pvalue, build_edgelist, apply_bh_fdr,
    build_master_network,
    save_null_qc, save_study_results, save_master_results, save_report,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit these paths for your environment
# ═══════════════════════════════════════════════════════════════════════════════

# --- Local paths (course data in Project_plan/counts_and_metadata/) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try local course data first, then HPC path
LOCAL_DATA = PROJECT_ROOT / "Project_plan" / "counts_and_metadata"
HPC_DATA = Path("/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts")

if LOCAL_DATA.exists():
    DATA_DIR = LOCAL_DATA
elif HPC_DATA.exists():
    DATA_DIR = HPC_DATA
else:
    DATA_DIR = LOCAL_DATA  # Will fail with a clear error

COUNTS_PATH = str(DATA_DIR / "logCPM_matrix_filtered_samples.csv")
METADATA_PATH = str(DATA_DIR / "metadata_with_sample_annotations.csv")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")


def get_device(cfg):
    """Resolve device string to torch.device."""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = PipelineConfig()
    cfg.counts_path = COUNTS_PATH
    cfg.metadata_path = METADATA_PATH
    cfg.output_dir = OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"mine_network_{ts}.log")
    report_file = os.path.join(OUTPUT_DIR, f"analysis_report_{ts}.txt")
    sys.stdout = TeeLogger(log_file)

    timings = {}
    info = {}
    device = get_device(cfg)

    print("=" * 80)
    print("MINE-BASED GENE NETWORK INFERENCE")
    print("Neural MI estimation replacing histogram MI")
    print("=" * 80)
    print(f"Device           : {device}")
    print(f"MINE hidden_dim  : {cfg.mine.hidden_dim}")
    print(f"MINE epochs      : {cfg.mine.n_epochs}")
    print(f"Pre-screen       : {cfg.prescreen.method} |r| > {cfg.prescreen.threshold}")
    print(f"Null permutations: {cfg.permutation.n_permutations}")
    print(f"P-value threshold: {cfg.permutation.p_value_threshold}")
    print(f"Batch pairs      : {cfg.mine.batch_pairs}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 0: Load data
    # ------------------------------------------------------------------
    with Timer("Load expression + metadata", timings):
        expr_full = load_expression(cfg.counts_path)
        metadata = load_metadata(cfg.metadata_path)
        studies = discover_studies(
            expr_full, metadata,
            min_samples=cfg.network.min_samples_per_study,
        )

    if not studies:
        print("[ERROR] No studies discovered. Check paths and metadata.")
        sys.exit(1)

    info["Studies"] = len(studies)
    info["Expression matrix"] = cfg.counts_path

    # ------------------------------------------------------------------
    # Step 1: Process each study independently
    # ------------------------------------------------------------------
    study_results = []
    common_gene_names = None

    for study in studies:
        study_name = study["name"]
        expr_data = study["expr"]
        gene_names = study["gene_names"]
        n_genes = len(gene_names)
        n_samples = expr_data.shape[1]

        print(f"\n{'=' * 80}")
        print(f"STUDY: {study_name}  ({n_genes} genes, {n_samples} samples)")
        print("=" * 80)

        info[f"{study_name}: genes"] = n_genes
        info[f"{study_name}: samples"] = n_samples

        # Z-score (continuous, no discretisation)
        with Timer(f"{study_name}: Z-score + pre-screen", timings):
            X = zscore_expression(expr_data)
            pair_indices = prescreen_pairs(
                X,
                method=cfg.prescreen.method,
                threshold=cfg.prescreen.threshold,
                max_pairs=cfg.prescreen.max_pairs,
                n_jobs=cfg.n_jobs,
            )
            n_cand = len(pair_indices)
            info[f"{study_name}: candidate pairs"] = f"{n_cand:,}"

        if n_cand == 0:
            print(f"[WARN] No candidate pairs for {study_name} — skipping")
            continue

        # MINE MI estimation on candidate pairs
        with Timer(f"{study_name}: MINE MI estimation ({n_cand:,} pairs)", timings):
            mi_values = estimate_mi_for_pairs(
                X, pair_indices, cfg.mine, device, verbose=True,
            )
            info[f"{study_name}: MI range"] = (
                f"{mi_values[mi_values > 0].min():.4f} – {mi_values.max():.4f}"
                if (mi_values > 0).any() else "all zero"
            )

        # Build null distribution
        with Timer(f"{study_name}: MINE null ({cfg.permutation.n_permutations} perms)", timings):
            null_mi = build_mine_null(
                X, cfg.mine,
                n_permutations=cfg.permutation.n_permutations,
                seed=cfg.permutation.seed,
                device=device,
            )
            mi_thr = save_null_qc(
                null_mi, study_name,
                cfg.permutation.p_value_threshold, OUTPUT_DIR,
            )
            info[f"{study_name}: MI threshold (p<{cfg.permutation.p_value_threshold})"] = (
                f"{mi_thr:.4f}"
            )

        # Filter edges
        with Timer(f"{study_name}: edge filtering", timings):
            adj_sig, p_values = filter_edges_by_pvalue(
                mi_values, pair_indices, null_mi, n_genes,
                p_threshold=cfg.permutation.p_value_threshold,
            )
            n_edges = int(np.triu(adj_sig, k=1).sum())
            info[f"{study_name}: significant edges"] = f"{n_edges:,}"

        # Edge list
        edgelist_df = build_edgelist(
            adj_sig, pair_indices, mi_values, p_values, gene_names,
        )

        # Optional BH-FDR
        bh_df = None
        if cfg.apply_bh_fdr:
            bh_df = apply_bh_fdr(
                pair_indices, mi_values, p_values, gene_names,
                fdr_alpha=cfg.bh_fdr_alpha,
            )

        # Save
        with Timer(f"{study_name}: saving", timings):
            save_study_results(
                study_name, adj_sig, edgelist_df, gene_names,
                OUTPUT_DIR, bh_df=bh_df,
            )

        # Track for master network
        if common_gene_names is None:
            common_gene_names = gene_names
            study_results.append({"name": study_name, "adj": adj_sig})
        else:
            common_set = set(common_gene_names) & set(gene_names)
            if len(common_set) < len(common_gene_names):
                print(f"[WARN] Gene intersection: {len(common_gene_names)} → {len(common_set)}")
                old_idx = [common_gene_names.index(g) for g in common_gene_names
                           if g in common_set]
                common_gene_names = [g for g in common_gene_names if g in common_set]
                for prev in study_results:
                    prev["adj"] = prev["adj"][np.ix_(old_idx, old_idx)]
                idx_curr = [gene_names.index(g) for g in common_gene_names]
                adj_sig = adj_sig[np.ix_(idx_curr, idx_curr)]
            study_results.append({"name": study_name, "adj": adj_sig})

        del mi_values, null_mi, p_values, X
        gc.collect()

    # ------------------------------------------------------------------
    # Step 2: Master network
    # ------------------------------------------------------------------
    if not study_results:
        print("[ERROR] No study results to combine.")
        sys.exit(1)

    n_studies = len(study_results)
    if cfg.network.min_study_fraction is not None:
        effective_min = max(1, round(cfg.network.min_study_fraction * n_studies))
        print(f"\n[INFO] Dynamic min_study_count: "
              f"{cfg.network.min_study_fraction*100:.0f}% of {n_studies} = {effective_min}")
    elif n_studies < cfg.network.min_study_count:
        effective_min = 1
        print(f"[WARN] Only {n_studies} studies, setting min_study_count=1")
    else:
        effective_min = cfg.network.min_study_count

    with Timer("Master network construction", timings):
        master_adj, edge_count = build_master_network(
            study_results, common_gene_names, min_count=effective_min,
        )
        n_master = int(np.triu(master_adj, k=1).sum())
        info["Master: genes"] = len(common_gene_names)
        info["Master: edges"] = f"{n_master:,}"

    # MCODE
    with Timer("MCODE module detection", timings):
        modules, membership = mcode(
            master_adj, common_gene_names,
            score_threshold=cfg.mcode.score_threshold,
            min_size=cfg.mcode.min_size,
            min_density=cfg.mcode.min_density,
        )
        info["Master: MCODE modules"] = len(modules)

    # Save master
    with Timer("Save master network", timings):
        save_master_results(
            master_adj, edge_count, common_gene_names,
            modules, membership,
            min_count=effective_min, n_studies=n_studies,
            output_dir=OUTPUT_DIR,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = sum(timings.values())
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}")
    print(f"Runtime: {format_time(total)}")
    print(f"Output:  {OUTPUT_DIR}")

    print("\nTIMING SUMMARY")
    print("-" * 80)
    for step, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = t / total * 100 if total > 0 else 0
        print(f"  {step:55s}: {format_time(t):>20s} ({pct:5.1f}%)")

    save_report(timings, info, report_file)
    print("=" * 80)


if __name__ == "__main__":
    main()
