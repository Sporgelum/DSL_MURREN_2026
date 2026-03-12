"""
Utility functions: data loading, logging, timing, edge filtering, saving.

Adapted from generate_net_python_pval.py for the MINE-based pipeline.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import igraph as ig
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Logging / Timing
# ═══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """Write to both stdout and a log file."""
    def __init__(self, log_file):
        self.terminal = sys.__stdout__
        self.log = open(log_file, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Timer:
    """Context manager that records wall-clock time for a named step."""
    def __init__(self, name, report_dict):
        self.name = name
        self.report_dict = report_dict

    def __enter__(self):
        self.start = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Starting: {self.name}")
        print("-" * 80)
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("-" * 80)
        print(f"[{ts}] Completed: {self.name}")
        print(f"[TIMING] Duration: {format_time(self.elapsed)}")
        self.report_dict[self.name] = self.elapsed


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes ({seconds:.1f}s)"
    else:
        h = seconds / 3600
        m = (seconds % 3600) / 60
        return f"{h:.2f} hours ({m:.1f}m)"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_expression(counts_path):
    """Load logCPM matrix (genes × samples), tab-separated."""
    expr = pd.read_csv(counts_path, sep="\t", index_col=0)
    print(f"[INFO] Expression: {expr.shape[0]} genes x {expr.shape[1]} samples")
    return expr


def load_metadata(metadata_path):
    """Load metadata with Run and BioProject columns."""
    md = pd.read_csv(metadata_path, sep="\t")
    for col in ("Run", "BioProject"):
        if col not in md.columns:
            raise ValueError(f"Metadata missing required column '{col}'")
    return md


def discover_studies(expr_full, metadata, min_samples=10):
    """
    Auto-discover studies from BioProject column.

    Returns list of dicts: {name, expr (genes × study_samples), gene_names}.
    """
    available = set(expr_full.columns)
    md_matched = metadata[metadata["Run"].isin(available)].copy()

    studies = []
    for bioproj, group in md_matched.groupby("BioProject"):
        runs = group["Run"].tolist()
        if len(runs) < min_samples:
            print(f"[WARN] {bioproj}: {len(runs)} samples < {min_samples} — skipping")
            continue
        sub = expr_full[runs]
        safe_name = str(bioproj).replace(" ", "_").replace("/", "-")
        studies.append({
            "name": safe_name,
            "expr": sub,
            "gene_names": sub.index.tolist(),
        })
        print(f"[INFO] Study: {safe_name} ({len(runs)} samples)")

    print(f"[INFO] Total studies: {len(studies)}")
    return studies


def zscore_expression(expr_data):
    """
    Z-score each gene across samples (mean=0, std=1).

    This standardises marginals so the permutation null is approximately
    gene-pair-agnostic (all genes ≈ N(0,1) after Z-scoring).
    """
    X = expr_data.values.astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero for constant genes
    return (X - mu) / std


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_edges_by_pvalue(mi_values, pair_indices, null_mi, n_genes,
                           p_threshold=0.001):
    """
    Filter candidate pairs by empirical p-value against the MINE null.

    Returns: adj_significant (n×n uint8), p_values (per candidate pair)
    """
    null_sorted = np.sort(null_mi)
    n_perm = len(null_sorted)

    # Vectorised p-value: fraction of null >= observed
    insert_idx = np.searchsorted(null_sorted, mi_values, side="left")
    p_values = (n_perm - insert_idx) / n_perm

    # Build adjacency
    sig_mask = p_values < p_threshold
    adj = np.zeros((n_genes, n_genes), dtype=np.uint8)
    sig_pairs = pair_indices[sig_mask]
    if len(sig_pairs) > 0:
        adj[sig_pairs[:, 0], sig_pairs[:, 1]] = 1
        adj[sig_pairs[:, 1], sig_pairs[:, 0]] = 1

    n_sig = sig_mask.sum()
    print(f"[INFO] Significant edges (p < {p_threshold}): {n_sig:,}")
    return adj, p_values


def build_edgelist(adj, pair_indices, mi_values, p_values, gene_names):
    """Build tidy DataFrame of significant edges."""
    gene_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(adj, k=1) == 1)

    # Map back to MI and p-value via pair_indices lookup
    pair_set = {(i, j): k for k, (i, j) in enumerate(pair_indices)}
    mi_list, p_list = [], []
    for r, c in zip(rows, cols):
        key = (min(r, c), max(r, c))
        k = pair_set.get(key, None)
        mi_list.append(mi_values[k] if k is not None else 0.0)
        p_list.append(p_values[k] if k is not None else 1.0)

    df = pd.DataFrame({
        "gene_A": gene_arr[rows],
        "gene_B": gene_arr[cols],
        "MI_MINE": mi_list,
        "p_value": p_list,
    })
    df.sort_values("p_value", inplace=True)
    return df


def apply_bh_fdr(pair_indices, mi_values, p_values, gene_names, fdr_alpha=0.05):
    """Benjamini-Hochberg FDR correction on candidate pairs."""
    n_tests = len(p_values)
    if n_tests == 0:
        return pd.DataFrame(columns=["gene_A", "gene_B", "MI_MINE", "p_value", "p_adjusted"])

    rank_order = np.argsort(p_values)
    sorted_p = p_values[rank_order]
    ranks = np.arange(1, n_tests + 1)
    bh_threshold = (ranks / n_tests) * fdr_alpha

    below = sorted_p <= bh_threshold
    if not below.any():
        print(f"[INFO] BH-FDR: no edges survive at alpha={fdr_alpha}")
        return pd.DataFrame(columns=["gene_A", "gene_B", "MI_MINE", "p_value", "p_adjusted"])

    cutoff = sorted_p[below].max()
    sig_mask = p_values <= cutoff
    gene_arr = np.array(gene_names)

    sig_idx = pair_indices[sig_mask]
    df = pd.DataFrame({
        "gene_A": gene_arr[sig_idx[:, 0]],
        "gene_B": gene_arr[sig_idx[:, 1]],
        "MI_MINE": mi_values[sig_mask],
        "p_value": p_values[sig_mask],
    })
    # Adjusted p-values
    n_sig = len(df)
    rank_in_sig = np.argsort(np.argsort(df["p_value"].values)) + 1
    df["p_adjusted"] = np.minimum(1.0, df["p_value"].values * n_tests / rank_in_sig)
    df.sort_values("p_adjusted", inplace=True)
    print(f"[INFO] BH-FDR edges (alpha={fdr_alpha}): {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Master Network
# ═══════════════════════════════════════════════════════════════════════════════

def build_master_network(study_results, gene_names, min_count=3):
    """Combine per-study adjacency into a consensus master network."""
    n = len(gene_names)
    edge_count = np.zeros((n, n), dtype=np.int16)
    for res in study_results:
        edge_count += res["adj"].astype(np.int16)
    edge_count = np.maximum(edge_count, edge_count.T)

    master_adj = (edge_count >= min_count).astype(np.uint8)
    np.fill_diagonal(master_adj, 0)

    n_edges = int(np.triu(master_adj, k=1).sum())
    print(f"[INFO] Master network: {n_edges:,} edges "
          f"(in >= {min_count} of {len(study_results)} studies)")
    return master_adj, edge_count


# ═══════════════════════════════════════════════════════════════════════════════
# Saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_null_qc(null_mi, study_name, p_threshold, output_dir):
    """Save null distribution summary for QC."""
    mi_thr = np.percentile(null_mi, (1.0 - p_threshold) * 100)
    out = os.path.join(output_dir, f"null_distribution_{study_name}.txt")
    with open(out, "w") as f:
        f.write(f"Null MI distribution (MINE) — {study_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"N permutations : {len(null_mi)}\n")
        f.write(f"Mean           : {null_mi.mean():.6f}\n")
        f.write(f"Std            : {null_mi.std():.6f}\n")
        f.write(f"99.9th pct     : {np.percentile(null_mi, 99.9):.6f}\n")
        f.write(f"MI threshold   : {mi_thr:.6f}  (p < {p_threshold})\n")
    print(f"[SAVED] {out}")
    return mi_thr


def save_study_results(study_name, adj, edgelist_df, gene_names, output_dir,
                       bh_df=None):
    """Save per-study edges and network files."""
    print(f"\n[INFO] Saving results for: {study_name}")

    edgelist_df.to_csv(
        os.path.join(output_dir, f"edges_mine_{study_name}.tsv"),
        sep="\t", index=False,
    )

    mtx_file = os.path.join(output_dir, f"adj_mine_{study_name}.mtx")
    mmwrite(mtx_file, csr_matrix(adj))

    adj_sym = np.maximum(adj, adj.T)
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    g.vs["name"] = gene_names
    g.write_graphml(os.path.join(output_dir, f"network_mine_{study_name}.graphml"))

    if bh_df is not None and len(bh_df) > 0:
        bh_df.to_csv(
            os.path.join(output_dir, f"edges_bh_fdr_{study_name}.tsv"),
            sep="\t", index=False,
        )

    print(f"[SAVED] Study {study_name}: edges, adjacency, GraphML")


def save_master_results(master_adj, edge_count, gene_names,
                        modules, membership, min_count, n_studies, output_dir):
    """Save master network, modules, and subgraph files."""
    print("\n[INFO] Saving master network...")
    gene_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(master_adj, k=1) == 1)

    # Edge list with study count
    pd.DataFrame({
        "gene_A": gene_arr[rows],
        "gene_B": gene_arr[cols],
        "n_studies": edge_count[rows, cols],
    }).sort_values("n_studies", ascending=False).to_csv(
        os.path.join(output_dir, "master_network_edgelist.tsv"),
        sep="\t", index=False,
    )

    # Adjacency + study count sparse matrices
    mmwrite(os.path.join(output_dir, "master_network_adjacency.mtx"),
            csr_matrix(master_adj))
    mmwrite(os.path.join(output_dir, "master_edge_study_counts.mtx"),
            csr_matrix(np.triu(edge_count)))

    # GraphML
    adj_sym = np.maximum(master_adj, master_adj.T)
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    g.vs["name"] = gene_names
    g.write_graphml(os.path.join(output_dir, "master_network.graphml"))

    # Module membership
    btm_rows = [{"Gene": gene, "Module": f"M{mid}"}
                for mid, genes in modules.items() for gene in genes]
    pd.DataFrame(btm_rows).to_csv(
        os.path.join(output_dir, "master_BTM_modules.tsv"), sep="\t", index=False)

    pd.DataFrame([{"gene": g, "module": m} for g, m in membership.items()]).to_csv(
        os.path.join(output_dir, "master_node_modules.tsv"), sep="\t", index=False)

    # Per-module subgraphs
    gene_name_list = list(gene_names)
    saved = 0
    for mid, mod_genes in modules.items():
        if len(mod_genes) < 3:
            continue
        idx = [gene_name_list.index(gn) for gn in mod_genes]
        sub = np.maximum(master_adj[np.ix_(idx, idx)],
                         master_adj[np.ix_(idx, idx)].T)
        sg = ig.Graph.Adjacency((sub > 0).tolist(), mode="undirected")
        sg.vs["name"] = mod_genes
        sg.write_graphml(os.path.join(output_dir, f"master_submodule_M{mid}.graphml"))
        saved += 1
    print(f"[SAVED] Master: edgelist, adjacency, GraphML, "
          f"{len(modules)} modules, {saved} subgraphs")


def save_report(timings, info, report_path):
    """Write summary report to text file."""
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MINE-BASED GENE NETWORK INFERENCE — REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("RESULTS\n" + "-" * 80 + "\n")
        for k, v in info.items():
            f.write(f"  {k:50s}: {v}\n")

        f.write("\nTIMING\n" + "-" * 80 + "\n")
        total = sum(timings.values())
        for step, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            pct = t / total * 100 if total > 0 else 0
            f.write(f"  {step:55s}: {format_time(t):>20s} ({pct:5.1f}%)\n")
        f.write(f"\n  {'TOTAL':55s}: {format_time(total):>20s}\n")
    print(f"[SAVED] Report: {report_path}")
