#!/usr/bin/env python3
"""
run_pipeline.py — Command-line entry point for MINE gene network inference.
============================================================================

Usage
-----
Default paths (auto-detect local course data or HPC)::

    python run_pipeline.py

Override output directory::

    python run_pipeline.py --output ./my_results

Use a specific Python environment (PowerShell)::

    & "C:\\path\\to\\python.exe" run_pipeline.py

Full custom paths::

    python run_pipeline.py \\
        --counts  /path/to/logCPM_matrix.csv \\
        --meta    /path/to/metadata.csv \\
        --output  /path/to/results \\
        --device  cuda \\
        --perms   1000 \\
        --pval    0.001 \\
        --mode    global

With gene-set annotation::

    python run_pipeline.py --gmt hallmark.gmt reactome.gmt

Disable pre-screening (only practical for small gene sets)::

    python run_pipeline.py --no-prescreen
"""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mine_network.config import PipelineConfig
from mine_network.pipeline import run_pipeline


def _auto_detect_data() -> tuple:
    """
    Try to find expression + metadata files automatically.

    Searches (in order):
      1. ../Project_plan/counts_and_metadata/  (local course structure)
      2. HPC path
    """
    project_root = Path(__file__).resolve().parent.parent

    candidates = [
        project_root / "Project_plan" / "counts_and_metadata",
        Path("/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/"
             "workingEnvironment/02_counts"),
    ]

    for d in candidates:
        counts = d / "logCPM_matrix_filtered_samples.csv"
        meta = d / "metadata_with_sample_annotations.csv"
        if counts.exists() and meta.exists():
            return str(counts), str(meta)

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="MINE-based gene network inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--counts", type=str, default=None,
                        help="Path to logCPM expression matrix (tab-separated).")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to sample metadata (tab-separated).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ./output).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device (default: auto).")

    # MINE parameters
    parser.add_argument("--hidden", type=int, default=64,
                        help="MINE hidden-layer width (default: 64).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="MINE training epochs per batch (default: 200).")
    parser.add_argument("--batch-pairs", type=int, default=512,
                        help="Gene pairs per MINE batch (default: 512).")

    # Pre-screening
    parser.add_argument("--no-prescreen", action="store_true",
                        help="Disable Pearson pre-screening (test all pairs).")
    parser.add_argument("--prescreen-threshold", type=float, default=0.3,
                        help="Pre-screen |r| threshold (default: 0.3).")
    parser.add_argument("--prescreen-method", type=str, default="pearson",
                        choices=["pearson", "spearman"],
                        help="Correlation method for pre-screening (default: pearson).")
    parser.add_argument("--max-pairs", type=int, default=5_000_000,
                        help="Max candidate pairs per study after pre-screen "
                             "(default: 5000000). Excess pairs are capped by "
                             "raising the |r| threshold dynamically.")

    # QC and optional MAD filtering
    parser.add_argument("--mad-top-genes", type=int, default=None,
                        help="Keep only top-N genes by MAD before study splitting."
                             " Example: 5000")
    parser.add_argument("--qc-preplot", action="store_true",
                        help="Save pre-filter QC figure (dendrogram + sample"
                             " distribution lines + Spearman heatmap).")
    parser.add_argument("--qc-postplot", action="store_true",
                        help="Save post-filter QC figure using the filtered"
                             " matrix (after MAD selection).")
    parser.add_argument("--qc-quantiles", type=int, default=200,
                        help="Number of quantile points in QC sample"
                             " distribution line plots (default: 200).")

    # Permutation
    parser.add_argument("--perms", type=int, default=10000,
                        help="Number of permutations (default: 10000).")
    parser.add_argument("--pval", type=float, default=0.001,
                        help="P-value threshold (default: 0.001).")
    parser.add_argument("--mode", type=str, default="global",
                        choices=["global", "per_pair"],
                        help="Null distribution mode (default: global).")

    # Network
    parser.add_argument("--min-studies", type=int, default=3,
                        help="Min studies for master edge (default: 3).")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Min samples per study (default: 3).")
    parser.add_argument("--module-method", type=str, default="mcode",
                        choices=["mcode", "leiden"],
                        help="First-pass module detector on master network.")
    parser.add_argument("--master-edge-weight", type=str,
                        default="n_studies",
                        choices=["n_studies", "mean_mi", "mean_neglog10p"],
                        help="Master edge weighting mode.")
    parser.add_argument("--normalize-weights", action="store_true",
                        help="Normalize study-level weights before aggregation.")
    parser.add_argument("--weight-clip-min", type=float, default=None,
                        help="Optional lower clip for study-level weights.")
    parser.add_argument("--weight-clip-max", type=float, default=None,
                        help="Optional upper clip for study-level weights.")
    parser.add_argument("--weight-eps", type=float, default=1e-12,
                        help="Epsilon for significance weight -log10(p + eps).")
    parser.add_argument("--leiden-resolution", type=float, default=1.0,
                        help="Leiden resolution parameter.")
    parser.add_argument("--leiden-iterations", type=int, default=-1,
                        help="Leiden iterations (-1 uses igraph default).")
    parser.add_argument("--submodule-size-threshold", type=int, default=None,
                        help="If set, rerun MCODE inside modules larger than"
                             " this size.")

    # Annotation
    parser.add_argument("--gmt", type=str, nargs="*", default=[],
                        help="GMT gene-set files for module annotation.")
    parser.add_argument("--download-gmt", action="store_true",
                        help="Download gene-set libraries from Enrichr API.")
    parser.add_argument("--enrichr-libs", type=str, nargs="*", default=[],
                        help="Specific Enrichr library names to download. "
                             "If --download-gmt is set without this, downloads "
                             "all defaults (GO, KEGG, Reactome, WikiPathway, MSigDB Hallmark).")

    args = parser.parse_args()

    # ── Build config ──
    cfg = PipelineConfig()

    # Paths
    if args.counts and args.meta:
        cfg.counts_path = args.counts
        cfg.metadata_path = args.meta
    else:
        auto_counts, auto_meta = _auto_detect_data()
        if auto_counts:
            cfg.counts_path = auto_counts
            cfg.metadata_path = auto_meta
            print(f"[INFO] Auto-detected data: {auto_counts}")
        else:
            print("[ERROR] Could not auto-detect data files. "
                  "Provide --counts and --meta.")
            sys.exit(1)

    cfg.output_dir = args.output or str(
        Path(__file__).resolve().parent / "output"
    )
    cfg.device = args.device

    # MINE
    cfg.mine.hidden_dim = args.hidden
    cfg.mine.n_epochs = args.epochs
    cfg.mine.batch_pairs = args.batch_pairs

    # Pre-screen
    cfg.prescreen.enabled = not args.no_prescreen
    cfg.prescreen.threshold = args.prescreen_threshold
    cfg.prescreen.method = args.prescreen_method
    cfg.prescreen.max_pairs = args.max_pairs

    # QC + MAD
    cfg.qc.mad_top_genes = args.mad_top_genes
    cfg.qc.plot_pre_filter = args.qc_preplot
    cfg.qc.plot_post_filter = args.qc_postplot
    cfg.qc.line_quantiles = args.qc_quantiles

    # Permutation
    cfg.permutation.n_permutations = args.perms
    cfg.permutation.p_value_threshold = args.pval
    cfg.permutation.mode = args.mode

    # Network
    cfg.network.min_study_count = args.min_studies
    cfg.network.min_samples_per_study = args.min_samples

    # Module detection + edge weighting
    cfg.module.method = args.module_method
    cfg.module.master_edge_weight = args.master_edge_weight
    cfg.module.normalize_weights = args.normalize_weights
    cfg.module.weight_clip_min = args.weight_clip_min
    cfg.module.weight_clip_max = args.weight_clip_max
    cfg.module.weight_eps = args.weight_eps
    cfg.module.leiden_resolution = args.leiden_resolution
    cfg.module.leiden_iterations = args.leiden_iterations
    cfg.module.submodule_size_threshold = args.submodule_size_threshold

    # Annotation
    if args.gmt:
        cfg.annotation.gmt_paths = args.gmt
    if args.download_gmt:
        cfg.annotation.download_enrichr = True
        if args.enrichr_libs:
            cfg.annotation.enrichr_libraries = args.enrichr_libs

    # ── Run ──
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
