"""
Annotate discovered modules using pathway databases.

For each of the 128 latent dimensions (modules), takes the top N genes
from the empirical method and tests for enrichment against:
  - GO Biological Process
  - KEGG Pathways
  - Reactome Pathways
  - WikiPathways
  - MSigDB Hallmark Gene Sets
  - Blood Transcription Modules (BTM, Li et al. 2014)

Uses Fisher's exact test with Benjamini-Hochberg FDR correction.

Gene sets are downloaded from Enrichr (Ma'ayan Lab, Mount Sinai) in GMT format.
Since the databases use human gene symbols, we first map our pig Ensembl IDs
to gene symbols using the mapping from map_gene_ids.py, then use ortholog
symbol matching (pig gene symbols are often identical or very similar to human).

Usage:
    python annotate_modules.py

Prerequisites:
    - run_real_data.py completed (produces top_genes_per_dim_empirical.csv)
    - map_gene_ids.py completed (produces gene_id_mapping.csv)
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact

RESULTS_DIR = Path("results/real_data")
INTERP_DIR = RESULTS_DIR / "interpretation"
MAPPING_FILE = RESULTS_DIR / "gene_id_mapping.csv"
OUTPUT_FILE = RESULTS_DIR / "module_annotations.csv"
GMT_CACHE_DIR = RESULTS_DIR / "gmt_cache"

# Top N genes per module to use for enrichment
TOP_N = 100

# Enrichr gene set library names
# These are downloaded as tab-separated GMT-like files from the Enrichr API
ENRICHR_LIBRARIES = {
    "GO_Biological_Process_2023": "GO Biological Process",
    "KEGG_2021_Human": "KEGG",
    "Reactome_2022": "Reactome",
    "WikiPathway_2023_Human": "WikiPathways",
    "MSigDB_Hallmark_2020": "MSigDB Hallmark",
}

# BTM is not on Enrichr — we include a built-in compact set
# Li et al. 2014 "Molecular signatures of antibody responses..."
BTM_URL = None  # Will handle separately


def download_enrichr_library(library_name):
    """Download a gene set library from the Enrichr API."""
    cache_file = GMT_CACHE_DIR / f"{library_name}.gmt"
    if cache_file.exists():
        return cache_file

    GMT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library_name}"

    print(f"    Downloading {library_name}...")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 100:
                cache_file.write_text(resp.text, encoding="utf-8")
                return cache_file
            else:
                print(f"    Attempt {attempt+1}: HTTP {resp.status_code}, length={len(resp.text)}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"    Attempt {attempt+1}: {e}")
            time.sleep(3)

    print(f"    WARNING: Failed to download {library_name}")
    return None


def parse_enrichr_gmt(filepath):
    """Parse an Enrichr GMT file into {pathway_name: set_of_genes}."""
    pathways = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                name = parts[0]
                # Enrichr GMT: name \t description \t gene1 \t gene2 ...
                # Some entries have weights like GENE,1.0 — strip weights
                genes = set()
                for g in parts[2:]:
                    g = g.split(",")[0].strip()
                    if g:
                        genes.add(g.upper())
                if genes:
                    pathways[name] = genes
    return pathways


def get_built_in_btm():
    """Return a small set of key Blood Transcription Modules (Li et al. 2014).
    
    These are representative BTM gene sets commonly used in vaccine research.
    Gene symbols are human — we match by uppercase symbol.
    """
    btm = {
        "BTM_M4.0 cell cycle (I)": {"CDK1", "CCNB1", "CCNB2", "CDC20", "BUB1", "AURKA", "AURKB", "PLK1", "TOP2A", "KIF11", "CENPE", "CENPF", "KIF2C", "BIRC5", "NUSAP1", "PRC1", "CDCA8", "NDC80", "TTK", "UBE2C"},
        "BTM_M6.6 antiviral IFN signature": {"IFI44L", "IFIT1", "IFIT3", "IFI6", "RSAD2", "MX1", "OAS1", "OAS2", "OAS3", "ISG15", "IFI44", "HERC5", "USP18", "SIGLEC1", "LAMP3", "OASL", "IFIT2", "DDX60", "IFI35", "XAF1"},
        "BTM_M7.1 enriched in T cells (I)": {"CD3D", "CD3E", "CD3G", "CD6", "CD28", "LCK", "ZAP70", "ITK", "TRAT1", "CD247", "TRGC2", "TRBC1", "GZMK", "CCR7", "TCF7", "LEF1", "IL7R", "SELL", "CD27", "BCL11B"},
        "BTM_M7.3 T cell activation (II)": {"CD69", "ICOS", "IL2RA", "TNFRSF9", "LAG3", "CTLA4", "HAVCR2", "TIGIT", "PDCD1", "TOX", "IFNG", "GZMB", "PRF1", "FASLG", "TNF", "NKG7", "CXCR3", "CCL5", "CCL4", "XCL1"},
        "BTM_M4.1 cell cycle (II)": {"MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "ORC1", "CDC6", "CDT1", "PCNA", "RRM2", "TK1", "TYMS", "POLA1", "RFC3", "RFC4", "FEN1", "RAD51", "BRCA1", "CHEK1"},
        "BTM_M11.0 enriched in monocytes (II)": {"CD14", "CD163", "FCGR1A", "FCGR2A", "FCGR3A", "CSF1R", "MERTK", "SIGLEC1", "TLR2", "TLR4", "TLR8", "CLEC7A", "MRC1", "CD68", "MSR1", "MARCO", "MPEG1", "LILRB2", "S100A8", "S100A9"},
        "BTM_M15 enriched in B cells (I)": {"CD19", "CD79A", "CD79B", "MS4A1", "CD22", "BANK1", "BLK", "BLNK", "PAX5", "TCL1A", "FCER2", "FCRL1", "FCRL2", "IGHM", "IGHD", "CD72", "VPREB3", "STAP1", "SPIB", "CR2"},
        "BTM_M16 TLR and inflammatory signaling": {"TLR1", "TLR2", "TLR4", "TLR6", "TLR8", "MYD88", "IRAK4", "TRAF6", "NFKB1", "RELA", "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4", "CXCL10", "PTGS2", "NLRP3"},
        "BTM_M37.0 immune activation (I)": {"STAT1", "STAT2", "IRF1", "IRF7", "IRF9", "GBP1", "GBP2", "GBP4", "GBP5", "TRIM22", "TRIM25", "TRIM5", "BST2", "PSMB8", "PSMB9", "TAP1", "TAP2", "B2M", "HLA-A", "HLA-B"},
        "BTM_M47.0 enriched in NK cells (I)": {"NCAM1", "NCR1", "NCR3", "KLRD1", "KLRF1", "KIR2DL1", "KIR2DL3", "KIR3DL1", "KIR3DL2", "GNLY", "GZMB", "PRF1", "NKG7", "FGFBP2", "SPON2", "CLIC3", "HOPX", "TBX21", "EOMES", "IL2RB"},
        "BTM_M47.1 enriched in NK cells (II)": {"KLRC1", "KLRK1", "KLRB1", "KLRG1", "CD160", "CD244", "FCGR3A", "SH2D1B", "MYOM2", "CMC1", "BNC2", "ZNF683", "PRSS23", "SLC22A5", "PLEK", "DTHD1", "MATK", "PTGDR", "CTSW", "IL18RAP"},
        "BTM_M35.0 enriched in B cells (II)": {"JCHAIN", "MZB1", "XBP1", "IRF4", "PRDM1", "SDC1", "TNFRSF17", "TXNDC5", "SSR4", "DERL3", "FKBP11", "SEC11C", "CREB3L1", "SLC38A5", "POU2AF1", "IGHA1", "IGHG1", "IGHG2", "IGHG3", "IGHG4"},
        "BTM_M67 activated dendritic cells": {"CD80", "CD86", "CD40", "CD83", "CCR7", "LAMP3", "IDO1", "IDO2", "CCL17", "CCL19", "CCL22", "FSCN1", "MARCKSL1", "IL12B", "EBI3", "SOCS2", "NFKBIZ", "RELB", "BCL2L1", "BIRC3"},
        "BTM_S1 IFN-stimulated genes": {"MX1", "MX2", "ISG15", "ISG20", "IFIT1", "IFIT2", "IFIT3", "IFIT5", "IFI6", "IFI27", "IFI44", "IFI44L", "IFITM1", "IFITM3", "OAS1", "OAS2", "OAS3", "OASL", "RSAD2", "DDX58", "HERC5", "USP18", "CMPK2", "EPSTI1", "LY6E", "SAMD9L", "SAMD9", "PARP9", "DTX3L", "PLSCR1"},
        "BTM_S4 monocyte surface signature": {"CD14", "FCGR1A", "FCGR2A", "FCGR3A", "CSF1R", "ITGAM", "CD36", "CD163", "LILRB1", "LILRB2", "LILRB4", "LILRA5", "SIGLEC9", "CLEC4E", "CLEC5A", "DPEP2", "PILRA", "TLR2", "TLR4", "TNFAIP2"},
        "BTM_M75 antigen presentation": {"HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSME1", "PSME2", "CIITA", "NLRC5", "RFX5"},
    }
    return btm


def load_gene_mapping():
    """Load the Ensembl→symbol mapping produced by map_gene_ids.py."""
    if not MAPPING_FILE.exists():
        print(f"ERROR: {MAPPING_FILE} not found. Run map_gene_ids.py first.")
        sys.exit(1)

    df = pd.read_csv(MAPPING_FILE)
    mapping = {}
    for _, row in df.iterrows():
        sym = row.get("gene_symbol")
        if pd.notna(sym) and str(sym).strip() and str(sym).strip().upper() != "NAN":
            mapping[row["ensembl_id"]] = str(sym).strip().upper()
    print(f"  Loaded mapping: {len(mapping)} IDs have symbols")
    return mapping


def load_modules(gene_mapping, top_n=TOP_N):
    """Load top genes per module from the FULL loading matrix and convert to uppercase gene symbols."""
    # Use the full loading matrix for more genes
    full_matrix = INTERP_DIR / "gene_loadings_empirical.csv"
    if not full_matrix.exists():
        print(f"ERROR: {full_matrix} not found. Run run_real_data.py first.")
        sys.exit(1)

    df = pd.read_csv(full_matrix, index_col=0)
    print(f"  Full loading matrix: {df.shape}")

    modules = {}
    for col in df.columns:
        # Get top N genes by absolute loading
        abs_vals = df[col].abs().sort_values(ascending=False)
        top_ids = abs_vals.head(top_n).index.tolist()

        symbols = set()
        for ens_id in top_ids:
            sym = gene_mapping.get(ens_id)
            if sym:
                symbols.add(sym)
        if symbols:
            modules[col] = symbols

    print(f"  Loaded {len(modules)} modules (top {top_n} genes each)")
    sizes = [len(v) for v in modules.values()]
    print(f"  Mapped genes per module: min={min(sizes)}, median={np.median(sizes):.0f}, max={max(sizes)}")
    return modules


def get_background_genes(gene_mapping):
    """All gene symbols in our dataset = background for enrichment."""
    return set(gene_mapping.values())


def fisher_enrichment(module_genes, pathway_genes, background_size):
    """Fisher's exact test (one-sided, over-representation).
    
    Contingency table:
                  In pathway   Not in pathway
    In module       a              b
    Not in module   c              d
    """
    a = len(module_genes & pathway_genes)
    b = len(module_genes - pathway_genes)
    c = len(pathway_genes - module_genes)
    d = background_size - a - b - c
    if d < 0:
        d = 0
    if a < 2:  # require at least 2 overlapping genes
        return 1.0, a
    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    return pval, a


def benjamini_hochberg(pvalues):
    """BH-FDR correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = np.array(pvalues)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            adjusted[i] = sorted_pvals[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * n / rank)
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


def run_enrichment(modules, all_pathways, background_size):
    """Run enrichment for all modules × all pathways."""
    results = []

    total_modules = len(modules)
    for idx, (dim, mod_genes) in enumerate(sorted(modules.items())):
        if (idx + 1) % 20 == 0:
            print(f"    Module {idx+1}/{total_modules}...")

        # Collect all p-values for this module (for FDR across all pathways)
        module_results = []
        for db_name, db_pathways in all_pathways.items():
            for pw_name, pw_genes in db_pathways.items():
                pval, overlap = fisher_enrichment(mod_genes, pw_genes, background_size)
                shared = sorted(mod_genes & pw_genes)
                module_results.append({
                    "module": dim,
                    "database": db_name,
                    "pathway": pw_name,
                    "overlap": overlap,
                    "module_size": len(mod_genes),
                    "pathway_size": len(pw_genes),
                    "pvalue": pval,
                    "shared_genes": ";".join(shared),
                })

        # FDR correction within this module
        pvals = [r["pvalue"] for r in module_results]
        fdr = benjamini_hochberg(pvals)
        for r, q in zip(module_results, fdr):
            r["fdr"] = q

        # Keep only significant (FDR < 0.05) or top hits
        significant = [r for r in module_results if r["fdr"] < 0.05]
        if not significant:
            # Keep the single best hit even if not significant
            module_results.sort(key=lambda x: x["pvalue"])
            significant = module_results[:1]
        else:
            significant.sort(key=lambda x: x["fdr"])

        results.extend(significant[:20])  # cap at top 20 per module

    return pd.DataFrame(results)


def summarize_annotations(ann_df):
    """Print a nice summary of the annotation results."""
    if ann_df.empty:
        print("  No significant annotations found.")
        return

    # Multiple FDR thresholds
    for fdr_thresh in [0.05, 0.10, 0.25]:
        sig = ann_df[ann_df["fdr"] < fdr_thresh]
        n_mods = sig["module"].nunique() if len(sig) > 0 else 0
        print(f"\n  FDR < {fdr_thresh}: {len(sig)} hits across {n_mods} modules")

        if len(sig) == 0:
            continue

        # Per-database summary
        for db, grp in sig.groupby("database"):
            n_m = grp["module"].nunique()
            print(f"    {db}: {len(grp)} hits in {n_m} modules")

    # Also report nominal p < 0.01 (no FDR)
    nom_sig = ann_df[ann_df["pvalue"] < 0.01]
    print(f"\n  Nominal p < 0.01 (no FDR): {len(nom_sig)} hits "
          f"across {nom_sig['module'].nunique()} modules")

    # Top annotations per database (by nominal p)
    print(f"\n  Top annotations per database (by p-value):")
    for db in sorted(ann_df["database"].unique()):
        db_hits = ann_df[ann_df["database"] == db].nsmallest(5, "pvalue")
        for _, row in db_hits.iterrows():
            marker = " **" if row["fdr"] < 0.05 else " *" if row["fdr"] < 0.25 else ""
            print(f"    [{db}] {row['module']}: {row['pathway'][:70]} "
                  f"(overlap={row['overlap']}, p={row['pvalue']:.2e}, "
                  f"FDR={row['fdr']:.2e}){marker}")

    # Modules with most annotations
    relaxed = ann_df[ann_df["fdr"] < 0.25]
    if len(relaxed) > 0:
        mod_counts = relaxed.groupby("module").size().sort_values(ascending=False)
        print(f"\n  Most annotated modules (FDR<0.25):")
        for mod, cnt in mod_counts.head(10).items():
            top_hit = relaxed[relaxed["module"] == mod].iloc[0]
            print(f"    {mod}: {cnt} annotations, best = {top_hit['pathway'][:60]} "
                  f"(FDR={top_hit['fdr']:.2e})")


def main():
    print("=" * 65)
    print("  Module Annotation — Pathway Enrichment Analysis")
    print("=" * 65)

    # 1. Load gene mapping
    print("\n[1/4] Loading gene ID mapping...")
    gene_mapping = load_gene_mapping()

    # 2. Load modules
    print("\n[2/4] Loading modules (empirical method)...")
    modules = load_modules(gene_mapping, top_n=TOP_N)
    background = get_background_genes(gene_mapping)
    bg_size = len(background)
    print(f"  Background gene universe: {bg_size} genes")

    # 3. Download / load pathway databases
    print("\n[3/4] Loading pathway databases...")
    all_pathways = {}

    for lib_name, display_name in ENRICHR_LIBRARIES.items():
        cache = download_enrichr_library(lib_name)
        if cache:
            pathways = parse_enrichr_gmt(cache)
            all_pathways[display_name] = pathways
            print(f"    {display_name}: {len(pathways)} gene sets")

    # Add built-in BTM
    btm = get_built_in_btm()
    all_pathways["BTM (Li et al.)"] = btm
    print(f"    BTM (Li et al.): {len(btm)} gene sets")

    total_sets = sum(len(v) for v in all_pathways.values())
    print(f"  Total: {total_sets} gene sets across {len(all_pathways)} databases")

    # 4. Run enrichment
    print(f"\n[4/4] Running Fisher's exact test (top {TOP_N} genes/module)...")
    ann_df = run_enrichment(modules, all_pathways, bg_size)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ann_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved {OUTPUT_FILE}")
    print(f"  Total rows: {len(ann_df)}")

    # Summary
    summarize_annotations(ann_df)

    # Also save a compact per-module summary
    if not ann_df.empty:
        compact = (
            ann_df[ann_df["fdr"] < 0.05]
            .sort_values(["module", "fdr"])
            .groupby("module")
            .first()
            .reset_index()
            [["module", "database", "pathway", "overlap", "fdr"]]
        )
        compact_path = RESULTS_DIR / "module_top_annotation.csv"
        compact.to_csv(compact_path, index=False)
        print(f"\n  Per-module best annotation: {compact_path}")
        print(f"  Modules with FDR<0.05 annotation: {len(compact)}/{len(modules)}")

    print("\n" + "=" * 65)
    print("  DONE — Module annotations complete")
    print("=" * 65)


if __name__ == "__main__":
    main()
