"""
Map Ensembl gene IDs (ENSSSCG*) to gene symbols using the Ensembl REST API.

Species: Sus scrofa (pig)

Reads all unique gene IDs from the real-data loading matrices,
queries Ensembl in batches, and produces a mapping CSV.
Also re-exports top-genes-per-dim files with gene symbols added.

Usage:
    python map_gene_ids.py
"""

import json
import time
import sys
from pathlib import Path

import pandas as pd
import requests

RESULTS_DIR = Path("results/real_data")
INTERP_DIR = RESULTS_DIR / "interpretation"
MAPPING_FILE = RESULTS_DIR / "gene_id_mapping.csv"
SYMBOL_DIR = RESULTS_DIR / "top_genes_with_symbols"

ENSEMBL_REST = "https://rest.ensembl.org"
BATCH_SIZE = 1000  # Ensembl POST endpoint limit


def collect_gene_ids():
    """Gather all unique Ensembl IDs from the empirical loading matrix."""
    f = INTERP_DIR / "gene_loadings_empirical.csv"
    if not f.exists():
        print(f"ERROR: {f} not found. Run run_real_data.py first.")
        sys.exit(1)
    df = pd.read_csv(f, index_col=0, nrows=0)  # just read the index
    # The index is gene IDs — but the file has genes as rows
    df_full = pd.read_csv(f, index_col=0, usecols=[0])
    gene_ids = list(df_full.index)
    print(f"Collected {len(gene_ids)} unique Ensembl IDs")
    return gene_ids


def query_ensembl_batch(ids):
    """POST lookup for a batch of Ensembl IDs."""
    url = f"{ENSEMBL_REST}/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = json.dumps({"ids": ids})

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            time.sleep(3)
    return {}


def map_all_genes(gene_ids):
    """Query Ensembl in batches and build mapping DataFrame."""
    records = []
    n_batches = (len(gene_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(gene_ids), BATCH_SIZE):
        batch = gene_ids[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} IDs)...")

        result = query_ensembl_batch(batch)

        for gid in batch:
            info = result.get(gid)
            if info and isinstance(info, dict):
                records.append({
                    "ensembl_id": gid,
                    "gene_symbol": info.get("display_name", ""),
                    "description": info.get("description", ""),
                    "biotype": info.get("biotype", ""),
                    "seq_region": info.get("seq_region_name", ""),
                    "start": info.get("start", ""),
                    "end": info.get("end", ""),
                    "strand": info.get("strand", ""),
                })
            else:
                records.append({
                    "ensembl_id": gid,
                    "gene_symbol": "",
                    "description": "",
                    "biotype": "",
                    "seq_region": "",
                    "start": "",
                    "end": "",
                    "strand": "",
                })

        # Be polite to the API
        time.sleep(1)

    df = pd.DataFrame(records)
    return df


def add_symbols_to_top_genes(mapping_df):
    """Re-export top genes per dim with gene symbols added."""
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    id_to_sym = dict(zip(mapping_df["ensembl_id"], mapping_df["gene_symbol"]))

    for method in ["decoder", "encoder", "empirical"]:
        src = INTERP_DIR / f"top_genes_per_dim_{method}.csv"
        if not src.exists():
            continue

        df = pd.read_csv(src)
        df["gene_symbol"] = df["gene"].map(id_to_sym).fillna("")
        # Reorder columns
        cols = ["dimension", "gene", "gene_symbol", "loading", "abs_loading", "rank"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        out = SYMBOL_DIR / f"top_genes_{method}_with_symbols.csv"
        df.to_csv(out, index=False)
        print(f"  Saved {out}")

        # Summary: how many were mapped
        n_mapped = (df["gene_symbol"] != "").sum()
        print(f"    {method}: {n_mapped}/{len(df)} genes mapped to symbols "
              f"({100*n_mapped/len(df):.1f}%)")


def print_sample_modules(mapping_df):
    """Print the first 3 modules with gene symbols for quick inspection."""
    id_to_sym = dict(zip(mapping_df["ensembl_id"], mapping_df["gene_symbol"]))
    src = INTERP_DIR / "top_genes_per_dim_empirical.csv"
    if not src.exists():
        return

    df = pd.read_csv(src)
    df["symbol"] = df["gene"].map(id_to_sym).fillna("?")

    print("\n  Sample modules (empirical, top 10 genes):")
    for dim in sorted(df["dimension"].unique())[:5]:
        sub = df[df["dimension"] == dim].head(10)
        genes = [f"{r['symbol']}({r['loading']:.3f})" for _, r in sub.iterrows()]
        print(f"    {dim}: {', '.join(genes)}")


def main():
    print("=" * 60)
    print("  Gene ID Mapping: ENSSSCG* → Gene Symbols (Sus scrofa)")
    print("=" * 60)

    # Check if mapping already exists
    rerun = "--rerun" in sys.argv
    if MAPPING_FILE.exists() and not rerun:
        print(f"\n  Mapping file already exists: {MAPPING_FILE}")
        existing = pd.read_csv(MAPPING_FILE)
        n_mapped = (existing["gene_symbol"] != "").sum()
        print(f"  {n_mapped}/{len(existing)} IDs mapped ({100*n_mapped/len(existing):.1f}%)")
        print("  Using existing mapping. (pass --rerun to re-query Ensembl)")
        add_symbols_to_top_genes(existing)
        print_sample_modules(existing)
        return

    # 1. Collect gene IDs
    print("\n[1/3] Collecting gene IDs...")
    gene_ids = collect_gene_ids()

    # 2. Query Ensembl
    print("\n[2/3] Querying Ensembl REST API...")
    mapping_df = map_all_genes(gene_ids)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(MAPPING_FILE, index=False)
    n_mapped = (mapping_df["gene_symbol"] != "").sum()
    print(f"\n  Saved {MAPPING_FILE}")
    print(f"  {n_mapped}/{len(mapping_df)} IDs mapped to symbols "
          f"({100*n_mapped/len(mapping_df):.1f}%)")

    # 3. Add symbols to top genes
    print("\n[3/3] Adding symbols to top-gene files...")
    add_symbols_to_top_genes(mapping_df)
    print_sample_modules(mapping_df)

    print("\n" + "=" * 60)
    print("  DONE — Gene ID mapping complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
