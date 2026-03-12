"""Helper: add gene symbols to top-gene files and print sample modules."""
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results/real_data")
INTERP_DIR = RESULTS_DIR / "interpretation"
SYMBOL_DIR = RESULTS_DIR / "top_genes_with_symbols"
SYMBOL_DIR.mkdir(parents=True, exist_ok=True)

mapping = pd.read_csv(RESULTS_DIR / "gene_id_mapping.csv")
id_to_sym = dict(zip(mapping["ensembl_id"], mapping["gene_symbol"].fillna("")))

for method in ["decoder", "encoder", "empirical"]:
    src = INTERP_DIR / f"top_genes_per_dim_{method}.csv"
    if not src.exists():
        continue
    df = pd.read_csv(src)
    df["gene_symbol"] = df["gene"].map(id_to_sym).fillna("")
    cols = ["dimension", "gene", "gene_symbol", "loading", "abs_loading", "rank"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    out = SYMBOL_DIR / f"top_genes_{method}_with_symbols.csv"
    df.to_csv(out, index=False)
    n_mapped = (df["gene_symbol"] != "").sum()
    print(f"{method}: {n_mapped}/{len(df)} mapped ({100*n_mapped/len(df):.1f}%), saved {out.name}")

# Show sample modules (empirical)
print("\nSample modules (empirical, top 8 genes):")
df = pd.read_csv(SYMBOL_DIR / "top_genes_empirical_with_symbols.csv")
for dim in sorted(df["dimension"].unique())[:5]:
    sub = df[df["dimension"] == dim].head(8)
    genes = []
    for _, r in sub.iterrows():
        sym = r["gene_symbol"] if r["gene_symbol"] else r["gene"]
        genes.append(f"{sym}({r['loading']:.3f})")
    print(f"  {dim}: {', '.join(genes)}")
