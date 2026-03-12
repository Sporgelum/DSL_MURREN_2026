"""
GMT file export for discovered Blood Transcription Modules.

Saves gene modules in the .gmt (Gene Matrix Transposed) format used by
standard GSEA tools (MSigDB, GSEA desktop, fgsea, etc.).

Protocol Section 5 — Functional Application (Repertoire / Pathways).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def modules_to_gmt(
    btm_modules: Dict[int, pd.DataFrame],
    output_path: str,
    module_annotations: Optional[Dict[int, str]] = None,
    description_prefix: str = "BTM_Module",
) -> str:
    """
    Write discovered modules to a .gmt file.

    GMT format (tab-separated, one line per module):
        MODULE_NAME <TAB> DESCRIPTION <TAB> GENE1 <TAB> GENE2 <TAB> ...

    Args:
        btm_modules: dict mapping module_index -> DataFrame with 'gene' column
        output_path: path to write the .gmt file
        module_annotations: optional dict mapping module_index -> functional label
        description_prefix: prefix for auto-generated module names

    Returns:
        output_path as string
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for j, module_df in sorted(btm_modules.items()):
        name = f"{description_prefix}_{j:03d}"

        if module_annotations and j in module_annotations:
            description = module_annotations[j]
        else:
            description = f"Latent_dim_{j}_n_genes_{len(module_df)}"

        genes = module_df["gene"].tolist()
        line = "\t".join([name, description] + genes)
        lines.append(line)

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} modules to {out}")
    return str(out)


def load_gmt(gmt_path: str) -> Dict[str, list]:
    """
    Read a .gmt file back into a dict of {module_name: [gene_list]}.
    """
    modules = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                name = parts[0]
                # parts[1] is description, parts[2:] are genes
                modules[name] = parts[2:]
    return modules
