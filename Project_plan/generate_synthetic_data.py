"""
Generate synthetic RNA-seq data with known module structure for testing.

Creates realistic-looking bulk RNA-seq data where:
  - Genes are grouped into known ground-truth modules
  - Different vaccine types activate different module combinations
  - Time points shift expression (baseline vs. post-vaccination)
  - Multiple studies with slight batch effects

This allows end-to-end testing of the pipeline AND validation of
whether the cVAE can recover the planted modules.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(
    n_samples: int = 600,
    n_genes: int = 500,
    n_true_modules: int = 10,
    genes_per_module: int = 30,
    n_studies: int = 3,
    n_vaccine_types: int = 3,
    noise_level: float = 0.5,
    batch_effect_strength: float = 0.3,
    output_dir: str = "data/raw",
    seed: int = 42,
):
    """
    Generate synthetic bulk RNA-seq data with planted module structure.

    The data is saved as one CSV per study in output_dir/.

    Args:
        n_samples:            Total samples across all studies
        n_genes:              Number of genes (kept small for fast testing)
        n_true_modules:       Number of ground-truth gene modules
        genes_per_module:     Genes per module (some overlap allowed)
        n_studies:            Number of studies to simulate
        n_vaccine_types:      Number of vaccine types
        noise_level:          Gaussian noise std
        batch_effect_strength: Per-study batch offset strength
        output_dir:           Where to write CSVs
        seed:                 Random seed
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    vaccine_types = [f"vaccine_{v}" for v in ["viral", "bacterial", "mrna"][:n_vaccine_types]]
    time_points = ["Day0", "Day7"]
    study_ids = [f"Study_{s+1}" for s in range(n_studies)]

    # ── Define ground-truth modules ──
    # Each module is a random subset of genes
    modules = {}
    for m in range(n_true_modules):
        start = (m * genes_per_module) % n_genes
        indices = [(start + i) % n_genes for i in range(genes_per_module)]
        modules[m] = indices

    # Save ground truth for later validation
    gt_rows = []
    for m, gene_indices in modules.items():
        for gi in gene_indices:
            gt_rows.append({"module": m, "gene": gene_names[gi]})
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(out.parent / "ground_truth_modules.csv", index=False)
    print(f"Ground truth: {n_true_modules} modules, {genes_per_module} genes each")
    print(f"Saved to {out.parent / 'ground_truth_modules.csv'}")

    # ── Define which modules each vaccine activates ──
    # Each vaccine activates a distinct but overlapping set of modules
    vaccine_module_map = {}
    for v_idx, vtype in enumerate(vaccine_types):
        # Each vaccine activates ~60% of modules, shifted
        active = set()
        for m in range(n_true_modules):
            if rng.random() < 0.6 or m % n_vaccine_types == v_idx:
                active.add(m)
        vaccine_module_map[vtype] = active

    # ── Generate samples ──
    samples_per_study = n_samples // n_studies
    all_data = []

    for s_idx, study_id in enumerate(study_ids):
        # Batch effect for this study
        batch_offset = rng.normal(0, batch_effect_strength, size=n_genes)

        for i in range(samples_per_study):
            sample_id = f"{study_id}_S{i:03d}"
            vaccine = vaccine_types[rng.integers(0, n_vaccine_types)]
            time_point = time_points[rng.integers(0, 2)]

            # Base expression: low-level noise
            expression = rng.normal(5.0, noise_level, size=n_genes)

            # Activate modules based on vaccine type
            for m in vaccine_module_map[vaccine]:
                module_strength = rng.normal(3.0, 0.5)
                # Stronger activation at Day7 than Day0
                if time_point == "Day7":
                    module_strength *= rng.uniform(1.5, 2.5)
                else:
                    module_strength *= rng.uniform(0.2, 0.5)  # baseline: weak

                for gi in modules[m]:
                    expression[gi] += module_strength

            # Add batch effect
            expression += batch_offset

            # Ensure non-negative (count-like)
            expression = np.maximum(expression, 0)

            row = {
                "sample_id": sample_id,
                "study_id": study_id,
                "vaccine_type": vaccine,
                "time_point": time_point,
            }
            for g_idx, gname in enumerate(gene_names):
                row[gname] = round(float(expression[g_idx]), 2)

            all_data.append(row)

    # ── Split into per-study CSVs ──
    full_df = pd.DataFrame(all_data)
    for study_id in study_ids:
        study_df = full_df[full_df["study_id"] == study_id].reset_index(drop=True)
        fpath = out / f"{study_id}.csv"
        study_df.to_csv(fpath, index=False)
        print(f"  {fpath}: {len(study_df)} samples")

    print(f"\nTotal: {len(full_df)} samples, {n_genes} genes, {n_studies} studies")
    print(f"Vaccine types: {vaccine_types}")
    print(f"Files written to: {out}")

    return full_df, modules


if __name__ == "__main__":
    generate_synthetic_data()
