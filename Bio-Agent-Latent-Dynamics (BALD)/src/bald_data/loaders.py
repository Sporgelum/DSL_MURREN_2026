from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _detect_sep(file_path: Path) -> str:
    """Detect delimiter from the first line of a text table."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if "\t" in first:
        return "\t"
    if "," in first:
        return ","
    return "\t"


def parse_day_order(sample_timepoint: str) -> int:
    """Convert free-text sample timepoint into a sortable day-order integer.

    Mapping strategy:
    - baseline/pre/before -> 0
    - day 1 / d1 / 24h / 1d -> 1
    - day 7 / d7 / 168h / 7d -> 7
    - other hour-based values (xh) -> max(1, round(hours / 24))
    - unknown -> 99
    """
    text = (sample_timepoint or "").strip().lower()

    if text == "":
        return 99

    if any(k in text for k in ["baseline", "pre", "before", "d0", "day0", "day 0"]):
        return 0

    if any(k in text for k in ["day 1", "day1", "d1", "24h", "1d", "6h", "12h"]):
        return 1

    if any(k in text for k in ["day 7", "day7", "d7", "168h", "7d"]):
        return 7

    if "h" in text:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            hours = int(digits)
            return max(1, int(round(hours / 24.0)))

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)

    return 99


def build_day_labels(meta: pd.DataFrame, timepoint_col: str = "SampleTimepoint") -> pd.DataFrame:
    """Add day order and day label columns to metadata."""
    out = meta.copy()
    out["day_order"] = out[timepoint_col].astype(str).map(parse_day_order)
    out["day_label"] = out["day_order"].map(lambda d: f"Day{d}" if d != 99 else "DayUnknown")
    return out


def load_logcpm_and_metadata(
    counts_path: Path,
    metadata_path: Path,
    run_col: str = "Run",
    project_col: str = "BioProject",
    timepoint_col: str = "SampleTimepoint",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load and align logCPM + metadata.

    Returns:
        expr_samples_by_genes: DataFrame [samples, genes]
        aligned_meta: DataFrame [samples, ...] with day columns included
        gene_names: list[str]
    """
    counts_sep = _detect_sep(counts_path)
    meta_sep = _detect_sep(metadata_path)

    expr = pd.read_csv(counts_path, sep=counts_sep, index_col=0)
    meta = pd.read_csv(metadata_path, sep=meta_sep)

    if run_col not in meta.columns:
        raise ValueError(f"Required metadata column missing: {run_col}")
    if project_col not in meta.columns:
        raise ValueError(f"Required metadata column missing: {project_col}")
    if timepoint_col not in meta.columns:
        raise ValueError(f"Required metadata column missing: {timepoint_col}")

    meta = meta[meta[run_col].astype(str).isin(expr.columns)].copy()
    meta = meta.drop_duplicates(subset=run_col).set_index(run_col)

    common_runs = sorted(set(expr.columns) & set(meta.index.astype(str)))
    if not common_runs:
        raise ValueError("No overlap found between expression columns and metadata Run IDs")

    expr = expr[common_runs]
    meta = meta.loc[common_runs]

    expr_samples_by_genes = expr.T
    gene_names = [str(g) for g in expr.index.tolist()]

    meta = build_day_labels(meta, timepoint_col=timepoint_col)

    return expr_samples_by_genes, meta, gene_names
