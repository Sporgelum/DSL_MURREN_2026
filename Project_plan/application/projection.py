"""
Projection Tool — Digital Inference using the trained cVAE encoder.

For a new dataset, pass expression values through the trained Encoder to
obtain immediate Activity Scores for every module. This enables rapid
comparison across vaccines and conditions without re-training.

Protocol Section 5 — Functional Application (Projection Tool / Digital Inference).
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional

from model.cvae import ConditionalVAE
from data.data_loader import preprocess_expression, encode_conditions


def project_new_data(
    model: ConditionalVAE,
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    gene_names: List[str],
    scaler=None,
    log_transform: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Project new expression data through the trained encoder to get
    module activity scores.

    Args:
        model:         Trained ConditionalVAE
        expression_df: New expression data (samples x genes)
        metadata_df:   Metadata for the new samples
        gene_names:    Gene names (must match training order)
        scaler:        Fitted StandardScaler from training (or None)
        log_transform: Whether to log-transform (should match training)
        device:        'cuda' or 'cpu'

    Returns:
        activity_df: DataFrame of shape (n_samples, latent_dim)
                     with module activity scores per sample
    """
    model.eval()
    model.to(device)

    # Preprocess expression
    X = expression_df[gene_names].values.astype(np.float32)
    if log_transform:
        X = np.log1p(np.maximum(X, 0))
    if scaler is not None:
        X = scaler.transform(X)

    # Encode conditions
    conditions, _ = encode_conditions(metadata_df)

    # Convert to tensors
    x_tensor = torch.FloatTensor(X).to(device)
    c_tensor = torch.FloatTensor(conditions).to(device)

    # Encode (deterministic — use mean, not sampled z)
    with torch.no_grad():
        activity_scores = model.encode(x_tensor, c_tensor)

    activity_np = activity_scores.cpu().numpy()
    columns = [f"Module_{j:03d}" for j in range(activity_np.shape[1])]

    activity_df = pd.DataFrame(activity_np, columns=columns)

    # Attach sample identifiers if available
    if "sample_id" in expression_df.columns:
        activity_df.index = expression_df["sample_id"].values

    return activity_df


def compare_conditions(
    activity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str = "vaccine_type",
) -> pd.DataFrame:
    """
    Compare mean module activity across groups (e.g., vaccine types).

    Returns a DataFrame of mean activity per group per module.
    """
    activity_with_group = activity_df.copy()
    activity_with_group[group_col] = metadata_df[group_col].values
    return activity_with_group.groupby(group_col).mean()


def save_projections(
    activity_df: pd.DataFrame,
    output_dir: str,
    filename: str = "module_activity_scores.csv",
) -> str:
    """Save activity scores to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / filename
    activity_df.to_csv(fpath)
    print(f"Saved activity scores to {fpath}")
    return str(fpath)
