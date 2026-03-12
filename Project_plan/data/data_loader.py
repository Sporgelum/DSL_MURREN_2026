"""
Data loading and preprocessing for bulk RNA-seq datasets.

Handles:
  - Loading heterogeneous multi-study CSV/TSV expression matrices
  - Metadata parsing (study ID, vaccine type, time point)
  - Log-transformation and normalization
  - One-hot encoding of condition labels
  - Train/val/test splitting
  - PyTorch Dataset and DataLoader creation
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class RNASeqDataset(Dataset):
    """PyTorch Dataset wrapping gene expression + condition vectors."""

    def __init__(
        self,
        expression: np.ndarray,
        conditions: np.ndarray,
        gene_names: List[str],
        sample_ids: List[str],
    ):
        self.expression = torch.FloatTensor(expression)
        self.conditions = torch.FloatTensor(conditions)
        self.gene_names = gene_names
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return self.expression.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.expression[idx], self.conditions[idx]


def load_expression_data(
    data_dir: str,
    metadata_cols: List[str],
    gene_prefix: str = "gene_",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load all CSV/TSV files from data_dir and concatenate them.

    Returns:
        expression_df: DataFrame of shape (n_samples, n_genes)
        metadata_df:   DataFrame of shape (n_samples, n_metadata_cols)
        gene_names:    List of gene identifiers
    """
    data_path = Path(data_dir)
    frames = []

    for fpath in sorted(data_path.glob("*.csv")):
        df = pd.read_csv(fpath)
        frames.append(df)

    for fpath in sorted(data_path.glob("*.tsv")):
        df = pd.read_csv(fpath, sep="\t")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV/TSV files found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Separate metadata and expression columns
    metadata_df = combined[metadata_cols].copy()
    gene_cols = [c for c in combined.columns if c not in metadata_cols and c != "sample_id"]
    expression_df = combined[gene_cols].copy()
    gene_names = list(gene_cols)

    sample_ids = (
        combined["sample_id"].tolist()
        if "sample_id" in combined.columns
        else [f"sample_{i}" for i in range(len(combined))]
    )

    return expression_df, metadata_df, gene_names, sample_ids


def preprocess_expression(
    expression_df: pd.DataFrame,
    log_transform: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Apply log1p transform and/or standard scaling to expression data.

    Returns:
        expression_array: numpy array (n_samples, n_genes)
        scaler:           fitted StandardScaler (or None)
    """
    X = expression_df.values.astype(np.float32)

    if log_transform:
        X = np.log1p(np.maximum(X, 0))

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, scaler


def encode_conditions(
    metadata_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, LabelEncoder]]:
    """
    One-hot encode all metadata columns and concatenate.

    Returns:
        condition_matrix: numpy array (n_samples, total_one_hot_dim)
        encoders:         dict mapping column name to its LabelEncoder
    """
    encoded_parts = []
    encoders = {}

    for col in metadata_df.columns:
        le = LabelEncoder()
        integer_encoded = le.fit_transform(metadata_df[col].astype(str))
        n_classes = len(le.classes_)
        one_hot = np.zeros((len(integer_encoded), n_classes), dtype=np.float32)
        one_hot[np.arange(len(integer_encoded)), integer_encoded] = 1.0
        encoded_parts.append(one_hot)
        encoders[col] = le

    condition_matrix = np.hstack(encoded_parts)
    return condition_matrix, encoders


def build_dataloaders(
    expression: np.ndarray,
    conditions: np.ndarray,
    gene_names: List[str],
    sample_ids: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data into train/val/test and return DataLoaders.
    """
    dataset = RNASeqDataset(expression, conditions, gene_names, sample_ids)
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_data(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int]:
    """
    End-to-end data preparation from config.

    Returns:
        train_loader, val_loader, test_loader, gene_names, condition_dim
    """
    expression_df, metadata_df, gene_names, sample_ids = load_expression_data(
        data_dir=cfg.data.data_dir,
        metadata_cols=cfg.data.metadata_cols,
        gene_prefix=cfg.data.gene_expression_prefix,
    )

    expression, scaler = preprocess_expression(
        expression_df,
        log_transform=cfg.data.log_transform,
        normalize=cfg.data.normalize,
    )

    conditions, encoders = encode_conditions(metadata_df)
    condition_dim = conditions.shape[1]

    train_loader, val_loader, test_loader = build_dataloaders(
        expression=expression,
        conditions=conditions,
        gene_names=gene_names,
        sample_ids=sample_ids,
        train_frac=cfg.data.train_fraction,
        val_frac=cfg.data.val_fraction,
        batch_size=cfg.training.batch_size,
        seed=cfg.data.random_seed,
    )

    return train_loader, val_loader, test_loader, gene_names, condition_dim
