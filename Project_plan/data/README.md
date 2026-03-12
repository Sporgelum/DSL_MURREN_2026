# Data Module — RNA-seq Data Loading & Preprocessing

## Goal
Load heterogeneous bulk RNA-seq datasets from multiple clinical studies, apply preprocessing transformations, and produce PyTorch DataLoaders ready for training the MI-Regularized cVAE.

## Main Idea
Vaccine-response studies generate expression matrices (samples × genes) alongside metadata (study ID, vaccine type, time point). This module unifies data from different studies into a single tensor pipeline. It handles the full journey from raw CSV files to batched, normalized, one-hot-conditioned tensors.

## Algorithm / Approach
1. **Loading**: Scan the `data/raw/` directory for CSV/TSV files per study. Concatenate all studies row-wise.
2. **Metadata separation**: Split columns into gene expression values and metadata (study_id, vaccine_type, time_point).
3. **Log-transform**: Apply `log1p` to raw counts to stabilize variance.
4. **Standard scaling**: Z-score normalize each gene across all samples (zero mean, unit variance).
5. **One-hot encoding**: Convert each categorical metadata column into a binary vector and concatenate all into a single condition vector per sample.
6. **Splitting**: Divide into train (80%), validation (10%), and test (10%) sets using a fixed random seed for reproducibility.

## Inputs
| Input | Description |
|---|---|
| `data/raw/*.csv` or `*.tsv` | Expression matrices with gene columns and metadata columns |
| `config.DataConfig` | Settings for column names, fractions, normalization flags |

## Outputs
| Output | Description |
|---|---|
| `train_loader` | PyTorch DataLoader for training batches |
| `val_loader` | PyTorch DataLoader for validation batches |
| `test_loader` | PyTorch DataLoader for test batches |
| `gene_names` | List of gene identifiers preserved for downstream extraction |
| `condition_dim` | Integer — total size of the concatenated one-hot condition vector |

## Key Components
- **`RNASeqDataset`** — Custom PyTorch Dataset pairing expression tensors with condition vectors.
- **`load_expression_data()`** — Reads and concatenates multi-file expression data.
- **`preprocess_expression()`** — Applies log1p + Z-score normalization.
- **`encode_conditions()`** — One-hot encodes categorical metadata.
- **`build_dataloaders()`** — Splits dataset and wraps in DataLoaders.
- **`prepare_data(cfg)`** — End-to-end convenience function driven by the config object.
