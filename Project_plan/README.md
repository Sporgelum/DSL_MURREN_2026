# BTM Discovery Pipeline — MI-Regularized Conditional VAE

## Overview
This project implements a **Conditional Variational Autoencoder (cVAE)** regularized with **Mutual Information (MI) maximization** in PyTorch to discover *de novo* **Blood Transcription Modules (BTMs)** from bulk RNA-seq vaccine-response data.

Unlike static, pre-defined pathway databases, these modules are **dynamically derived** from actual immune challenge data, capturing coordinated gene programs that activate during vaccination.

## Project Structure

```
Project_plan/
│
├── main.py                        # Master orchestrator — runs the full pipeline
├── config.py                      # All hyperparameters and paths
├── project_plan_readme.md         # Original protocol specification
├── README.md                      # This file
│
├── data/                          # Data loading & preprocessing
│   ├── README.md                  # Module documentation
│   ├── data_loader.py             # CSV loading, normalization, one-hot encoding, DataLoaders
│   └── __init__.py
│
├── model/                         # Neural network architecture
│   ├── README.md                  # Module documentation
│   ├── cvae.py                    # Encoder, Decoder, ConditionalVAE
│   ├── mi_regularizer.py          # MINE / NWJ mutual information estimators
│   ├── losses.py                  # Reconstruction, KL divergence, total loss
│   └── __init__.py
│
├── training/                      # Training loop
│   ├── README.md                  # Module documentation
│   ├── trainer.py                 # Epoch loop, validation, early stopping, checkpointing
│   └── __init__.py
│
├── extraction/                    # Post-training module extraction
│   ├── README.md                  # Module documentation
│   ├── weight_extractor.py        # Decoder weight matrix → gene rankings → BTM modules
│   └── __init__.py
│
├── application/                   # Downstream biological applications
│   ├── README.md                  # Module documentation
│   ├── gmt_export.py              # Export modules as .gmt for GSEA
│   ├── annotation.py              # Cross-reference with MSigDB pathways
│   ├── projection.py              # Encode new data → module activity scores
│   └── __init__.py
│
└── utils/                         # Shared utilities
    ├── README.md                  # Module documentation
    ├── utils.py                   # Seed, device, plotting, parameter counting
    └── __init__.py
```

## Pipeline Stages

### Stage 1 — Data Loading (`data/`)
Load heterogeneous bulk RNA-seq CSVs from multiple studies. Apply log1p transform and Z-score normalization. One-hot encode metadata (study_id, vaccine_type, time_point). Split into train/val/test DataLoaders.

### Stage 2 — Model Construction (`model/`)
Build the cVAE: Encoder compresses ~20,000 genes + condition into a 128-dim latent space. Decoder reconstructs expression from latent code + condition. A separate MI estimator network (MINE/NWJ) prevents latent collapse.

### Stage 3 — Training (`training/`)
Alternating optimization: update MI estimator, then update cVAE on combined loss:

$$\mathcal{L} = \|X - \hat{X}\|^2 + \beta \cdot D_{KL}(q(Z|X) \| p(Z)) - \lambda \cdot I(X; Z)$$

Features: KL annealing, cosine LR scheduling, gradient clipping, early stopping, checkpointing.

### Stage 4 — Module Extraction (`extraction/`)
Extract the decoder weight matrix. For each latent dimension (module): rank genes by absolute weight, apply Z-score thresholding ($>2.5\sigma$), enforce size constraints (10–500 genes).

### Stage 5 — Application (`application/`)
- **GMT Export**: Save modules as `.gmt` for GSEA tools
- **Annotation**: Cross-reference with MSigDB (Jaccard similarity) to label modules (e.g., "Module 5 = Early Interferon Response")
- **Projection**: Pass new data through the encoder for instant module activity scores

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn umap-learn

# 2. Place RNA-seq data in data/raw/ (CSV format with metadata columns)

# 3. (Optional) Place reference GMT in data/reference/msigdb_immunologic.gmt

# 4. Run the pipeline
python main.py
```

## Configuration
All hyperparameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `latent_dim` | 128 | Number of modules (latent dimensions) |
| `encoder_hidden_dims` | [2048, 512, 256] | Encoder layer sizes |
| `mi_weight` | 0.1 | Lambda for MI regularization |
| `kl_anneal_epochs` | 30 | Epochs to linearly ramp KL weight |
| `zscore_threshold` | 2.5 | Sigma threshold for gene selection |
| `epochs` | 200 | Maximum training epochs |
| `early_stopping_patience` | 15 | Patience for early stopping |

## Outputs
| File | Description |
|---|---|
| `results/btm_modules.gmt` | Discovered gene modules in GMT format |
| `results/annotations/module_annotations.csv` | Functional labels per module |
| `results/projections/module_activity_scores.csv` | Per-sample module scores |
| `checkpoints/best_checkpoint.pt` | Best trained model weights |
| `logs/training_history.json` | Per-epoch training metrics |
| `logs/training_curves.png` | Loss curve visualizations |

## Theoretical Background
See [project_plan_readme.md](project_plan_readme.md) for the full protocol covering data description, objectives, architecture rationale, and biological application.



##### Project Structure
| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |


|Directory	| Files	| Purpose
|Root |	main.py, config.py, README.md, config_README.md	Orchestrator, configuration, docs
data/	data_loader.py, README.md	CSV loading, normalization, one-hot encoding, DataLoaders
data/raw/	README.md	Placeholder for input RNA-seq CSVs
data/reference/	README.md	Placeholder for MSigDB reference .gmt files
model/	cvae.py, mi_regularizer.py, losses.py, README.md	Encoder/Decoder, MINE/NWJ MI estimation, loss functions
training/	trainer.py, README.md	Training loop with KL annealing, early stopping, checkpointing
extraction/	weight_extractor.py, README.md	Decoder weight extraction, gene ranking, Z-score thresholding
application/	gmt_export.py, annotation.py, projection.py, README.md	GMT export, GSEA annotation, digital inference
utils/	utils.py, README.md	Seed control, plotting, latent space visualization
results/	README.md	Output directory for modules, annotations, projections

##### Protocol coverage
Section 1 (Data) → data/data_loader.py — multi-study loading, log-transform, Z-score
Section 2 (Objective) → architecture designed to discover de novo BTMs
Section 3 (Implementation) → model/cvae.py (conditioning + bottleneck), model/mi_regularizer.py (MINE/NWJ), model/losses.py (combined loss)
Section 4 (Extraction) → extraction/weight_extractor.py — effective weight matrix, gene ranking, Z-score selection
Section 5 (Application) → application/gmt_export.py (pathway repertoire), application/projection.py (digital inference), application/annotation.py (functional labeling)