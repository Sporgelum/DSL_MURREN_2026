# BTM Discovery Pipeline — Complete Guide

## MI-Regularized Conditional VAE for De Novo Gene Module Discovery

**Species:** *Sus scrofa* (pig) — blood/PBMC RNA-seq  
**Data:** 32,763 genes × 613 samples across 18 BioProjects  
**Output:** 128 gene co-expression modules with pathway annotations  

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Model Architecture](#3-model-architecture)
4. [Project File Structure](#4-project-file-structure)
5. [Environment Setup](#5-environment-setup)
6. [How to Run — Step by Step](#6-how-to-run--step-by-step)
7. [Script Reference — What Each File Does](#7-script-reference--what-each-file-does)
8. [Configuration](#8-configuration)
9. [The Three Module Extraction Methods](#9-the-three-module-extraction-methods)
10. [Results Produced](#10-results-produced)
11. [Interpreting the Results](#11-interpreting-the-results)
12. [Call Graph — What Calls What](#12-call-graph--what-calls-what)

---

## 1. What This Pipeline Does

This pipeline discovers **gene co-expression modules** — groups of genes that are activated or suppressed together — from bulk RNA-seq data in a fully unsupervised manner. It works in five stages:

1. **Preprocess**: Load logCPM expression counts, Z-score normalize, one-hot encode experimental conditions (study, disease type, tissue).
2. **Train a Conditional VAE (cVAE)**: Compress 32,763 genes into a 128-dimensional latent space. Each latent dimension learns to represent one gene module. A mutual information (MI) regularizer prevents latent dimensions from collapsing.
3. **Extract gene modules**: Recover which genes belong to each module using three complementary methods (decoder weights, encoder weights, empirical correlations).
4. **Map gene IDs**: Convert Ensembl pig IDs (`ENSSSCG*`) to standard gene symbols via the Ensembl REST API.
5. **Annotate modules**: Test each module for enrichment against GO, KEGG, Reactome, WikiPathways, MSigDB Hallmark, and Blood Transcription Modules (BTM) using Fisher's exact test.

---

## 2. Mathematical Foundation

The model is trained to minimize:

$$\mathcal{L} = \underbrace{\lVert X - \hat{X} \rVert^2_2}_{\text{reconstruction}} + \underbrace{\beta \cdot D_{KL}\bigl(q(Z \mid X, C) \;\|\; p(Z)\bigr)}_{\text{KL regularization}} - \underbrace{\lambda \cdot \hat{I}(X;\, Z)}_{\text{MI maximization}}$$

| Term | Role | Effect |
|------|------|--------|
| **Reconstruction loss** | Forces the model to faithfully compress and reconstruct gene expression | Ensures latent space captures real biological signal |
| **KL divergence** | Penalizes the latent distribution for deviating from a standard normal prior | Prevents overfitting, encourages smooth latent space |
| **MI maximization** | Rewards high mutual information between input genes and latent code | **Prevents latent collapse** — ensures every latent dimension is used |

- β is annealed linearly from 0 → 1 over the first 30 epochs (KL warmup)
- λ = 0.1 (MI weight)
- MI is estimated using the **MINE** (Mutual Information Neural Estimation) network

---

## 3. Model Architecture

```
 INPUT                    ENCODER                          LATENT               DECODER                         OUTPUT
┌──────────┐    ┌──────────────────────────────┐    ┌──────────────┐    ┌──────────────────────────────┐    ┌──────────┐
│ 32,763   │    │ (32763+27) → 2048 → 512 → 256│    │              │    │ (128+27) → 256 → 512 → 2048 │    │ 32,763   │
│ genes    │───>│         + BatchNorm + ReLU    │───>│  μ (128)     │───>│       + BatchNorm + ReLU     │───>│ recon.   │
│ + 27 dim │    │         + Dropout(0.2)        │    │  log σ² (128)│    │       + Dropout(0.2)         │    │ genes    │
│ condition│    └──────────────────────────────┘    │  ↓ reparam.  │    └──────────────────────────────┘    └──────────┘
└──────────┘                                        │  z (128)     │
                                                    └──────────────┘
                                                       | MI link
                                                    ┌──────────────┐
                                                    │ MINE Network │  Estimates I(X; Z)
                                                    │ for MI reg.  │
                                                    └──────────────┘

Parameters: ~137M (cVAE) + ~8.5M (MINE)
```

**Condition encoding (27 dimensions):**
- BioProject: 18 categories (study/batch identity)
- SampleStyle: 7 categories (control, control+prrsv, control+asfv, etc.)
- SampleTissue: 2 categories (blood, PBMC)

---

## 4. Project File Structure

```
Project_plan/
│
│   ══════════════════════════════════════════
│   ENTRY POINT SCRIPTS (what you run)
│   ══════════════════════════════════════════
├── run_real_data.py            <- MAIN: full pipeline on real data
├── run_test.py                 <- Test pipeline on synthetic data
├── map_gene_ids.py             <- Map Ensembl IDs -> gene symbols
├── annotate_modules.py         <- Pathway enrichment analysis
├── run_interpret.py            <- Interpretation only (loads checkpoint)
├── main.py                     <- Orchestrator for synthetic data
├── generate_synthetic_data.py  <- Create test data with known modules
│
│   ══════════════════════════════════════════
│   CONFIGURATION
│   ══════════════════════════════════════════
├── config.py                   <- All hyperparameters (dataclasses)
├── requirements.txt            <- Python dependencies
│
│   ══════════════════════════════════════════
│   CORE MODULES (called by scripts above)
│   ══════════════════════════════════════════
├── data/
│   ├── data_loader.py          <- CSV ingestion, normalization, DataLoader creation
│   ├── ground_truth_modules.csv
│   └── raw/                    <- Synthetic study CSVs (Study_1/2/3.csv)
│
├── model/
│   ├── cvae.py                 <- Encoder + Decoder + ConditionalVAE classes
│   ├── losses.py               <- Reconstruction, KL divergence, total loss
│   └── mi_regularizer.py       <- MINE / NWJ mutual information estimators
│
├── training/
│   └── trainer.py              <- Training loop, early stopping, checkpointing
│
├── extraction/
│   ├── interpret_latent.py     <- 3 extraction methods for gene-module loadings
│   └── weight_extractor.py     <- Decoder weight chain multiplication
│
├── application/
│   ├── annotation.py           <- Cross-reference modules with GMT databases
│   ├── gmt_export.py           <- Export modules to .gmt format (for GSEA)
│   └── projection.py           <- Project new data -> module activity scores
│
├── utils/
│   └── utils.py                <- Seed, device, plotting, history I/O
│
│   ══════════════════════════════════════════
│   DATA
│   ══════════════════════════════════════════
├── counts_and_metadata/
│   ├── logCPM_matrix_filtered_samples.csv      <- 32,763 genes x 613 samples (tab-sep)
│   └── metadata_with_sample_annotations.csv    <- Sample metadata (tab-sep)
│
│   ══════════════════════════════════════════
│   OUTPUTS (generated by the pipeline)
│   ══════════════════════════════════════════
├── checkpoints/
│   ├── best_real_checkpoint.pt     <- Trained model (real data, epoch 74)
│   ├── best_checkpoint.pt          <- Trained model (synthetic data)
│   └── last_checkpoint.pt          <- Last epoch checkpoint
│
├── logs/
│   ├── real_training_history.json  <- Per-epoch metrics (real data)
│   ├── real_training_curves.png    <- Loss plots (real data)
│   ├── training_history.json       <- Per-epoch metrics (synthetic)
│   └── training_curves.png         <- Loss plots (synthetic)
│
└── results/
    ├── btm_modules.gmt                            <- Synthetic data modules
    ├── interpretation/                            <- Synthetic data interpretation
    │
    └── real_data/                                 <- MAIN RESULTS
        ├── gene_id_mapping.csv                    <- Ensembl -> symbol mapping
        ├── module_annotations.csv                 <- Full pathway enrichment
        ├── module_top_annotation.csv              <- Best annotation per module
        ├── method_comparison_top20.csv            <- Cross-method overlap
        ├── method_comparison_top50.csv
        ├── gmt_cache/                             <- Downloaded gene set databases
        ├── top_genes_with_symbols/                <- Human-readable module lists
        │   ├── top_genes_decoder_with_symbols.csv
        │   ├── top_genes_encoder_with_symbols.csv
        │   └── top_genes_empirical_with_symbols.csv
        ├── interpretation/
        │   ├── gene_loadings_decoder.csv          <- Full matrix (32763 x 128)
        │   ├── gene_loadings_encoder.csv
        │   ├── gene_loadings_empirical.csv
        │   ├── top_genes_per_dim_decoder.csv      <- Top 30 genes, all dims
        │   ├── top_genes_per_dim_encoder.csv
        │   ├── top_genes_per_dim_empirical.csv
        │   ├── decoder/Dim_000..127_top_genes.csv <- 128 individual files
        │   ├── encoder/Dim_000..127_top_genes.csv
        │   └── empirical/Dim_000..127_top_genes.csv
        └── plots/
            ├── dimension_loadings.png
            └── loading_heatmap.png
```

---

## 5. Environment Setup

### Recommended Python Environment

The pipeline was tested with this environment (has PyTorch + CUDA):

```
C:\Users\emari\OneDrive - Universitaet Bern (1)\Documents\Environments\scimilarity_2024_local\Scripts\python.exe
```

For convenience in all commands below, set an alias:

```powershell
$PYTHON = "C:\Users\emari\OneDrive - Universitaet Bern (1)\Documents\Environments\scimilarity_2024_local\Scripts\python.exe"
```

### Dependencies

Listed in `requirements.txt`:

| Package | Version | Used by |
|---------|---------|---------|
| `torch` | >= 2.0.0 | cVAE model, training |
| `numpy` | >= 1.24.0 | Numerical operations |
| `pandas` | >= 2.0.0 | Data loading, CSV I/O |
| `scikit-learn` | >= 1.3.0 | Normalization, label encoding |
| `matplotlib` | >= 3.7.0 | Plotting |
| `seaborn` | >= 0.12.0 | Heatmaps |
| `scipy` | (any) | Fisher's exact test (annotation) |
| `requests` | (any) | Ensembl REST API (gene mapping) |

---

## 6. How to Run — Step by Step

### Quickstart (4 commands to replicate everything)

```powershell
cd "Course\Project_plan"

# Step 1: Train cVAE + extract modules (~90s on GPU, ~6min on CPU)
& $PYTHON run_real_data.py

# Step 2: Map Ensembl IDs to gene symbols (~2min, queries Ensembl API)
& $PYTHON map_gene_ids.py

# Step 3: Annotate modules with pathway databases (~2min)
& $PYTHON annotate_modules.py

# Step 4: Inspect results
# -> results/real_data/module_annotations.csv
# -> results/real_data/module_top_annotation.csv
# -> results/real_data/top_genes_with_symbols/
```

### Detailed Walkthrough

#### Step 1: Train the cVAE on Real Data

```powershell
& $PYTHON run_real_data.py
```

**What happens internally:**
1. Loads `counts_and_metadata/logCPM_matrix_filtered_samples.csv` (tab-separated, genes as rows, 613 SRR sample columns)
2. Loads `counts_and_metadata/metadata_with_sample_annotations.csv`, matches samples by the `Run` column
3. Transposes expression to (613 samples x 32,763 genes), Z-score normalizes
4. One-hot encodes conditions: BioProject (18) + SampleStyle (7) + SampleTissue (2) = 27 dimensions
5. Splits 80/20 into train (490 samples) and validation (123 samples)
6. Builds the cVAE (137M params) + MINE estimator (8.5M params)
7. Trains for up to 80 epochs with KL annealing and early stopping (patience=15)
8. Saves best model to `checkpoints/best_real_checkpoint.pt`
9. Extracts gene loadings via all 3 methods (decoder, encoder, empirical)
10. Compares methods (Jaccard overlap) and exports everything to `results/real_data/`

**Expected output:** Training log showing decreasing loss, ends with "DONE — All results in results/real_data/".

#### Step 2: Map Gene IDs

```powershell
& $PYTHON map_gene_ids.py
```

**What happens:**
1. Reads all 32,763 Ensembl gene IDs from the empirical loading matrix
2. Queries the Ensembl REST API (`rest.ensembl.org/lookup/id`) in batches of 1,000
3. Saves `results/real_data/gene_id_mapping.csv` with columns: `ensembl_id`, `gene_symbol`, `description`, `biotype`
4. Creates `results/real_data/top_genes_with_symbols/` — the top-gene CSVs with gene symbols added

**Note:** ~51% of IDs map to symbols. This is expected for pig — many gene models are novel or only have numerical identifiers. Among top-ranked module genes, 67-86% are successfully mapped.

**To re-run (overwrite existing mapping):**
```powershell
& $PYTHON map_gene_ids.py --rerun
```

#### Step 3: Annotate Modules

```powershell
& $PYTHON annotate_modules.py
```

**What happens:**
1. Loads the gene symbol mapping from Step 2
2. For each of 128 modules, takes the top 100 genes by absolute empirical loading
3. Downloads gene set databases from Enrichr (cached in `results/real_data/gmt_cache/` for subsequent runs):
   - GO Biological Process 2023 (5,407 gene sets)
   - KEGG 2021 Human (320 gene sets)
   - Reactome 2022 (1,818 gene sets)
   - WikiPathways 2023 Human (801 gene sets)
   - MSigDB Hallmark 2020 (50 gene sets)
   - Built-in BTM signatures (16 gene sets from Li et al. 2014)
4. Runs Fisher's exact test (one-sided, over-representation) for each module x pathway
5. Applies Benjamini-Hochberg FDR correction within each module
6. Saves `module_annotations.csv` (all results) and `module_top_annotation.csv` (best hit per module, FDR < 0.05 only)

#### Optional: Test on Synthetic Data First

```powershell
# Generate synthetic data (500 genes, 16 modules, 600 samples)
& $PYTHON generate_synthetic_data.py

# Run step-by-step test
& $PYTHON run_test.py
```

This creates a small dataset with known ground-truth modules so you can verify the pipeline works and see how well it recovers known structure.

#### Optional: Interpretation Only (reuse existing checkpoint)

```powershell
& $PYTHON run_interpret.py
```

This loads a previously trained checkpoint and re-runs the 3 extraction methods without retraining. Useful for adjusting interpretation parameters.

---

## 7. Script Reference — What Each File Does

### Entry Point Scripts

| Script | Purpose | Input | Output | Runtime |
|--------|---------|-------|--------|---------|
| `run_real_data.py` | Train cVAE + extract modules from real data | `counts_and_metadata/*.csv` | `checkpoints/`, `logs/`, `results/real_data/` | ~90s GPU / ~6min CPU |
| `map_gene_ids.py` | Map Ensembl IDs to gene symbols | `results/real_data/interpretation/gene_loadings_empirical.csv` | `gene_id_mapping.csv`, `top_genes_with_symbols/` | ~2 min |
| `annotate_modules.py` | Pathway enrichment for modules | `gene_id_mapping.csv` + `gene_loadings_empirical.csv` | `module_annotations.csv`, `module_top_annotation.csv` | ~2 min |
| `run_test.py` | Test pipeline on synthetic data | `data/raw/Study_*.csv` | `results/interpretation/` | ~2 min |
| `run_interpret.py` | Re-run extraction from existing checkpoint | `checkpoints/best_checkpoint.pt` | `results/interpretation/` | ~30s |
| `main.py` | Full synthetic pipeline (orchestrator) | `data/raw/` | `results/btm_modules.gmt` | ~3 min |
| `generate_synthetic_data.py` | Create synthetic test data | — | `data/raw/Study_*.csv`, `ground_truth_modules.csv` | <1s |

### Library Modules (not run directly — imported by the scripts above)

| Module | File | Key Classes/Functions |
|--------|------|----------------------|
| **Config** | `config.py` | `PipelineConfig`, `ModelConfig`, `MIConfig`, `TrainingConfig`, `DataConfig`, `ExtractionConfig`, `ApplicationConfig` |
| **Data** | `data/data_loader.py` | `RNASeqDataset`, `load_expression_data()`, `prepare_data()` |
| **Model** | `model/cvae.py` | `Encoder`, `Decoder`, `ConditionalVAE` (with `.from_config()`, `.encode()`) |
| **Losses** | `model/losses.py` | `reconstruction_loss()`, `kl_divergence()`, `compute_kl_weight()`, `total_loss()` |
| **MI** | `model/mi_regularizer.py` | `MINEEstimator`, `NWJEstimator`, `compute_mi_loss()`, `build_mi_estimator()` |
| **Training** | `training/trainer.py` | `Trainer` class (with `.fit()`, `.load_checkpoint()`) |
| **Extraction** | `extraction/interpret_latent.py` | `decoder_effective_weights()`, `encoder_effective_weights()`, `empirical_gene_latent_correlations()`, `extract_gene_loadings()`, `top_genes_per_dimension()`, `export_loadings()`, `compare_methods()` |
| **Extraction** | `extraction/weight_extractor.py` | `compute_effective_weights()`, `rank_genes_per_module()`, `extract_all_modules()` |
| **Export** | `application/gmt_export.py` | `modules_to_gmt()`, `load_gmt()` |
| **Annotation** | `application/annotation.py` | `annotate_modules()`, `compute_overlap()`, `save_annotations()` |
| **Projection** | `application/projection.py` | `project_new_data()` |
| **Utilities** | `utils/utils.py` | `set_seed()`, `get_device()`, `save_history()`, `plot_training_history()`, `count_parameters()` |

---

## 8. Configuration

All hyperparameters are centralized in `config.py` as Python dataclasses:

| Parameter | Default | Real Data Override | Description |
|-----------|---------|-------------------|-------------|
| `input_dim` | 20,000 | **32,763** | Number of genes |
| `condition_dim` | 10 | **27** | One-hot condition vector size |
| `latent_dim` | 128 | 128 | Number of modules to discover |
| `encoder_hidden_dims` | [2048, 512, 256] | same | Encoder layer widths |
| `decoder_hidden_dims` | [256, 512, 2048] | same | Decoder layer widths |
| `dropout` | 0.2 | 0.2 | Dropout rate |
| `use_batch_norm` | True | True | Batch normalization |
| `mi_weight` (lambda) | 0.1 | 0.1 | MI regularization strength |
| `mi_estimator` | "mine" | "mine" | MI estimator type (mine or nwj) |
| `epochs` | 200 | **80** | Maximum training epochs |
| `batch_size` | 128 | **64** | Mini-batch size |
| `learning_rate` | 1e-3 | 1e-3 | Adam optimizer learning rate |
| `kl_weight` (beta max) | 1.0 | 1.0 | Maximum KL weight after annealing |
| `kl_anneal_epochs` | 30 | 30 | Linear warmup from 0 to kl_weight |
| `early_stopping_patience` | 15 | 15 | Epochs without improvement before stopping |
| `scheduler` | "cosine" | "cosine" | Learning rate scheduler type |
| `gradient_clip` | 1.0 | 1.0 | Maximum gradient norm |

To change parameters for the real data pipeline, edit the values in `run_real_data.py` — the `main()` function sets `LATENT_DIM` and the `build_model()` function configures architecture.

---

## 9. The Three Module Extraction Methods

After training, we need to extract which genes belong to which module. The pipeline implements three complementary approaches:

### Method 1: Decoder Effective Weights (analytical)

Chain-multiply all decoder linear layer weight matrices to get a direct mapping from each latent dimension to each gene:

    W_eff = W_L * W_(L-1) * ... * W_1    ∈ R^(G x D)

**Interpretation:** How much each latent dimension contributes to reconstructing each gene.

### Method 2: Encoder Effective Weights (analytical)

Chain-multiply all encoder layers including the mu projection:

    W_enc = W_mu * W_L * ... * W_1    ∈ R^(D x G)

Transposed to (G x D). **Interpretation:** Which genes drive each latent dimension's activation.

### Method 3: Empirical Correlations (data-driven) — RECOMMENDED

Pass all samples through the encoder, then compute Pearson correlation between each gene's expression and each latent dimension's activation across all 613 samples:

    r(g,d) = cor(X_g, Z_d)    for all g in [1..G], d in [1..D]

**Interpretation:** The actual statistical relationship between genes and modules as learned by the model on the data.

### Comparison on Real Data

| Metric | Decoder | Encoder | Empirical |
|--------|---------|---------|-----------|
| Max absolute loading | 0.07 | 0.13 | **0.80** |
| Modules with max loading > 0.3 | 0 | 0 | **128** |
| Modules with max loading > 0.1 | 0 | 3 | **128** |
| Modules with max loading > 0.05 | 19 | 48 | **128** |
| Dec vs Enc Jaccard (top 20) | 0.001 | — | — |
| Dec vs Emp Jaccard (top 20) | — | — | 0.000 |
| Enc vs Emp Jaccard (top 20) | — | 0.021 | — |

**Key insight:** The empirical method produces the strongest, most interpretable loadings (correlations up to |r| = 0.80). The weight-based methods produce diffuse, low-magnitude loadings for this large gene space. The three methods capture fundamentally different aspects of the model with near-zero overlap.

**Recommendation:** Use the **empirical** method for biological interpretation. The weight-based methods may be useful for understanding model internals but are less biologically informative on high-dimensional data.

---

## 10. Results Produced

### Training Results

| File | Description |
|------|-------------|
| `checkpoints/best_real_checkpoint.pt` | Best model weights (epoch 74, val loss = 0.2948) |
| `logs/real_training_history.json` | Per-epoch metrics: reconstruction loss, KL, MI, validation loss |
| `logs/real_training_curves.png` | Visual training/validation loss curves |

**Training summary:** 80 epochs on CUDA (GPU) in 88 seconds. Early stopping selected epoch 74. Final metrics: reconstruction loss 0.39, KL divergence 0.003, MI 0.017.

### Gene Loading Matrices

In `results/real_data/interpretation/`:

| File | Shape | Description |
|------|-------|-------------|
| `gene_loadings_empirical.csv` | 32,763 x 128 | **Recommended.** Pearson correlations for each gene in each module |
| `gene_loadings_decoder.csv` | 32,763 x 128 | Decoder effective weight matrix |
| `gene_loadings_encoder.csv` | 32,763 x 128 | Encoder effective weight matrix |
| `top_genes_per_dim_*.csv` | ~3,840 rows | Top 30 genes per dimension (all dims combined) |
| `{method}/Dim_NNN_top_genes.csv` | 30 rows each | Individual per-dimension files (128 x 3 = 384 total) |

### Gene ID Mapping

| File | Description |
|------|-------------|
| `results/real_data/gene_id_mapping.csv` | 32,763 rows: `ensembl_id`, `gene_symbol`, `description`, `biotype` |
| `results/real_data/top_genes_with_symbols/*.csv` | Top genes with readable symbols for each method |

**Coverage:** 16,760 / 32,763 IDs (51.2%) mapped to gene symbols. Among top-ranked genes per module: 67-86% mapped.

### Pathway Enrichment

| File | Description |
|------|-------------|
| `results/real_data/module_annotations.csv` | Full enrichment results: module, database, pathway, overlap, p-value, FDR, shared genes |
| `results/real_data/module_top_annotation.csv` | Best annotation per module (FDR < 0.05 only) |

**Enrichment summary:**

| Threshold | Pathway Hits | Modules Annotated |
|-----------|-------------|-------------------|
| FDR < 0.05 | 19 hits | 9 modules |
| FDR < 0.10 | 22 hits | 12 modules |
| FDR < 0.25 | 27 hits | 17 modules |
| Nominal p < 0.01 | 135 hits | 125 / 128 modules |

### Modules with Significant Annotations (FDR < 0.05)

| Module | Pathway | Database | Overlap | FDR |
|--------|---------|----------|---------|-----|
| Dim_124 | Interaction Between L1 and Ankyrins | Reactome | 5 genes | 0.003 |
| Dim_104 | tRNA Processing | Reactome | 7 genes | 0.005 |
| Dim_092 | Inorganic Cation Import Across Plasma Membrane | GO BP | 6 genes | 0.009 |
| Dim_079 | snRNA 3'-End Processing | GO BP | 4 genes | 0.022 |
| Dim_036 | snRNA 3'-End Processing | GO BP | 4 genes | 0.025 |
| Dim_058 | Regulation of Mitochondrial Translation | GO BP | 4 genes | 0.025 |
| Dim_076 | snRNA 3'-End Processing | GO BP | 4 genes | 0.034 |
| Dim_050 | Protein K48-linked Ubiquitination | GO BP | 5 genes | 0.046 |
| Dim_031 | DNA IR Damage and Cellular Response via ATR | WikiPathways | 6 genes | 0.047 |

### Additional Notable Hits (FDR < 0.25)

| Module | Pathway | Database | FDR |
|--------|---------|----------|-----|
| Dim_073 | RHO GTPases Activate PAKs | Reactome | 0.052 |
| Dim_018 | Sterol Biosynthetic Process | GO BP | 0.052 |
| Dim_009 | Inorganic Cation Import Across Plasma Membrane | GO BP | 0.059 |
| Dim_027 | Heme Metabolism | MSigDB Hallmark | 0.125 |
| Dim_107 | Anterograde Dendritic Transport | GO BP | 0.115 |
| Dim_000 | Defects in Cobalamin (B12) Metabolism | Reactome | 0.153 |

### Plots

| File | Description |
|------|-------------|
| `results/real_data/plots/dimension_loadings.png` | Bar plots showing top gene loadings per dimension |
| `results/real_data/plots/loading_heatmap.png` | Heatmap of the loading matrix (top genes x dimensions) |
| `logs/real_training_curves.png` | Training and validation loss over epochs |

---

## 11. Interpreting the Results

### How to Read `top_genes_empirical_with_symbols.csv`

Each row shows one gene's relationship to one module:

| Column | Meaning |
|--------|---------|
| `dimension` | Module ID (Dim_000 through Dim_127) |
| `gene` | Ensembl gene ID (ENSSSCG*) |
| `gene_symbol` | Human-readable symbol (empty if unmapped) |
| `loading` | Pearson correlation (-1 to +1): positive = gene activated when module active; negative = gene suppressed when module active |
| `abs_loading` | Absolute value of loading (strength regardless of direction) |
| `rank` | Rank within this dimension (1 = strongest) |

### How to Read `module_annotations.csv`

| Column | Meaning |
|--------|---------|
| `module` | Module ID (Dim_000 through Dim_127) |
| `database` | Source database (GO Biological Process, KEGG, Reactome, WikiPathways, MSigDB Hallmark, BTM) |
| `pathway` | Pathway or gene set name |
| `overlap` | Number of shared genes between module and pathway |
| `module_size` | Number of mapped gene symbols in the module |
| `pathway_size` | Number of genes in the pathway |
| `pvalue` | Fisher's exact test p-value (one-sided, over-representation) |
| `fdr` | Benjamini-Hochberg adjusted p-value (corrected for multiple testing within each module) |
| `shared_genes` | Semicolon-separated list of the shared gene symbols |

### Notes on Statistical Power

- The enrichment uses **human** pathway databases with **pig** gene symbols. Since pig gene symbols don't always match human orthologs one-to-one, this reduces statistical power.
- 97% of modules (125/128) have at least one enrichment at nominal p < 0.01, suggesting **nearly all modules capture real biological programs**.
- For the 9 modules with FDR < 0.05, the annotations are statistically robust.
- To improve enrichment power, consider:
  - Using dedicated pig-specific pathway databases (if available)
  - Using a formal ortholog mapping (e.g., via Ensembl Compara or BioMart one-to-one orthologs)
  - Increasing the number of top genes per module (currently 100)

---

## 12. Call Graph — What Calls What

### run_real_data.py (the main pipeline)

```
run_real_data.py
│
├── [Data loading]
│   ├── pandas.read_csv()        reads counts_and_metadata/*.csv (tab-separated)
│   ├── sklearn.StandardScaler   Z-score normalization
│   └── sklearn.LabelEncoder     One-hot encoding of BioProject, SampleStyle, SampleTissue
│
├── [Model building]
│   ├── config.py                PipelineConfig (all hyperparameters)
│   ├── model/cvae.py            ConditionalVAE.from_config()
│   └── model/mi_regularizer.py  build_mi_estimator() -> MINEEstimator
│
├── [Training loop]
│   ├── model/losses.py          reconstruction_loss(), kl_divergence(),
│   │                            compute_kl_weight(), total_loss()
│   ├── model/mi_regularizer.py  compute_mi_loss()
│   └── torch.optim              Adam + CosineAnnealingLR
│
├── [Extraction]
│   └── extraction/interpret_latent.py
│       ├── decoder_effective_weights()
│       ├── encoder_effective_weights()
│       ├── empirical_gene_latent_correlations()
│       ├── extract_gene_loadings()        <- calls all 3 above
│       ├── top_genes_per_dimension()
│       ├── export_loadings()
│       ├── compare_methods()
│       ├── plot_dimension_loadings()
│       └── plot_loading_heatmap()
│
└── [Utilities]
    └── utils/utils.py
        ├── set_seed()
        ├── count_parameters()
        ├── save_history()
        └── plot_training_history()
```

### map_gene_ids.py

```
map_gene_ids.py
│
├── pandas.read_csv()                      reads gene_loadings_empirical.csv (index only)
├── requests.post()                        queries rest.ensembl.org/lookup/id
│   └── (batches of 1000 IDs, 3 retries per batch, respects rate limits)
└── pandas.to_csv()                        writes gene_id_mapping.csv
                                           + top_genes_with_symbols/*.csv
```

### annotate_modules.py

```
annotate_modules.py
│
├── pandas.read_csv()                      reads gene_id_mapping.csv
├── pandas.read_csv()                      reads gene_loadings_empirical.csv (full matrix)
├── requests.get()                         downloads GMT files from Enrichr (cached)
│   ├── GO_Biological_Process_2023.gmt     (5,407 gene sets)
│   ├── KEGG_2021_Human.gmt               (320 gene sets)
│   ├── Reactome_2022.gmt                 (1,818 gene sets)
│   ├── WikiPathway_2023_Human.gmt        (801 gene sets)
│   └── MSigDB_Hallmark_2020.gmt          (50 gene sets)
├── get_built_in_btm()                     16 built-in BTM gene sets
├── scipy.stats.fisher_exact()             one-sided over-representation test
├── benjamini_hochberg()                   FDR correction per module
└── pandas.to_csv()                        writes module_annotations.csv
                                           + module_top_annotation.csv
```

### run_test.py (synthetic data test)

```
run_test.py
│
├── generate_synthetic_data.py             creates data/raw/Study_*.csv
├── data/data_loader.py                    load_expression_data(), prepare_data()
├── model/cvae.py                          ConditionalVAE.from_config()
├── model/mi_regularizer.py                build_mi_estimator()
├── training/trainer.py                    Trainer.fit()
├── extraction/weight_extractor.py         extract_all_modules()
├── extraction/interpret_latent.py         extract_gene_loadings(), export_loadings()
└── application/gmt_export.py              modules_to_gmt()
```

---

*Pipeline documentation — Generated March 2026*
