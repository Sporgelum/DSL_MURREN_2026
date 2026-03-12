# Application Module — GMT Export, GSEA Annotation & Projection Tool

## Goal
Transform discovered BTM modules into practical bioinformatics tools: (1) a .gmt gene set file for use in standard GSEA workflows, (2) functional annotations by cross-referencing with known pathway databases, and (3) a projection tool for scoring new datasets against the learned modules.

## Main Idea
The trained cVAE captures coordinated gene programs as latent dimensions. This module bridges the gap between the neural network output and real-world biological utility. It packages results in formats that integrate with the existing gene set analysis ecosystem (MSigDB, fgsea, GSEA desktop) and provides a rapid inference pipeline for new data.

## Scripts

### 1. `gmt_export.py` — Gene Set Export
Converts discovered modules into the standard GMT (Gene Matrix Transposed) format.

**Algorithm**: For each module, collect the selected gene list and write one tab-separated line: `MODULE_NAME <TAB> DESCRIPTION <TAB> GENE1 <TAB> GENE2 ...`

| Input | Output |
|---|---|
| `btm_modules` dict (from extraction) | `.gmt` file at configured path |

### 2. `annotation.py` — GSEA / MSigDB Annotation
Cross-references each module's gene set against known biological pathways to assign functional labels (e.g., "Module 5 = Early Interferon Response").

**Algorithm**:
1. Load a reference .gmt file (e.g., MSigDB Hallmark, C7 Immunologic)
2. For each module, compute Jaccard similarity with every reference pathway
3. Rank and keep top-K annotations per module (minimum overlap threshold)

| Input | Output |
|---|---|
| `btm_modules` + reference `.gmt` | `module_annotations.csv` with pathway, overlap, Jaccard per module |

### 3. `projection.py` — Digital Inference / Projection Tool
Passes new expression data through the trained encoder to obtain per-sample module Activity Scores.

**Algorithm**:
1. Preprocess new expression data (log1p + scaler from training)
2. One-hot encode new metadata
3. Forward pass through the encoder (deterministic: use mu, not sampled z)
4. Output: matrix of (n_samples × n_modules) activity scores

| Input | Output |
|---|---|
| New expression matrix + metadata + trained model | `module_activity_scores.csv` — per-sample module scores |
| Activity scores + group column | Mean activity comparison across conditions |

## Use Cases
- **Standard GSEA**: Load the `.gmt` file into fgsea/GSEA desktop with any new RNA-seq experiment to check which "Vaccine-Response Modules" are enriched.
- **Digital Inference**: Score a new vaccine trial's samples against all modules instantly. Compare: "Does New Vaccine A trigger the same Innate Module as Old Vaccine B?"
- **Meta-analysis**: Project multiple studies into the shared module space for cross-study comparison.
