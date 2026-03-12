# Extraction Module — Decoder Weight Extraction & BTM Discovery

## Goal
After training, extract the decoder weight matrix from the cVAE to identify Blood Transcription Modules (BTMs) — coordinated groups of genes that represent specific biological processes activated during immune responses.

## Main Idea
The decoder of the cVAE learns a mapping from latent dimensions back to gene expression space. Each latent dimension $j$ effectively acts as a "module" — the genes with the largest absolute weights for that dimension are the genes most strongly associated with that module's biological signal. By extracting and thresholding these weights, we obtain interpretable gene sets.

## Algorithm

1. **Effective weight computation**: Multiply through all decoder linear layers to get an approximate linear map $W_{\text{eff}}$ from latent space to gene space:
   $$W_{\text{eff}} = W_L \cdot W_{L-1} \cdots W_1$$

2. **Gene ranking**: For each latent dimension (module) $j$, extract column $j$ of $W_{\text{eff}}$ and rank genes by absolute weight.

3. **Z-score thresholding**: Compute the Z-score of absolute weights within each module. Select genes exceeding a threshold (default: $>2.5\sigma$).

4. **Size constraints**: Enforce minimum (10) and maximum (500) module sizes to ensure biological relevance.

## Inputs
| Input | Description |
|---|---|
| Trained `ConditionalVAE` | Model with learned decoder weights |
| `gene_names` | List of gene identifiers (matching expression columns) |
| `ExtractionConfig` | Z-score threshold, top-N, min/max module size |

## Outputs
| Output | Description |
|---|---|
| `btm_modules` | Dict mapping module index → DataFrame of selected genes with weights and Z-scores |
| Per-module DataFrames | Columns: gene, weight, abs_weight, zscore |

## Key Components
- **`compute_effective_weights()`** — Chains decoder linear layers to get a (n_genes × latent_dim) matrix.
- **`rank_genes_per_module()`** — Ranks genes by abs contribution for each module.
- **`select_module_genes()`** — Applies Z-score or top-N thresholding with size constraints.
- **`extract_all_modules()`** — End-to-end pipeline from model to gene lists.
