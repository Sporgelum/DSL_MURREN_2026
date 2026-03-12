# MINE-Based Gene Network Inference

## Overview

This pipeline replaces **histogram-based mutual information (MI)** with **MINE
(Mutual Information Neural Estimation)** for inferring gene co-expression
networks from bulk RNA-seq data.

### Why MINE instead of histogram MI?

| Feature | Histogram MI | MINE |
|---------|-------------|------|
| Data processing | Discretise into bins (KBinsDiscretiser) | Operate on continuous Z-scored data |
| Information loss | Binning discards fine structure | No binning needed |
| Small sample behaviour | Noisy with few samples + many bins | Better sample efficiency via neural net |
| Nonlinear dependencies | Limited by bin resolution | Neural net can model complex patterns |
| Computational cost | O(n) per pair, very fast | O(n × epochs) per pair, GPU-accelerated |

## Pipeline Steps

1. **Load** logCPM expression + metadata
2. **Discover studies** from BioProject column
3. **Per study:**
   - Z-score expression (mean=0, std=1 per gene)
   - Pre-screen pairs by Pearson |r| > threshold (fast reduction)
   - Estimate MI via **batched MINE** on candidate pairs (GPU-parallel)
   - Build null distribution via permutation + MINE
   - Filter edges by empirical p-value
4. **Master network:** multi-study consensus (edge appears in ≥ k studies)
5. **MCODE** module detection on master graph

## Key Innovation: Batched MINE

Training one neural network per gene pair is too slow for millions of
candidates. Instead, we train **B independent MINE networks simultaneously**
using batched tensor operations (`torch.bmm`). Each network has its own weights
but all B share a single forward/backward pass on the GPU.

```
B=512 pairs × 200 epochs ≈ 5 seconds on GPU
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings (MINE, pre-screen, null, network, MCODE) |
| `mine_estimator.py` | Batched MINE core: `BatchedMINE`, `estimate_mi_for_pairs`, `build_mine_null` |
| `prescreen.py` | Fast Pearson/Spearman correlation pre-screening |
| `mcode.py` | MCODE module detection (Bader & Hogue 2003) |
| `utils.py` | Data loading, logging, timing, saving |
| `run_mine_network.py` | Main orchestrator — entry point |

## Usage

```bash
# From Project_MINE_network/
python run_mine_network.py

# Or with a specific environment:
& "C:\path\to\python.exe" run_mine_network.py
```

Edit `run_mine_network.py` top section or `config.py` to adjust data paths
and hyperparameters.

## Expected Output (`output/`)

- Per-study: significant edge lists, adjacency matrices (`.npz`), GraphML
- Master network: consensus edge list, adjacency, GraphML with MCODE modules
- Null QC: histogram PNGs showing the null distribution
- Analysis report: runtime breakdown + summary statistics
- Log file: full stdout capture

## Requirements

- Python 3.9+
- `torch` (CPU or CUDA)
- `numpy`, `pandas`, `scipy`, `joblib`
- `igraph` (for MCODE)
- `matplotlib` (for null QC plots)

## Reference

Belghazi et al., *Mutual Information Neural Estimation*, ICML 2018
([arXiv:1801.04062](https://arxiv.org/abs/1801.04062))
