# Configuration — `config.py`

## Goal
Centralize all hyperparameters, file paths, and settings for the entire BTM discovery pipeline in a single, version-controllable Python file.

## Main Idea
Instead of scattering magic numbers across scripts, all tunable parameters live in structured dataclasses. Each pipeline stage reads its relevant sub-config. This makes experiments reproducible and hyperparameter sweeps straightforward.

## Config Sections

| Dataclass | Controls |
|---|---|
| `DataConfig` | Data directory, metadata columns, train/val/test splits, normalization flags |
| `ModelConfig` | cVAE architecture: input/latent dims, hidden layer sizes, dropout, batch norm |
| `MIConfig` | MI estimator type (MINE/NWJ), regularization weight λ, hidden dim |
| `TrainingConfig` | Epochs, batch size, learning rate, KL annealing, early stopping, device |
| `ExtractionConfig` | Z-score threshold, top-N genes, min/max module sizes |
| `ApplicationConfig` | GMT output path, GSEA database name, projection output directory |
| `PipelineConfig` | Master config aggregating all sub-configs |

## How to Modify
Edit `config.py` directly, or override in code:

```python
from config import PipelineConfig

cfg = PipelineConfig()
cfg.model.latent_dim = 64          # fewer modules
cfg.training.epochs = 300          # longer training
cfg.extraction.zscore_threshold = 3.0  # stricter gene selection
```

## Inputs
None — this is a pure configuration file.

## Outputs
`PipelineConfig` dataclass instance consumed by all pipeline stages.
