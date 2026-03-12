"""
Configuration for MINE-based gene network inference.

Drop-in replacement for histogram MI in generate_net_python_pval.py,
using neural MI estimation (Belghazi et al., ICML 2018) on continuous data.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MINEConfig:
    """MINE statistics network and training."""
    hidden_dim: int = 64          # T(a,b): 2 → H → H → 1
    n_epochs: int = 200           # SGD epochs per batch of pairs
    learning_rate: float = 1e-3
    ema_alpha: float = 0.01       # EMA bias correction (Paper §3.2)
    batch_pairs: int = 512        # Gene pairs processed simultaneously
    gradient_clip: float = 1.0
    n_eval_shuffles: int = 5      # Averaged marginal shuffles for final estimate


@dataclass
class PrescreenConfig:
    """Fast correlation pre-screening to reduce candidate pairs."""
    method: str = "pearson"       # "pearson" or "spearman"
    threshold: float = 0.3        # Minimum |r| to keep a pair
    max_pairs: int = 500_000      # Hard cap on candidate pairs per study


@dataclass
class PermutationConfig:
    """Permutation-based null distribution for MINE."""
    n_permutations: int = 10_000  # Null samples (batched → fast)
    seed: int = 42
    p_value_threshold: float = 0.001


@dataclass
class NetworkConfig:
    """Master network construction."""
    min_study_count: int = 3
    min_study_fraction: float = 0.30   # Dynamic: edge in >= 30% of studies
    min_samples_per_study: int = 3     # keep studies with >= 3 samples


@dataclass
class MCODEConfig:
    """MCODE module detection (Bader & Hogue 2003)."""
    score_threshold: float = 0.2
    min_size: int = 3
    min_density: float = 0.3


@dataclass
class PipelineConfig:
    """Master configuration."""
    mine: MINEConfig = field(default_factory=MINEConfig)
    prescreen: PrescreenConfig = field(default_factory=PrescreenConfig)
    permutation: PermutationConfig = field(default_factory=PermutationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcode: MCODEConfig = field(default_factory=MCODEConfig)

    # Paths — set at runtime in run_mine_network.py
    counts_path: str = ""
    metadata_path: str = ""
    output_dir: str = ""

    # Hardware
    device: str = "auto"          # "auto", "cuda", "cpu"
    n_jobs: int = -1              # CPU cores for Pearson pre-screening

    # Optional BH-FDR
    apply_bh_fdr: bool = False
    bh_fdr_alpha: float = 0.05
