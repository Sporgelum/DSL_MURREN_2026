"""
Configuration for the MINE-enhanced cVAE pipeline.

Extends the original config.py with MINE-specific hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """cVAE architecture (same as original)."""
    input_dim: int = 20000
    condition_dim: int = 10
    latent_dim: int = 128
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [2048, 512, 256])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 2048])
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class MINEConfig:
    """
    MINE-enhanced mutual information configuration.

    New vs original:
      - proj_dim, per_dim_hidden, noise_std: deeper statistics network
      - ema_alpha: EMA bias correction
      - use_dimwise: dimension-wise MI instead of global
      - tc_weight, tc_n_pairs: Total Correlation penalty
      - use_adaptive_clip: Paper §8.1.1 gradient balancing
    """
    # --- Shared ---
    mi_weight: float = 0.1              # λ_MI: MI maximization strength
    hidden_dim: int = 256               # Hidden dim for joint estimator
    ema_alpha: float = 0.01             # EMA decay for bias correction

    # --- Deep statistics network (Improvement 6) ---
    proj_dim: int = 512                 # Gene projection dimension
    noise_std: float = 0.3              # Gaussian noise injection std

    # --- Dimension-wise MI (Improvement 3) ---
    use_dimwise: bool = True            # Use per-dimension MI instead of global
    per_dim_hidden: int = 128           # Hidden dim per per-dimension network

    # --- Total Correlation / disentanglement (Improvement 4) ---
    tc_weight: float = 0.05            # λ_TC: disentanglement strength
    tc_n_pairs: int = 32               # Random dimension pairs per batch
    tc_hidden_dim: int = 64            # Hidden dim for pairwise network

    # --- Adaptive gradient clipping (Improvement 2) ---
    use_adaptive_clip: bool = True     # Enable adaptive MI gradient clipping


@dataclass
class TrainingConfig:
    """Training loop settings."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    mi_learning_rate: float = 1e-4     # Separate LR for MINE networks (lower is more stable)
    weight_decay: float = 1e-5
    kl_weight: float = 1.0
    kl_anneal_epochs: int = 30
    early_stopping_patience: int = 20   # More patience — more loss terms to stabilize
    gradient_clip: float = 1.0
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    mi_steps_per_vae_step: int = 1     # How many MI updates per VAE update


@dataclass
class ExtractionConfig:
    """Module extraction settings."""
    top_n_genes: int = 100              # Top genes per module for annotation
    mine_extraction_epochs: int = 50    # Epochs for post-hoc MINE-based extraction
    mine_extraction_lr: float = 1e-3
    mine_extraction_hidden: int = 128


@dataclass
class PipelineConfig:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    mine: MINEConfig = field(default_factory=MINEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
