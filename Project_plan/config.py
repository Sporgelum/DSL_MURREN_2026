"""
Configuration for the MI-Regularized cVAE pipeline.
All hyperparameters, paths, and settings are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""
    data_dir: str = "data/raw"
    metadata_cols: List[str] = field(default_factory=lambda: [
        "study_id", "vaccine_type", "time_point"
    ])
    gene_expression_prefix: str = "gene_"
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    normalize: bool = True
    log_transform: bool = True
    random_seed: int = 42


@dataclass
class ModelConfig:
    """cVAE architecture settings."""
    input_dim: int = 20000          # Number of genes
    condition_dim: int = 10         # Dimension of one-hot encoded metadata
    latent_dim: int = 128           # Bottleneck dimension (= number of modules)
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [2048, 512, 256])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 2048])
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class MIConfig:
    """Mutual Information regularization settings."""
    mi_weight: float = 0.1          # Lambda for MI penalty in the loss
    mi_hidden_dim: int = 256        # Hidden dim of the MI estimation network
    mi_estimator: str = "mine"      # 'mine' or 'nwj'


@dataclass
class TrainingConfig:
    """Training loop settings."""
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    kl_weight: float = 1.0          # Beta for KL divergence term
    kl_anneal_epochs: int = 30      # Linear KL annealing over N epochs
    early_stopping_patience: int = 15
    scheduler: str = "cosine"       # 'cosine' or 'plateau'
    gradient_clip: float = 1.0
    device: str = "auto"            # 'auto', 'cuda', or 'cpu'
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class ExtractionConfig:
    """Module extraction settings."""
    zscore_threshold: float = 2.5   # Sigma threshold for gene selection
    top_n_genes: Optional[int] = None  # If set, use top-N instead of Z-score
    min_module_size: int = 10
    max_module_size: int = 500


@dataclass
class ApplicationConfig:
    """Downstream application settings."""
    gmt_output_path: str = "results/btm_modules.gmt"
    gsea_database: str = "MSigDB"
    projection_output_dir: str = "results/projections"


@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mi: MIConfig = field(default_factory=MIConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
