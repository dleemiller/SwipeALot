"""Configuration dataclasses for the swipe keyboard model."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Transformer architecture
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    # Vocabulary (will be set from tokenizer)
    vocab_size: int = 100

    # Sequence lengths
    max_path_len: int = 64
    max_char_len: int = 38

    # Path features
    path_input_dim: int = 8  # (x, y, dx, dy, d2x, d2y, speed, curvature)

    # Tasks
    predict_path: bool = True
    predict_char: bool = True  # Core MLM objective (can be disabled with char_loss_weight=0)
    predict_length: bool = True  # Auxiliary CLS head to predict swipable length
    predict_path_uncertainty: bool = False  # Predict log sigma for path coords


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Dataset
    dataset_name: str = "futo-org/swipe.futo.org"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

    # Data processing
    max_path_len: int = 64
    max_char_len: int = 38
    path_resample_mode: str = "time"  # "time" (dwell-aware) or "spatial" (distance-uniform)

    # Masking strategy
    char_mask_prob: Any = 0.15  # Float or [min, max] range for random masking
    path_mask_prob: float = 0.15
    path_mask_block_max_len: int = 32  # Max contiguous masked block length
    mask_path: bool = True
    mask_vocab_only: bool = False  # Only mask vocabulary tokens (a-z, 0-9)

    # DataLoader
    batch_size: int | None = None  # Deprecated; use training.training_args.per_device_* instead
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Training configuration with custom parameters and HuggingFace TrainingArguments passthrough."""

    # Custom loss weights
    char_loss_weight: float = 1.0
    path_loss_weight: float = 0.1
    length_loss_weight: float = 0.1  # Auxiliary CLS length prediction
    path_loss_dims: list[int] | None = None  # e.g. [0,1] to supervise x/y only
    path_loss_end_weight: float = 1.0  # Linear ramp weight applied from start->end
    path_loss_radial_weight: float = 0.0  # Extra weight for points farther from center
    path_sigma_min: float = 0.1  # Variance floor for heteroscedastic path loss
    path_best_of_k: int = 1  # Use best-of-K MSE when log sigma is available (K>=1)

    # Uncertainty regularization (Phase 3: prevent late-stage collapse)
    path_sigma_target_min: float = 0.1  # Target healthy range lower bound for endpoint sigma
    path_sigma_target_max: float = 0.4  # Target healthy range upper bound for endpoint sigma
    uncertainty_reg_weight: float = 0.1  # Weight for uncertainty regularization loss

    # Custom loss settings
    use_focal_loss: bool = False
    focal_gamma: float = 0.0
    use_char_freq_weights: bool = False
    char_weights_path: str | None = None

    # Pairwise masking + contrastive objective
    use_pairwise_masking: bool = False
    pairwise_modality_prob: float = 0.2  # Probability of using modality-based masking (vs inverted)
    pairwise_zero_attention_prob: float = 0.5  # Probability of zeroing attention in modality mode
    # NOTE: OmegaConf structured configs do not support unions of container types, so these
    # are typed as `Any` and validated/consumed by the collator (float or 2-item range).
    pairwise_inverted_char_prob_heavy: Any = (0.5, 0.7)
    pairwise_inverted_path_prob_heavy: Any = (0.5, 0.7)
    pairwise_inverted_char_prob_light: Any = (0.1, 0.2)
    pairwise_inverted_path_prob_light: Any = (0.1, 0.2)
    pairwise_right_half_prob: float = (
        0.0  # Fraction of non-modality batches using right-half masking
    )
    pairwise_right_half_path_prob: Any = (0.6, 0.8)
    pairwise_right_half_reverse_prob: float = 0.0
    pairwise_inverted_reverse_prob: float = 0.0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1

    # Matryoshka settings
    matryoshka_dims: list[int] | None = None
    matryoshka_weights: list[float] | None = None

    # HuggingFace TrainingArguments (passthrough for standard parameters)
    # This allows any standard HF argument to be passed without explicit definition
    training_args: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        conf = OmegaConf.structured(self)
        OmegaConf.save(conf, path)


@dataclass
class Config:
    """Complete configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from YAML file using OmegaConf.

        Provides validation, type checking, and better error messages.
        Supports variable interpolation (e.g., ${model.d_model}).
        """
        # Load YAML and merge with structured config for validation
        yaml_conf = OmegaConf.load(path)

        # Create structured config from dataclass (provides validation)
        structured_conf = OmegaConf.structured(cls)

        # Merge YAML onto structured config (validates types and structure)
        merged = OmegaConf.merge(structured_conf, yaml_conf)

        # Convert to regular dataclass instance
        return OmegaConf.to_object(merged)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file using OmegaConf."""
        # Convert dataclass to OmegaConf
        conf = OmegaConf.structured(self)

        # Save to file
        OmegaConf.save(conf, path)
