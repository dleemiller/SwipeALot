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
    path_input_dim: int = 6  # (x, y, dx, dy, ds, log_dt)

    # Tasks
    predict_path: bool = True
    predict_char: bool = True  # Core MLM objective (can be disabled with char_loss_weight=0)
    predict_length: bool = True  # Auxiliary CLS head to predict swipable length


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

    # Masking strategy
    char_mask_prob: Any = 0.15  # Float or [min, max] range for random masking
    path_mask_prob: float = 0.15
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

    # Custom loss settings
    use_focal_loss: bool = False
    focal_gamma: float = 0.0
    use_char_freq_weights: bool = False
    char_weights_path: str | None = None

    # Pairwise masking + contrastive objective
    use_pairwise_masking: bool = False
    pairwise_modality_prob: float = 0.2  # Probability of using modality-based masking (vs inverted)
    pairwise_zero_attention_prob: float = 0.5  # Probability of zeroing attention in modality mode
    pairwise_inverted_char_prob_heavy: float | tuple[float, float] = (0.5, 0.7)
    pairwise_inverted_path_prob_heavy: float | tuple[float, float] = (0.5, 0.7)
    pairwise_inverted_char_prob_light: float | tuple[float, float] = (0.1, 0.2)
    pairwise_inverted_path_prob_light: float | tuple[float, float] = (0.1, 0.2)
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
