"""Configuration dataclasses for the swipe keyboard model."""

from dataclasses import dataclass, field

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

    # Tasks
    predict_path: bool = True


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
    char_mask_prob: float = 0.15
    path_mask_prob: float = 0.15
    mask_path: bool = True

    # DataLoader
    batch_size: int = 512
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5  # Minimum LR for cosine annealing
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000

    # Loss weights
    char_loss_weight: float = 1.0
    path_loss_weight: float = 0.1

    # Logging and checkpointing
    log_interval: int = 100
    val_interval: int = 1000  # Run validation every N steps
    save_interval: int = 1  # Save checkpoint every N epochs
    keep_n_checkpoints: int = 2  # Keep best + N most recent checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda"

    # Mixed precision training
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"

    # Character loss shaping
    use_focal_loss: bool = False
    focal_gamma: float = 0.0
    use_char_freq_weights: bool = False
    char_freq_max_samples: int = 100000
    char_weights_path: str | None = None

    # Pairwise masking + contrastive objective
    use_pairwise_masking: bool = False
    pairwise_modality_prob: float = 0.2  # Probability of using modality-based masking (vs inverted)
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1


@dataclass
class CrossEncoderDataConfig:
    """Data configuration for cross-encoder training."""

    # Dataset
    dataset_name: str = "futo-org/swipe.futo.org"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

    # Negative mining
    negative_pool_path: str | None = None
    num_negatives: int = 3  # Number of negatives per positive (1:3 ratio)
    difficulty_sampling: bool = True  # Sample negatives by difficulty score

    # Sequence lengths
    max_path_len: int = 64
    max_word_len: int = 48

    # DataLoader
    batch_size: int = 256
    num_workers: int = 4


@dataclass
class CrossEncoderTrainingConfig:
    """Training configuration for cross-encoder."""

    # Pretrained model
    base_checkpoint: str | None = None  # Path to pretrained encoder checkpoint

    # Encoder freezing
    freeze_encoder: bool = True  # Freeze encoder during training

    # Optimization
    head_learning_rate: float = 1e-4  # LR for classification head
    encoder_learning_rate: float = 5e-6  # LR for encoder (when unfrozen)
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500

    # Loss
    mnr_scale: float = 10.0  # Temperature scaling for MNR loss

    # Logging and checkpointing
    log_interval: int = 100
    val_interval: int = 500
    save_interval: int = 1
    keep_n_checkpoints: int = 2
    log_dir: str = "logs/cross_encoder"
    checkpoint_dir: str = "checkpoints/cross_encoder"

    # Device
    device: str = "cuda"

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"


@dataclass
class CrossEncoderConfig:
    """Complete configuration for cross-encoder training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: CrossEncoderDataConfig = field(default_factory=CrossEncoderDataConfig)
    training: CrossEncoderTrainingConfig = field(default_factory=CrossEncoderTrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "CrossEncoderConfig":
        """Load configuration from YAML file."""
        yaml_conf = OmegaConf.load(path)
        structured_conf = OmegaConf.structured(cls)
        merged = OmegaConf.merge(structured_conf, yaml_conf)
        return OmegaConf.to_object(merged)

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
