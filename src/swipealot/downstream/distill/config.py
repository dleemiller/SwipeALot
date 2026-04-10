"""Structured config for downstream distillation fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class DistillModelConfig:
    encoder_path: str | None = "checkpoints/base_20251217_211504/final"
    freeze_encoder: bool = False  # Task adaptation: don't freeze
    encoder_lr_scale: float = 0.1  # Backbone LR = base_lr * 0.1

    # Projector
    projector_dim: int = 128  # D: configurable 26-128

    # Temporal adapter (matches encodercnn TemporalAdapter)
    adapter_num_stages: int = 2  # 128 -> 32 with 2 stages
    adapter_kernel_size: int = 5
    adapter_stride: int = 2
    adapter_double_channels: bool = False  # double channels at last adapter stage
    adapter_fold_last_stage: bool = False  # replace last conv stage with channel fold

    # Decoder
    decoder_type: str = "dfsmn"
    rnn_hidden: int = 128  # decoder hidden dimension

    # DFSMN decoder
    dfsmn_num_layers: int = 10
    dfsmn_proj_dim: int = 64
    dfsmn_context: int = 7

    # CTC
    num_chars: int = 26  # a-z
    blank_idx: int = 26

    # Text masking
    text_mask_prob: float = 1.0  # 1.0 = full modality mode

    # Load full distill model from a previous run (skips from_encoder_pretrained)
    init_checkpoint: str | None = None

    # Stage 0 (attention shaping) checkpoint for warm-starting encoder + projector
    stage0_checkpoint: str | None = None


@dataclass
class DistillDataConfig:
    dataset_name: str = "futo-org/swipe.futo.org"
    dataset_config: str | None = None
    train_split: str = "train"
    val_split: str = "validation"
    path_resample_mode: str = "time"
    exclude_sources: list[str] = field(default_factory=list)

    # OT cost filter: remove samples with optimal transport cost above threshold
    ot_cost_threshold: float | None = 0.04

    # Extra NPZ datasets (path features + words, no attention needed)
    extra_npz_paths: list[str] = field(default_factory=list)

    # Vocabulary file for trie-constrained beam search eval
    vocab_path: str | None = None

    max_train_samples: int | None = None
    max_eval_samples: int | None = 10_000


@dataclass
class DistillTrainingConfig:
    training_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillConfig:
    model: DistillModelConfig = field(default_factory=DistillModelConfig)
    data: DistillDataConfig = field(default_factory=DistillDataConfig)
    training: DistillTrainingConfig = field(default_factory=DistillTrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> DistillConfig:
        yaml_conf = OmegaConf.load(path)
        structured_conf = OmegaConf.structured(cls)
        merged = OmegaConf.merge(structured_conf, yaml_conf)
        return OmegaConf.to_object(merged)
