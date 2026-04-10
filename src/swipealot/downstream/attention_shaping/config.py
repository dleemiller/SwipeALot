"""Structured config for attention shaping training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class AttentionShapingModelConfig:
    encoder_path: str = "checkpoints/base_20260212_171550/final"
    projector_dim: int = 128
    attention_target_layers: list[int] = field(default_factory=lambda: [4, 5, 6])
    softplus_beta: float = 1.0
    huber_beta: float = 0.01
    char_kl_weight: float = 0.2
    valid_huber_weight: float = 5.0
    length_loss_weight: float = 0.1


@dataclass
class AttentionShapingDataConfig:
    dataset_name: str = "futo-org/swipe.futo.org"
    train_split: str = "train"
    val_split: str = "validation"
    path_resample_mode: str = "time"
    max_train_samples: int | None = None
    max_eval_samples: int | None = 10_000


@dataclass
class AttentionShapingTrainingConfig:
    training_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionShapingConfig:
    model: AttentionShapingModelConfig = field(default_factory=AttentionShapingModelConfig)
    data: AttentionShapingDataConfig = field(default_factory=AttentionShapingDataConfig)
    training: AttentionShapingTrainingConfig = field(default_factory=AttentionShapingTrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> AttentionShapingConfig:
        yaml_conf = OmegaConf.load(path)
        structured_conf = OmegaConf.structured(cls)
        merged = OmegaConf.merge(structured_conf, yaml_conf)
        return OmegaConf.to_object(merged)
