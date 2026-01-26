"""Structured config for downstream text→path fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class Text2PathModelConfig:
    encoder_path: str = "checkpoints/base_20251217_211504/final"
    freeze_encoder: bool = True

    decoder_n_layers: int = 2
    decoder_n_heads: int | None = None
    decoder_d_ff: int | None = None
    dropout: float | None = None

    out_dim: int = 2
    sigma_min: float = 1e-3
    target_eps: float = 1e-4
    scheduled_sampling_ratio: float = 0.0
    scheduled_sampling_warmup_steps: int = 0


@dataclass
class Text2PathDataConfig:
    dataset_name: str = "futo-org/swipe.futo.org"
    train_split: str = "train"
    val_split: str = "validation"
    path_resample_mode: str = "time"

    max_train_samples: int | None = None
    max_eval_samples: int | None = 10_000


@dataclass
class Text2PathTrainingConfig:
    training_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Text2PathConfig:
    model: Text2PathModelConfig = field(default_factory=Text2PathModelConfig)
    data: Text2PathDataConfig = field(default_factory=Text2PathDataConfig)
    training: Text2PathTrainingConfig = field(default_factory=Text2PathTrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Text2PathConfig:
        yaml_conf = OmegaConf.load(path)
        structured_conf = OmegaConf.structured(cls)
        merged = OmegaConf.merge(structured_conf, yaml_conf)
        return OmegaConf.to_object(merged)
