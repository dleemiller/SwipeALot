"""Structured config for downstream text-to-path CVAE fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class PathVAEModelConfig:
    encoder_path: str = "checkpoints/base_20251217_211504/final"
    freeze_encoder: bool = True

    decoder_n_layers: int = 2
    decoder_n_heads: int | None = None
    decoder_d_ff: int | None = None
    dropout: float | None = None

    out_dim: int = 2
    sigma_min: float = 1e-3
    target_eps: float = 1e-4
    path_loss_radial_weight: float = 0.0
    path_sigma_target_min: float = 0.1
    path_sigma_target_max: float = 0.4
    uncertainty_reg_weight: float = 0.0

    latent_dim: int = 16
    latent_hidden_dim: int | None = None
    kl_weight: float = 0.1
    smoothness_weight: float = 0.0
    smoothness_order: int = 2
    speed_smoothness_weight: float = 0.0

    spline_enabled: bool = False
    spline_num_ctrl: int = 16
    spline_degree: int = 3
    spline_adaptive: bool = False
    spline_min_ctrl: int = 12
    spline_max_ctrl: int = 32
    spline_ctrl_per_char: float = 0.5

    path_encoder_n_layers: int = 2
    path_encoder_n_heads: int | None = None
    path_encoder_d_ff: int | None = None
    path_encoder_dropout: float | None = None


@dataclass
class PathVAEDataConfig:
    dataset_name: str = "futo-org/swipe.futo.org"
    train_split: str = "train"
    val_split: str = "validation"
    path_resample_mode: str = "time"
    reverse_prob: float = 0.0

    max_train_samples: int | None = None
    max_eval_samples: int | None = 10_000


@dataclass
class PathVAETrainingConfig:
    training_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathVAEConfig:
    model: PathVAEModelConfig = field(default_factory=PathVAEModelConfig)
    data: PathVAEDataConfig = field(default_factory=PathVAEDataConfig)
    training: PathVAETrainingConfig = field(default_factory=PathVAETrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> PathVAEConfig:
        yaml_conf = OmegaConf.load(path)
        structured_conf = OmegaConf.structured(cls)
        merged = OmegaConf.merge(structured_conf, yaml_conf)
        return OmegaConf.to_object(merged)
