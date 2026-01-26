"""Configuration for the downstream text-to-path CVAE model."""

from __future__ import annotations

from transformers import PretrainedConfig

from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


class SwipeTextToPathCVAEConfig(PretrainedConfig):
    model_type = "swipe_text2path_cvae"

    def __init__(
        self,
        *,
        encoder_config: dict | SwipeTransformerConfig | None = None,
        decoder_n_layers: int = 2,
        decoder_n_heads: int | None = None,
        decoder_d_ff: int | None = None,
        dropout: float | None = None,
        out_dim: int = 2,
        sigma_min: float = 1e-3,
        target_eps: float = 1e-4,
        path_loss_radial_weight: float = 0.0,
        path_sigma_target_min: float = 0.1,
        path_sigma_target_max: float = 0.4,
        uncertainty_reg_weight: float = 0.0,
        latent_dim: int = 16,
        latent_hidden_dim: int | None = None,
        kl_weight: float = 0.1,
        smoothness_weight: float = 0.0,
        smoothness_order: int = 2,
        speed_smoothness_weight: float = 0.0,
        spline_enabled: bool = False,
        spline_num_ctrl: int = 16,
        spline_degree: int = 3,
        spline_adaptive: bool = False,
        spline_min_ctrl: int = 12,
        spline_max_ctrl: int = 32,
        spline_ctrl_per_char: float = 0.5,
        path_encoder_n_layers: int = 2,
        path_encoder_n_heads: int | None = None,
        path_encoder_d_ff: int | None = None,
        path_encoder_dropout: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_cfg = SwipeTransformerConfig()
        elif isinstance(encoder_config, SwipeTransformerConfig):
            encoder_cfg = encoder_config
        elif isinstance(encoder_config, dict):
            encoder_cfg = SwipeTransformerConfig(**encoder_config)
        else:
            raise TypeError("encoder_config must be a dict, SwipeTransformerConfig, or None")

        self.encoder_config = encoder_cfg.to_dict()

        self.decoder_n_layers = int(decoder_n_layers)
        self.decoder_n_heads = (
            int(decoder_n_heads) if decoder_n_heads is not None else int(encoder_cfg.n_heads)
        )
        self.decoder_d_ff = int(decoder_d_ff) if decoder_d_ff is not None else int(encoder_cfg.d_ff)
        self.dropout = float(dropout) if dropout is not None else float(encoder_cfg.dropout)

        self.out_dim = int(out_dim)
        self.sigma_min = float(sigma_min)
        self.target_eps = float(target_eps)
        self.path_loss_radial_weight = float(path_loss_radial_weight)
        self.path_sigma_target_min = float(path_sigma_target_min)
        self.path_sigma_target_max = float(path_sigma_target_max)
        self.uncertainty_reg_weight = float(uncertainty_reg_weight)

        self.latent_dim = int(latent_dim)
        self.latent_hidden_dim = (
            int(latent_hidden_dim) if latent_hidden_dim is not None else int(encoder_cfg.d_model)
        )
        self.kl_weight = float(kl_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.smoothness_order = int(smoothness_order)
        self.speed_smoothness_weight = float(speed_smoothness_weight)
        self.spline_enabled = bool(spline_enabled)
        self.spline_num_ctrl = int(spline_num_ctrl)
        self.spline_degree = int(spline_degree)
        self.spline_adaptive = bool(spline_adaptive)
        self.spline_min_ctrl = int(spline_min_ctrl)
        self.spline_max_ctrl = int(spline_max_ctrl)
        self.spline_ctrl_per_char = float(spline_ctrl_per_char)

        self.path_encoder_n_layers = int(path_encoder_n_layers)
        self.path_encoder_n_heads = (
            int(path_encoder_n_heads)
            if path_encoder_n_heads is not None
            else int(self.decoder_n_heads)
        )
        self.path_encoder_d_ff = (
            int(path_encoder_d_ff) if path_encoder_d_ff is not None else int(self.decoder_d_ff)
        )
        self.path_encoder_dropout = (
            float(path_encoder_dropout) if path_encoder_dropout is not None else float(self.dropout)
        )
