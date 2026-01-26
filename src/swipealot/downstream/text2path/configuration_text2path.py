"""Configuration for the downstream text→path model."""

from __future__ import annotations

from transformers import PretrainedConfig

from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


class SwipeTextToPathConfig(PretrainedConfig):
    model_type = "swipe_text2path"

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
        scheduled_sampling_ratio: float = 0.0,
        scheduled_sampling_warmup_steps: int = 0,
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
        self.scheduled_sampling_ratio = float(scheduled_sampling_ratio)
        self.scheduled_sampling_warmup_steps = int(scheduled_sampling_warmup_steps)
