"""Configuration for the attention shaping model."""

from __future__ import annotations

from transformers import PretrainedConfig

from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


class SwipeAttentionShapingConfig(PretrainedConfig):
    model_type = "swipe_attention_shaping"

    def __init__(
        self,
        *,
        encoder_config: dict | SwipeTransformerConfig | None = None,
        projector_dim: int = 128,
        target_layers: list[int] | None = None,
        softplus_beta: float = 1.0,
        huber_beta: float = 0.01,
        char_kl_weight: float = 0.2,
        valid_huber_weight: float = 5.0,
        length_loss_weight: float = 0.1,
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

        self.projector_dim = int(projector_dim)
        self.target_layers = target_layers if target_layers is not None else [4, 5, 6]
        self.softplus_beta = float(softplus_beta)
        self.huber_beta = float(huber_beta)
        self.char_kl_weight = float(char_kl_weight)
        self.valid_huber_weight = float(valid_huber_weight)
        self.length_loss_weight = float(length_loss_weight)
