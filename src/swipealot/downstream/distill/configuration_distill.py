"""Configuration for the downstream distillation model."""

from __future__ import annotations

from transformers import PretrainedConfig

from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


class SwipeDistillConfig(PretrainedConfig):
    model_type = "swipe_distill"

    def __init__(
        self,
        *,
        encoder_config: dict | SwipeTransformerConfig | None = None,
        projector_dim: int = 128,
        adapter_num_stages: int = 2,
        adapter_kernel_size: int = 5,
        adapter_stride: int = 2,
        rnn_type: str = "lstm",
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        rnn_bidirectional: bool = True,
        rnn_dropout: float = 0.1,
        num_chars: int = 26,
        blank_idx: int = 26,
        encoder_lr_scale: float = 0.1,
        text_mask_prob: float = 1.0,
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
        self.adapter_num_stages = int(adapter_num_stages)
        self.adapter_kernel_size = int(adapter_kernel_size)
        self.adapter_stride = int(adapter_stride)
        self.rnn_type = str(rnn_type)
        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = int(rnn_layers)
        self.rnn_bidirectional = bool(rnn_bidirectional)
        self.rnn_dropout = float(rnn_dropout)
        self.num_chars = int(num_chars)
        self.blank_idx = int(blank_idx)
        self.encoder_lr_scale = float(encoder_lr_scale)
        self.text_mask_prob = float(text_mask_prob)
