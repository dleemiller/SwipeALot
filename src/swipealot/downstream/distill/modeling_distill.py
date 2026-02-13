"""Downstream model: distillation bottleneck for TCN training.

Architecture:
    Path [B, 128, 8] + Chars [B, 48] (masked)
        -> SwipeALot encoder -> [B, seq_len, 768]
        -> extract path token positions [B, 128, 768]
        -> Projector Linear(768, D) -> [B, 128, D] -> transpose -> [B, D, 128]
        -> TemporalAdapter (stride-2, constant channels) -> [B, D, 32]
        -> BiLSTM (no input_proj) -> CTC logits -> CTC loss

    No path_features concatenation: the projector output D channels already
    encode all path information via the transformer. Keeping D constant through
    the adapter avoids wasteful parameters for mobile deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from swipealot.downstream.distill.configuration_distill import SwipeDistillConfig
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig
from swipealot.huggingface.modeling_swipe import SwipeTransformerModel


@dataclass
class SwipeDistillOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None  # [B, T', num_chars+1] CTC logits
    projected: torch.FloatTensor | None = None  # [B, D, 128] projector output (distillation target)
    encoder_last_hidden_state: torch.FloatTensor | None = None


class TemporalAdapter(nn.Module):
    """Temporal downsampling adapter (matches encodercnn TemporalAdapter)."""

    def __init__(
        self,
        input_channels: int,
        num_stages: int = 2,
        kernel_size: int = 5,
        stride: int = 2,
        double_channels: bool = False,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.stride = stride

        layers = []
        in_ch = input_channels
        for _ in range(num_stages):
            out_ch = in_ch * 2 if double_channels else in_ch
            padding = (kernel_size - 1) // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch

        self.output_channels = in_ch
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CTCDecoder(nn.Module):
    """BiLSTM/BiGRU decoder with CTC head (matches encodercnn DirectCTCDecoder)."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        num_chars: int = 26,
    ):
        super().__init__()
        if input_dim != hidden_size:
            self.input_proj = nn.Linear(input_dim, hidden_size)
        else:
            self.input_proj = nn.Identity()
        self.input_norm = nn.LayerNorm(hidden_size)

        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        self.ctc_head = nn.Linear(rnn_output_dim, num_chars + 1)  # +1 for blank

        self._init_weights()

    def _init_weights(self):
        if isinstance(self.input_proj, nn.Linear):
            nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
            nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.ctc_head.weight, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward from channel-first input.

        Args:
            x: [B, C, T] from temporal adapter

        Returns:
            logits: [B, T, num_chars+1]
        """
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.input_proj(x)
        x = self.input_norm(x)
        x, _ = self.rnn(x)
        return self.ctc_head(x)


class SwipeDistillModel(PreTrainedModel):
    config_class = SwipeDistillConfig

    def __init__(self, config: SwipeDistillConfig):
        super().__init__(config)

        encoder_cfg = SwipeTransformerConfig(**config.encoder_config)
        self.encoder = SwipeTransformerModel(encoder_cfg)

        d_model = int(encoder_cfg.d_model)
        self.max_path_len = int(encoder_cfg.max_path_len)

        # Projector: 768 -> D
        self.projector = nn.Linear(d_model, config.projector_dim)

        # Temporal adapter: [B, D, 128] -> [B, D, T'] (constant channels)
        self.temporal_adapter = TemporalAdapter(
            input_channels=config.projector_dim,
            num_stages=config.adapter_num_stages,
            kernel_size=config.adapter_kernel_size,
            stride=config.adapter_stride,
            double_channels=False,
        )

        # CTC decoder (no input_proj when adapter output == rnn_hidden)
        self.ctc_decoder = CTCDecoder(
            input_dim=self.temporal_adapter.output_channels,
            hidden_size=config.rnn_hidden,
            num_layers=config.rnn_layers,
            bidirectional=config.rnn_bidirectional,
            rnn_type=config.rnn_type,
            dropout=config.rnn_dropout,
            num_chars=config.num_chars,
        )

        self.ctc_loss_fn = nn.CTCLoss(blank=config.blank_idx, reduction="mean", zero_infinity=True)

        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        encoder_path: str,
        *,
        config: SwipeDistillConfig | None = None,
        freeze_encoder: bool = False,
        **kwargs,
    ) -> SwipeDistillModel:
        encoder = SwipeTransformerModel.from_pretrained(encoder_path, **kwargs)
        encoder_cfg = encoder.config

        if config is None:
            config = SwipeDistillConfig(encoder_config=encoder_cfg)
        else:
            config.encoder_config = encoder_cfg.to_dict()

        model = cls(config)
        model.encoder = encoder
        if freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
        return model

    def _build_full_attention_mask(
        self,
        *,
        input_ids: torch.Tensor,
        path_coords: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build full mixed attention mask [B, 1+P+1+C]."""
        batch_size = input_ids.shape[0]
        char_len = input_ids.shape[1]
        full_len = 1 + self.max_path_len + 1 + char_len

        if attention_mask is not None and attention_mask.shape[1] == full_len:
            return attention_mask

        device = input_ids.device
        dtype = torch.long

        # Text mask from padding
        if attention_mask is not None:
            text_mask = attention_mask
        else:
            pad_id = int(self.encoder.config.pad_token_id)
            text_mask = input_ids.ne(pad_id).long()

        # Full mask: [CLS=1] + [path=1] + [SEP=1] + [text mask]
        cls_mask = torch.ones((batch_size, 1), dtype=dtype, device=device)
        path_mask = torch.ones((batch_size, self.max_path_len), dtype=dtype, device=device)
        sep_mask = torch.ones((batch_size, 1), dtype=dtype, device=device)
        return torch.cat([cls_mask, path_mask, sep_mask, text_mask], dim=1)

    def _mask_text_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask all non-pad text tokens for path-only modality.

        Args:
            input_ids: [B, C] character token IDs

        Returns:
            masked_ids: [B, C] with all non-pad tokens replaced by mask_token_id
        """
        mask_token_id = int(getattr(self.encoder.config, "mask_token_id", 5))
        pad_token_id = int(self.encoder.config.pad_token_id)

        is_pad = input_ids.eq(pad_token_id)
        masked = input_ids.clone()
        masked[~is_pad] = mask_token_id
        return masked

    def forward(
        self,
        *,
        path_coords: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            path_coords: [B, 128, 8] path features for encoder input
            input_ids: [B, C] character tokens (will be masked)
            attention_mask: [B, C] or [B, 1+P+1+C] attention mask
            labels: [B, max_label_len] CTC target character indices (0-25 for a-z)
            label_lengths: [B] actual length of each label
            **kwargs: absorbs extra collator keys (path_features, words)
        """
        return_dict = (
            bool(return_dict) if return_dict is not None else bool(self.config.use_return_dict)
        )

        # Mask all text input (path-only modality)
        masked_ids = self._mask_text_input(input_ids)

        # Build full attention mask
        full_attention_mask = self._build_full_attention_mask(
            input_ids=masked_ids, path_coords=path_coords, attention_mask=attention_mask
        )

        # Encoder forward
        enc_out = self.encoder(
            input_ids=masked_ids,
            path_coords=path_coords,
            attention_mask=full_attention_mask,
            return_dict=True,
        )
        encoder_hidden = enc_out.last_hidden_state  # [B, 1+P+1+C, d_model]

        # Extract path token representations [B, 128, d_model]
        path_reps = encoder_hidden[:, 1 : 1 + self.max_path_len, :]

        # Project [B, 128, d_model] -> [B, 128, D] -> transpose -> [B, D, 128]
        projected = self.projector(path_reps).transpose(1, 2)

        # Temporal adapter: [B, D, 128] -> [B, D, T'] (constant channels)
        adapted = self.temporal_adapter(projected)

        # CTC decoder: [B, D, T'] -> [B, T', num_chars+1]
        logits = self.ctc_decoder(adapted)

        # Compute CTC loss
        loss = None
        if labels is not None and label_lengths is not None:
            log_probs = torch.log_softmax(logits, dim=-1)  # [B, T', num_chars+1]
            log_probs = log_probs.transpose(0, 1)  # [T', B, num_chars+1] for CTC

            input_lengths = torch.full(
                (log_probs.shape[1],),
                log_probs.shape[0],
                dtype=torch.long,
                device=logits.device,
            )

            loss = self.ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

        if not return_dict:
            return loss, logits, projected, encoder_hidden

        return SwipeDistillOutput(
            loss=loss,
            logits=logits,
            projected=projected,
            encoder_last_hidden_state=encoder_hidden,
        )

    def get_encoder_params(self) -> list[nn.Parameter]:
        """Get encoder parameters (for low LR)."""
        return list(self.encoder.parameters())

    def get_new_params(self) -> list[nn.Parameter]:
        """Get projector + adapter + decoder parameters (for normal LR)."""
        params = []
        params.extend(self.projector.parameters())
        params.extend(self.temporal_adapter.parameters())
        params.extend(self.ctc_decoder.parameters())
        return params

    def get_adapter_decoder_state_dict(self) -> dict:
        """Extract temporal adapter + CTC decoder state dicts for Phase 3 init."""
        state = {}
        for k, v in self.temporal_adapter.state_dict().items():
            state[f"temporal_adapter.{k}"] = v
        for k, v in self.ctc_decoder.state_dict().items():
            state[f"ctc_decoder.{k}"] = v
        return state
