"""Downstream model: autoregressive swipe path generation from text."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from swipealot.downstream.text2path.configuration_text2path import SwipeTextToPathConfig
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig
from swipealot.huggingface.modeling_swipe import SwipeTransformerModel


@dataclass
class SwipeTextToPathOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    mu_logit: torch.FloatTensor | None = None  # [B, P, out_dim] in logit space
    log_sigma: torch.FloatTensor | None = None  # [B, P, out_dim] (log of stddev in logit space)
    path_xy: torch.FloatTensor | None = None  # [B, P, out_dim] sampled in [0,1]
    encoder_last_hidden_state: torch.FloatTensor | None = None  # [B, S, d_model]


def _logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x) - torch.log1p(-x)


class SwipeTextToPathModel(PreTrainedModel):
    config_class = SwipeTextToPathConfig

    def __init__(self, config: SwipeTextToPathConfig):
        super().__init__(config)

        encoder_cfg = SwipeTransformerConfig(**config.encoder_config)
        self.encoder = SwipeTransformerModel(encoder_cfg)

        d_model = int(encoder_cfg.d_model)
        max_path_len = int(encoder_cfg.max_path_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=int(config.decoder_n_heads),
            dim_feedforward=int(config.decoder_d_ff),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(config.decoder_n_layers))

        self.bos_embed = nn.Parameter(torch.zeros(d_model))
        self.path_point_embed = nn.Linear(int(config.out_dim), d_model)
        self.pos_embed = nn.Embedding(max_path_len, d_model)
        self.out_mu = nn.Linear(d_model, int(config.out_dim))
        self.out_log_sigma = nn.Linear(d_model, int(config.out_dim))

        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        encoder_path: str,
        *,
        config: SwipeTextToPathConfig | None = None,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> SwipeTextToPathModel:
        encoder = SwipeTransformerModel.from_pretrained(encoder_path, **kwargs)
        encoder_cfg = encoder.config

        if config is None:
            config = SwipeTextToPathConfig(encoder_config=encoder_cfg)
        else:
            # Ensure the downstream config matches the loaded encoder (sequence lengths/d_model/etc).
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
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Accept either full mixed attention_mask or text-only attention_mask."""
        if attention_mask is None:
            pad_id = int(self.encoder.config.pad_token_id)
            text_mask = input_ids.ne(pad_id).long()
            attention_mask = text_mask

        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must be [B, L], got shape={tuple(attention_mask.shape)}"
            )

        batch_size, mask_len = attention_mask.shape
        max_path_len = int(self.encoder.config.max_path_len)
        char_len = int(input_ids.shape[1])
        full_len = 1 + max_path_len + 1 + char_len

        if mask_len == full_len:
            return attention_mask

        if mask_len != char_len:
            raise ValueError(
                f"Got attention_mask length {mask_len}, expected either char_len={char_len} "
                f"or full_len={full_len}."
            )

        cls = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        sep = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        path_zeros = torch.zeros(
            (batch_size, max_path_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        return torch.cat([cls, path_zeros, sep, attention_mask], dim=1)

    def _encoder_memory(
        self, *, encoder_hidden: torch.Tensor, full_attention_mask: torch.Tensor, char_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_path_len = int(self.encoder.config.max_path_len)
        sep_pos = 1 + max_path_len
        memory = encoder_hidden[:, sep_pos : sep_pos + 1 + char_len, :]
        memory_mask = full_attention_mask[:, sep_pos : sep_pos + 1 + char_len]
        return memory, memory_mask

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        path_coords: torch.Tensor | None = None,
        labels_xy: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        tgt_in_xy: torch.Tensor | None = None,
        temperature: float | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            input_ids: [B, C]
            attention_mask: either [B, C] (text-only) or [B, 1+P+1+C] (full mixed)
            path_coords: optional path tensor; for text-only use cases this is ignored and a zero
                path is synthesized so the encoder can run.
            labels_xy: training targets [B, P, 2] in [0, 1] (used for teacher forcing)
            tgt_in_xy: optional [B, P-1, 2] for scheduled sampling (prev-point inputs)
            labels_mask: optional [B, P] mask (1=supervise point, 0=ignore)
            temperature: optional sampling temperature for `path_xy`
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        return_dict = (
            bool(return_dict) if return_dict is not None else bool(self.config.use_return_dict)
        )

        batch_size = int(input_ids.shape[0])
        device = input_ids.device

        if path_coords is None:
            path_coords = torch.zeros(
                (
                    batch_size,
                    int(self.encoder.config.max_path_len),
                    int(self.encoder.config.path_input_dim),
                ),
                dtype=torch.float32,
                device=device,
            )

        full_attention_mask = self._build_full_attention_mask(
            input_ids=input_ids, attention_mask=attention_mask
        )

        enc_out = self.encoder(
            input_ids=input_ids,
            path_coords=path_coords,
            attention_mask=full_attention_mask,
            return_dict=True,
            **kwargs,
        )
        encoder_hidden = enc_out.last_hidden_state
        if encoder_hidden is None:
            raise RuntimeError("Encoder did not return last_hidden_state")

        char_len = int(input_ids.shape[1])
        memory, memory_mask = self._encoder_memory(
            encoder_hidden=encoder_hidden,
            full_attention_mask=full_attention_mask,
            char_len=char_len,
        )
        memory_key_padding_mask = memory_mask.eq(0)

        path_len = int(self.encoder.config.max_path_len)
        if labels_xy is None:
            raise ValueError(
                "labels_xy is required for autoregressive training; use generate_path for inference."
            )
        if labels_xy.shape[:2] != (batch_size, path_len):
            raise ValueError(
                f"labels_xy must be [B, P, 2] with P={path_len}, got shape={tuple(labels_xy.shape)}"
            )

        bos = self.bos_embed.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        if path_len > 1:
            if tgt_in_xy is not None:
                if tgt_in_xy.shape != (batch_size, path_len - 1, int(self.config.out_dim)):
                    raise ValueError("tgt_in_xy must be [B, P-1, out_dim] with P=max_path_len.")
                prev_xy = tgt_in_xy
            else:
                prev_xy = labels_xy[:, :-1, :]
            prev_emb = self.path_point_embed(prev_xy)
            tgt = torch.cat([bos, prev_emb], dim=1)
        else:
            tgt = bos

        pos = torch.arange(path_len, device=device)
        tgt = tgt + self.pos_embed(pos).unsqueeze(0)

        tgt_key_padding_mask = None
        if labels_mask is not None:
            if labels_mask.shape != (batch_size, path_len):
                raise ValueError(
                    f"labels_mask must be [B, P] with P={path_len}, got shape={tuple(labels_mask.shape)}"
                )
            tgt_in_mask = torch.zeros_like(labels_mask)
            tgt_in_mask[:, 0] = 1
            if path_len > 1:
                tgt_in_mask[:, 1:] = labels_mask[:, :-1]
            tgt_key_padding_mask = tgt_in_mask.eq(0)

        tgt_mask = self._causal_mask(path_len, device=device)
        dec = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        mu_logit = self.out_mu(dec)
        sigma = F.softplus(self.out_log_sigma(dec)) + float(self.config.sigma_min)
        log_sigma = torch.log(sigma)

        loss = None
        eps = float(self.config.target_eps)
        y = labels_xy.clamp(min=eps, max=1.0 - eps)
        z = _logit(y)

        nll = 0.5 * ((z - mu_logit) / sigma) ** 2 + log_sigma
        nll = nll.sum(dim=-1)  # [B, P]

        if labels_mask is not None:
            mask = labels_mask.to(dtype=nll.dtype)
            denom = mask.sum().clamp(min=1.0)
            loss = (nll * mask).sum() / denom
        else:
            loss = nll.mean()

        sample_temp = 1.0 if temperature is None else float(temperature)
        eps_noise = torch.randn_like(mu_logit)
        sampled_logit = mu_logit + sample_temp * sigma * eps_noise
        path_xy = torch.sigmoid(sampled_logit)

        if not return_dict:
            return loss, mu_logit, log_sigma, path_xy, encoder_hidden

        return SwipeTextToPathOutput(
            loss=loss,
            mu_logit=mu_logit,
            log_sigma=log_sigma,
            path_xy=path_xy,
            encoder_last_hidden_state=encoder_hidden,
        )

    @torch.no_grad()
    def generate_path(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        seed_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids is required")

        batch_size = int(input_ids.shape[0])
        device = input_ids.device

        path_coords = torch.zeros(
            (
                batch_size,
                int(self.encoder.config.max_path_len),
                int(self.encoder.config.path_input_dim),
            ),
            dtype=torch.float32,
            device=device,
        )

        full_attention_mask = self._build_full_attention_mask(
            input_ids=input_ids, attention_mask=attention_mask
        )
        enc_out = self.encoder(
            input_ids=input_ids,
            path_coords=path_coords,
            attention_mask=full_attention_mask,
            return_dict=True,
        )
        encoder_hidden = enc_out.last_hidden_state
        if encoder_hidden is None:
            raise RuntimeError("Encoder did not return last_hidden_state")

        char_len = int(input_ids.shape[1])
        memory, memory_mask = self._encoder_memory(
            encoder_hidden=encoder_hidden,
            full_attention_mask=full_attention_mask,
            char_len=char_len,
        )
        memory_key_padding_mask = memory_mask.eq(0)

        path_len = int(self.encoder.config.max_path_len)
        bos = self.bos_embed.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        tgt_emb = bos
        generated = []

        sample_temp = float(temperature)
        burnin = 0
        if seed_xy is not None:
            if (
                seed_xy.dim() != 3
                or seed_xy.shape[0] != batch_size
                or seed_xy.shape[2] != int(self.config.out_dim)
            ):
                raise ValueError("seed_xy must be [B, K, out_dim]")
            burnin = int(seed_xy.shape[1])
            if burnin > path_len:
                raise ValueError("seed_xy length exceeds max_path_len")
            if burnin > 0:
                tgt_emb = torch.cat([tgt_emb, self.path_point_embed(seed_xy)], dim=1)
                generated.extend(seed_xy[:, i : i + 1, :] for i in range(burnin))
        for step in range(burnin, path_len):
            seq_len = int(tgt_emb.shape[1])
            pos = torch.arange(seq_len, device=device)
            tgt = tgt_emb + self.pos_embed(pos).unsqueeze(0)
            tgt_mask = self._causal_mask(seq_len, device=device)

            dec = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            dec_last = dec[:, -1:, :]
            mu_logit = self.out_mu(dec_last)
            sigma = F.softplus(self.out_log_sigma(dec_last)) + float(self.config.sigma_min)
            eps_noise = torch.randn_like(mu_logit)
            sampled_logit = mu_logit + sample_temp * sigma * eps_noise
            point = torch.sigmoid(sampled_logit)
            generated.append(point)

            if step + 1 < path_len:
                tgt_emb = torch.cat([tgt_emb, self.path_point_embed(point)], dim=1)

        return torch.cat(generated, dim=1)
