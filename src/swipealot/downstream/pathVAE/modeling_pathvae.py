"""Downstream model: generate a swipe path from text using a conditional VAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from swipealot.downstream.pathVAE.configuration_pathvae import SwipeTextToPathCVAEConfig
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig
from swipealot.huggingface.modeling_swipe import SwipeTransformerModel


@dataclass
class SwipeTextToPathCVAEOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    recon_loss: torch.FloatTensor | None = None
    kl_loss: torch.FloatTensor | None = None
    smoothness_loss: torch.FloatTensor | None = None
    speed_smoothness_loss: torch.FloatTensor | None = None
    uncertainty_penalty: torch.FloatTensor | None = None
    mu_logit: torch.FloatTensor | None = None  # [B, P, out_dim] in logit space
    log_sigma: torch.FloatTensor | None = None  # [B, P, out_dim] (log of stddev in logit space)
    path_xy: torch.FloatTensor | None = None  # [B, P, out_dim] sampled in [0,1]
    z: torch.FloatTensor | None = None  # [B, latent_dim]
    prior_mu: torch.FloatTensor | None = None
    prior_logvar: torch.FloatTensor | None = None
    post_mu: torch.FloatTensor | None = None
    post_logvar: torch.FloatTensor | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None  # [B, S, d_model]


def _logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x) - torch.log1p(-x)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def _kl_normal(
    mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl.sum(dim=-1)


def _sample_latent(mu: torch.Tensor, logvar: torch.Tensor, scale: float) -> torch.Tensor:
    std = torch.exp(0.5 * logvar) * scale
    eps = torch.randn_like(std)
    return mu + eps * std


def _bspline_basis_matrix(num_points: int, num_ctrl: int, degree: int) -> np.ndarray:
    if num_ctrl <= degree:
        raise ValueError("num_ctrl must be greater than spline degree")
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    internal_count = num_ctrl - degree - 1
    if internal_count > 0:
        internal = np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
        knots = np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)], axis=0)
    else:
        knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)], axis=0)

    basis = np.zeros((num_points, num_ctrl), dtype=np.float64)
    for i in range(num_ctrl):
        left = knots[i]
        right = knots[i + 1]
        basis[:, i] = np.where((t >= left) & (t < right), 1.0, 0.0)
    basis[t == 1.0, -1] = 1.0

    for p in range(1, degree + 1):
        next_basis = np.zeros_like(basis)
        for i in range(num_ctrl):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom1 > 0:
                term1 = ((t - knots[i]) / denom1) * basis[:, i]
            else:
                term1 = 0.0
            if denom2 > 0 and i + 1 < num_ctrl:
                term2 = ((knots[i + p + 1] - t) / denom2) * basis[:, i + 1]
            else:
                term2 = 0.0
            next_basis[:, i] = term1 + term2
        basis = next_basis

    return basis.astype(np.float32)


class SwipeTextToPathCVAEModel(PreTrainedModel):
    config_class = SwipeTextToPathCVAEConfig

    def __init__(self, config: SwipeTextToPathCVAEConfig):
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

        self.spline_enabled = bool(getattr(config, "spline_enabled", False))
        if self.spline_enabled:
            ctrl_max = int(
                max(
                    getattr(config, "spline_max_ctrl", 0),
                    getattr(config, "spline_num_ctrl", 16),
                )
            )
            self.ctrl_embed = nn.Embedding(ctrl_max, d_model)
            self.query_embed = None
        else:
            self.query_embed = nn.Embedding(max_path_len, d_model)
            self.ctrl_embed = None
        self.out_mu = nn.Linear(d_model, int(config.out_dim))
        self.out_log_sigma = nn.Linear(d_model, int(config.out_dim))
        self.recon_nll = nn.GaussianNLLLoss(reduction="none", full=True)

        self.path_point_embed = nn.Linear(int(config.out_dim), d_model)
        self.path_pos_embed = nn.Embedding(max_path_len, d_model)

        if int(config.path_encoder_n_layers) > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config.path_encoder_n_heads),
                dim_feedforward=int(config.path_encoder_d_ff),
                dropout=float(config.path_encoder_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.path_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=int(config.path_encoder_n_layers)
            )
        else:
            self.path_encoder = None

        latent_hidden = int(config.latent_hidden_dim)
        latent_dim = int(config.latent_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(d_model, latent_hidden),
            nn.GELU(),
            nn.Linear(latent_hidden, 2 * latent_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(d_model * 2, latent_hidden),
            nn.GELU(),
            nn.Linear(latent_hidden, 2 * latent_dim),
        )
        self.latent_to_tgt = nn.Linear(latent_dim, d_model)

        self._basis_cache: dict[tuple[int, int, int, torch.dtype], torch.Tensor] = {}

        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        encoder_path: str,
        *,
        config: SwipeTextToPathCVAEConfig | None = None,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> SwipeTextToPathCVAEModel:
        encoder = SwipeTransformerModel.from_pretrained(encoder_path, **kwargs)
        encoder_cfg = encoder.config

        if config is None:
            config = SwipeTextToPathCVAEConfig(encoder_config=encoder_cfg)
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
        del char_len
        return encoder_hidden, full_attention_mask

    def _num_ctrl_points(
        self, *, attention_mask: torch.Tensor | None, input_ids: torch.Tensor
    ) -> int:
        cfg = self.config
        num_ctrl = int(getattr(cfg, "spline_num_ctrl", 16))
        if not bool(getattr(cfg, "spline_adaptive", False)):
            return num_ctrl

        if attention_mask is None:
            pad_id = int(self.encoder.config.pad_token_id)
            attention_mask = input_ids.ne(pad_id).long()
        lengths = attention_mask.sum(dim=1).float()
        avg_len = float(lengths.mean().item())
        base = int(getattr(cfg, "spline_min_ctrl", 12))
        per_char = float(getattr(cfg, "spline_ctrl_per_char", 0.5))
        num_ctrl = int(round(base + per_char * avg_len))
        num_ctrl = max(int(getattr(cfg, "spline_min_ctrl", base)), num_ctrl)
        num_ctrl = min(int(getattr(cfg, "spline_max_ctrl", num_ctrl)), num_ctrl)
        return num_ctrl

    def _spline_basis(
        self, *, path_len: int, num_ctrl: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        degree = int(getattr(self.config, "spline_degree", 3))
        key = (path_len, num_ctrl, degree, dtype)
        cached = self._basis_cache.get(key)
        if cached is None:
            basis = _bspline_basis_matrix(path_len, num_ctrl, degree)
            cached = torch.tensor(basis, dtype=dtype)
            self._basis_cache[key] = cached
        return cached.to(device=device)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        path_coords: torch.Tensor | None = None,
        labels_xy: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        temperature: float | None = None,
        sample_latent: bool = True,
        latent_scale: float | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            input_ids: [B, C]
            attention_mask: either [B, C] (text-only) or [B, 1+P+1+C] (full mixed)
            path_coords: optional path tensor; for text-only use cases this is ignored and a zero
                path is synthesized so the encoder can run.
            labels_xy: optional training targets [B, P, 2] in [0, 1]
            labels_mask: optional [B, P] mask (1=supervise point, 0=ignore)
            temperature: optional sampling temperature for `path_xy`
            sample_latent: if False, use the latent mean instead of sampling
            latent_scale: optional scale applied to latent stddev when sampling
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
        text_context = _masked_mean(memory, memory_mask)

        prior_stats = self.prior_net(text_context)
        prior_mu, prior_logvar = prior_stats.chunk(2, dim=-1)

        post_mu = None
        post_logvar = None
        if labels_xy is not None:
            path_len = int(self.encoder.config.max_path_len)
            if labels_xy.shape[:2] != (batch_size, path_len):
                raise ValueError(
                    f"labels_xy must be [B, P, 2] with P={path_len}, got shape={tuple(labels_xy.shape)}"
                )
            path_emb = self.path_point_embed(labels_xy)
            pos = torch.arange(path_len, device=device)
            path_emb = path_emb + self.path_pos_embed(pos).unsqueeze(0)
            path_key_padding = labels_mask.eq(0) if labels_mask is not None else None
            if self.path_encoder is not None:
                path_emb = self.path_encoder(path_emb, src_key_padding_mask=path_key_padding)
            path_summary = _masked_mean(path_emb, labels_mask)

            post_stats = self.posterior_net(torch.cat([text_context, path_summary], dim=-1))
            post_mu, post_logvar = post_stats.chunk(2, dim=-1)

        latent_mu = post_mu if post_mu is not None else prior_mu
        latent_logvar = post_logvar if post_logvar is not None else prior_logvar
        if latent_mu is None or latent_logvar is None:
            raise RuntimeError("Failed to compute latent distribution.")

        scale = 1.0 if latent_scale is None else float(latent_scale)
        if sample_latent:
            z = _sample_latent(latent_mu, latent_logvar, scale)
        else:
            z = latent_mu

        path_len = int(self.encoder.config.max_path_len)
        ctrl_mu_logit = None
        ctrl_sigma_raw = None
        basis = None
        if self.spline_enabled:
            num_ctrl = self._num_ctrl_points(attention_mask=attention_mask, input_ids=input_ids)
            if self.ctrl_embed is None:
                raise RuntimeError("Spline decoding requires ctrl_embed to be initialized.")
            if num_ctrl > int(self.ctrl_embed.num_embeddings):
                raise ValueError("num_ctrl exceeds spline_max_ctrl.")
            pos = torch.arange(num_ctrl, device=device)
            tgt = self.ctrl_embed(pos).unsqueeze(0).expand(batch_size, -1, -1)
            tgt = tgt + self.latent_to_tgt(z).unsqueeze(1)

            dec = self.decoder(
                tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask
            )

            ctrl_mu_logit = self.out_mu(dec)
            ctrl_sigma_raw = self.out_log_sigma(dec)
            basis = self._spline_basis(
                path_len=path_len, num_ctrl=num_ctrl, device=device, dtype=ctrl_mu_logit.dtype
            )
            mu_logit = torch.einsum("pk,bkd->bpd", basis, ctrl_mu_logit)
            sigma_raw = torch.einsum("pk,bkd->bpd", basis, ctrl_sigma_raw)
            sigma = F.softplus(sigma_raw) + float(self.config.sigma_min)
            log_sigma = torch.log(sigma)
        else:
            pos = torch.arange(path_len, device=device)
            if self.query_embed is None:
                raise RuntimeError("query_embed is not initialized.")
            tgt = self.query_embed(pos).unsqueeze(0).expand(batch_size, -1, -1)
            tgt = tgt + self.latent_to_tgt(z).unsqueeze(1)

            dec = self.decoder(
                tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask
            )

            mu_logit = self.out_mu(dec)
            sigma = F.softplus(self.out_log_sigma(dec)) + float(self.config.sigma_min)
            log_sigma = torch.log(sigma)

        mu_xy = torch.sigmoid(mu_logit)

        recon_loss = None
        uncertainty_penalty = None
        if labels_xy is not None:
            eps = float(self.config.target_eps)
            y = labels_xy.clamp(min=eps, max=1.0 - eps)

            var = sigma * sigma
            nll = self.recon_nll(mu_xy, y, var)  # [B, P, D]

            if labels_mask is not None:
                weight = labels_mask.to(dtype=nll.dtype).unsqueeze(-1)
            else:
                weight = torch.ones_like(nll[..., :1])

            radial_weight = float(getattr(self.config, "path_loss_radial_weight", 0.0))
            if radial_weight != 0.0:
                center = torch.tensor([0.5, 0.5], device=y.device, dtype=y.dtype)
                diff = y[..., :2] - center
                dist = torch.sqrt((diff * diff).sum(dim=-1))
                max_dist = (0.5 * 0.5 + 0.5 * 0.5) ** 0.5
                if max_dist > 0:
                    dist = dist / max_dist
                weight = weight * (1.0 + radial_weight * dist).unsqueeze(-1)

            denom = weight.sum().clamp(min=1.0)
            recon_loss = (nll * weight).sum() / denom

            reg_weight = float(getattr(self.config, "uncertainty_reg_weight", 0.0))
            if reg_weight > 0.0:
                log_sigma_local = torch.log(sigma + 1e-8)
                log_sigma_min = torch.log(
                    torch.tensor(
                        float(getattr(self.config, "path_sigma_target_min", 0.1)),
                        device=log_sigma_local.device,
                    )
                )
                log_sigma_max = torch.log(
                    torch.tensor(
                        float(getattr(self.config, "path_sigma_target_max", 0.4)),
                        device=log_sigma_local.device,
                    )
                )
                too_low = torch.clamp(log_sigma_min - log_sigma_local, min=0.0).pow(2)
                too_high = torch.clamp(log_sigma_local - log_sigma_max, min=0.0).pow(2)
                penalty = too_low + too_high
                if labels_mask is not None:
                    mask = labels_mask.to(dtype=penalty.dtype).unsqueeze(-1)
                    denom = mask.sum().clamp(min=1.0)
                    uncertainty_penalty = (penalty * mask).sum() / denom
                else:
                    uncertainty_penalty = penalty.mean()

        kl_loss = None
        if post_mu is not None and post_logvar is not None:
            kl = _kl_normal(post_mu, post_logvar, prior_mu, prior_logvar)
            kl_loss = kl.mean()

        smoothness_loss = None
        smoothness_weight = float(getattr(self.config, "smoothness_weight", 0.0))
        if labels_xy is not None and smoothness_weight > 0.0:
            path_mean = torch.sigmoid(mu_logit)
            order = int(getattr(self.config, "smoothness_order", 2))
            if order == 1:
                diffs = path_mean[:, 1:, :] - path_mean[:, :-1, :]
                smooth = (diffs**2).sum(dim=-1)
                if labels_mask is not None:
                    mask = labels_mask[:, 1:] * labels_mask[:, :-1]
                else:
                    mask = None
            else:
                diffs = path_mean[:, 2:, :] - 2 * path_mean[:, 1:-1, :] + path_mean[:, :-2, :]
                smooth = (diffs**2).sum(dim=-1)
                if labels_mask is not None:
                    mask = labels_mask[:, 2:] * labels_mask[:, 1:-1] * labels_mask[:, :-2]
                else:
                    mask = None

            if mask is not None:
                mask = mask.to(dtype=smooth.dtype)
                denom = mask.sum().clamp(min=1.0)
                smoothness_loss = (smooth * mask).sum() / denom
            else:
                smoothness_loss = smooth.mean()

        speed_smoothness_loss = None
        speed_weight = float(getattr(self.config, "speed_smoothness_weight", 0.0))
        if labels_xy is not None and speed_weight > 0.0:
            path_mean = torch.sigmoid(mu_logit)
            delta = path_mean[:, 1:, :] - path_mean[:, :-1, :]
            speed = torch.sqrt((delta**2).sum(dim=-1) + 1e-6)
            speed_diff = speed[:, 1:] - speed[:, :-1]
            if labels_mask is not None:
                mask = labels_mask[:, 2:] * labels_mask[:, 1:-1] * labels_mask[:, :-2]
            else:
                mask = None
            if mask is not None:
                mask = mask.to(dtype=speed_diff.dtype)
                denom = mask.sum().clamp(min=1.0)
                speed_smoothness_loss = ((speed_diff**2) * mask).sum() / denom
            else:
                speed_smoothness_loss = (speed_diff**2).mean()

        loss = None
        if recon_loss is not None:
            loss = recon_loss
            if kl_loss is not None:
                loss = loss + float(self.config.kl_weight) * kl_loss
            if smoothness_loss is not None:
                loss = loss + smoothness_weight * smoothness_loss
            if speed_smoothness_loss is not None:
                loss = loss + speed_weight * speed_smoothness_loss
            if uncertainty_penalty is not None:
                loss = loss + float(self.config.uncertainty_reg_weight) * uncertainty_penalty

        sample_temp = 1.0 if temperature is None else float(temperature)
        if self.spline_enabled:
            if basis is None:
                raise RuntimeError("Spline sampling requires basis to be initialized.")
            eps_ctrl = torch.randn(
                (batch_size, int(basis.shape[1]), int(mu_logit.shape[-1])),
                device=mu_logit.device,
                dtype=mu_logit.dtype,
            )
            eps_path = torch.einsum("pk,bkd->bpd", basis, eps_ctrl)
            norm = torch.sqrt((basis * basis).sum(dim=1).clamp_min(1e-6)).to(
                device=eps_path.device, dtype=eps_path.dtype
            )
            eps_path = eps_path / norm.view(1, -1, 1)
            path_xy = mu_xy + sample_temp * sigma * eps_path
        else:
            eps_noise = torch.randn_like(mu_logit)
            path_xy = mu_xy + sample_temp * sigma * eps_noise
        path_xy = torch.clamp(path_xy, 0.0, 1.0)

        if not return_dict:
            return (
                loss,
                recon_loss,
                kl_loss,
                mu_logit,
                log_sigma,
                path_xy,
                z,
                prior_mu,
                prior_logvar,
                post_mu,
                post_logvar,
                encoder_hidden,
            )

        return SwipeTextToPathCVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            smoothness_loss=smoothness_loss,
            speed_smoothness_loss=speed_smoothness_loss,
            uncertainty_penalty=uncertainty_penalty,
            mu_logit=mu_logit,
            log_sigma=log_sigma,
            path_xy=path_xy,
            z=z,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            post_mu=post_mu,
            post_logvar=post_logvar,
            encoder_last_hidden_state=encoder_hidden,
        )

    @torch.no_grad()
    def generate_path(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        path_coords: torch.Tensor | None = None,
        temperature: float = 1.0,
        sample_latent: bool = True,
        latent_scale: float | None = None,
    ) -> torch.Tensor:
        out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            path_coords=path_coords,
            labels_xy=None,
            temperature=float(temperature),
            sample_latent=bool(sample_latent),
            latent_scale=latent_scale,
            return_dict=True,
        )
        assert out.path_xy is not None
        return out.path_xy
