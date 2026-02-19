"""Attention shaping model: self-distill encoder attention into path representations.

Architecture:
    Teacher (frozen, unmasked text, output_attentions=True):
        -> char->path attention from target layers -> [B, 26, 128] targets

    Student (trainable, masked text):
        -> path reps [B, 128, 768] -> projector -> LayerNorm -> Conv1d
        -> softplus -> [B, 26, 128] prediction

    Loss: temporal_huber + char_kl + valid_char_huber + length_loss (CLS)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from swipealot.analysis.attention_extractor import compute_char_to_path_attention_profile
from swipealot.downstream.attention_shaping.configuration import SwipeAttentionShapingConfig
from swipealot.downstream.attention_shaping.loss import AttentionDistillationLoss
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig
from swipealot.huggingface.modeling_swipe import SwipeTransformerModel

# Letter token IDs: a=17, b=18, ..., z=42
_LETTER_ID_START = 17
_LETTER_ID_END = 42
_NUM_LETTERS = 26


@dataclass
class SwipeAttentionShapingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    pred_attention: torch.FloatTensor | None = None  # [B, 26, 128]
    target_attention: torch.FloatTensor | None = None  # [B, 26, 128]
    temporal_loss: float | None = None
    char_kl_loss: float | None = None
    valid_huber_loss: float | None = None
    length_loss: float | None = None


class SwipeAttentionShapingModel(PreTrainedModel):
    config_class = SwipeAttentionShapingConfig

    def __init__(self, config: SwipeAttentionShapingConfig):
        super().__init__(config)

        encoder_cfg = SwipeTransformerConfig(**config.encoder_config)
        self.teacher = SwipeTransformerModel(encoder_cfg)
        self.student = SwipeTransformerModel(encoder_cfg)

        d_model = int(encoder_cfg.d_model)
        self.max_path_len = int(encoder_cfg.max_path_len)
        self.max_char_len = int(encoder_cfg.max_char_len)

        self.projector = nn.Linear(d_model, config.projector_dim)
        self.projector_norm = nn.LayerNorm(config.projector_dim)
        self.prediction_head = nn.Conv1d(config.projector_dim, _NUM_LETTERS, kernel_size=1)

        self.loss_fn = AttentionDistillationLoss(
            huber_beta=config.huber_beta,
            char_kl_weight=config.char_kl_weight,
            valid_huber_weight=config.valid_huber_weight,
        )

        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        encoder_path: str,
        *,
        config: SwipeAttentionShapingConfig | None = None,
        **kwargs,
    ) -> SwipeAttentionShapingModel:
        encoder = SwipeTransformerModel.from_pretrained(encoder_path, **kwargs)
        encoder_cfg = encoder.config

        if config is None:
            config = SwipeAttentionShapingConfig(encoder_config=encoder_cfg)
        else:
            config.encoder_config = encoder_cfg.to_dict()

        model = cls(config)

        # Copy encoder weights into both teacher and student
        model.teacher.load_state_dict(encoder.state_dict())
        model.student.load_state_dict(encoder.state_dict())

        # Freeze teacher
        model.teacher.eval()
        for p in model.teacher.parameters():
            p.requires_grad = False

        return model

    def _build_full_attention_mask(
        self,
        *,
        input_ids: torch.Tensor,
        path_coords: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build full mixed attention mask [B, 1+P+1+C].

        Always rebuilds with CLS/PATH/SEP=1, using only the text portion
        from the incoming mask (which may be text-only or full-length).
        """
        batch_size = input_ids.shape[0]
        char_len = input_ids.shape[1]
        full_len = 1 + self.max_path_len + 1 + char_len
        device = input_ids.device

        if attention_mask is not None:
            if attention_mask.shape[1] == full_len:
                # Full-length mask — extract just the text portion
                text_mask = attention_mask[:, (1 + self.max_path_len + 1) :]
            else:
                text_mask = attention_mask
        else:
            pad_id = int(self.student.config.pad_token_id)
            text_mask = input_ids.ne(pad_id).long()

        cls_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        path_mask = torch.ones((batch_size, self.max_path_len), dtype=torch.long, device=device)
        sep_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        return torch.cat([cls_mask, path_mask, sep_mask, text_mask], dim=1)

    def _mask_text_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace ALL character positions with MASK (including PAD/EOS).

        Fills the entire [B, 48] text input with mask_token_id so the student
        cannot infer word length from the token pattern or attention mask.
        """
        mask_token_id = int(getattr(self.student.config, "mask_token_id", 3))
        return torch.full_like(input_ids, mask_token_id)

    def _extract_attention_targets(
        self,
        attentions: tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract char->path attention targets from teacher attentions.

        Uses compute_char_to_path_attention_profile on each target layer,
        then mean-aggregates. Maps char_len -> 26 letters via scatter_add.

        Returns:
            targets: [B, 26, 128] attention targets
            char_mask: [B, 26] which letters appear in each sample
        """
        target_layers = self.config.target_layers
        char_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Extract per-layer profiles and mean-aggregate
        profiles = []
        for layer_idx in target_layers:
            layer_attn = (attentions[layer_idx],)
            result = compute_char_to_path_attention_profile(
                layer_attn,
                path_len=self.max_path_len,
                char_len=char_len,
                head_aggregation="mean",
                layer_aggregation="last",
                renormalize_over_path=True,
            )
            profiles.append(result["profile"])  # [B, char_len, 128]

        # Mean across target layers -> [B, char_len, 128]
        profile = torch.stack(profiles, dim=0).mean(dim=0)

        # Map char_len -> 26 letters via scatter_add on token IDs
        # input_ids: [B, char_len] with token IDs (a=17..z=42)
        targets = torch.zeros(batch_size, _NUM_LETTERS, self.max_path_len, device=device)
        char_mask = torch.zeros(batch_size, _NUM_LETTERS, device=device, dtype=torch.bool)

        # Build letter index: token_id - 17 -> 0..25, -1 for non-letters
        letter_idx = input_ids - _LETTER_ID_START  # [B, char_len]
        is_letter = (letter_idx >= 0) & (letter_idx < _NUM_LETTERS)

        # For scatter_add, we need valid indices everywhere (use 0 for non-letters, mask later)
        safe_idx = letter_idx.clamp(0, _NUM_LETTERS - 1)  # [B, char_len]

        # scatter_add: profile [B, char_len, 128] -> targets [B, 26, 128]
        # Expand index for the path dimension
        scatter_idx = safe_idx.unsqueeze(-1).expand(-1, -1, self.max_path_len)  # [B, char_len, 128]

        # Zero out non-letter contributions before scatter
        masked_profile = profile * is_letter.unsqueeze(-1).float()
        targets.scatter_add_(1, scatter_idx, masked_profile)

        # Build char_mask: which of the 26 letters appear
        char_mask.scatter_(1, safe_idx, is_letter)

        return targets, char_mask

    def forward(
        self,
        *,
        path_coords: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = (
            bool(return_dict) if return_dict is not None else bool(self.config.use_return_dict)
        )

        # Build full attention mask
        full_attention_mask = self._build_full_attention_mask(
            input_ids=input_ids, path_coords=path_coords, attention_mask=attention_mask
        )

        # --- Teacher forward (frozen, unmasked, with attentions) ---
        self.teacher.eval()
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                path_coords=path_coords,
                attention_mask=full_attention_mask,
                output_attentions=True,
                return_dict=True,
            )
        targets, char_mask = self._extract_attention_targets(teacher_out.attentions, input_ids)

        # --- Student forward (fully masked text, all 48 positions attend) ---
        masked_ids = self._mask_text_input(input_ids)
        student_attention_mask = self._build_full_attention_mask(
            input_ids=masked_ids, path_coords=path_coords, attention_mask=None
        )

        student_out = self.student(
            input_ids=masked_ids,
            path_coords=path_coords,
            attention_mask=student_attention_mask,
            return_dict=True,
        )
        encoder_hidden = student_out.last_hidden_state  # [B, 1+P+1+C, d_model]

        # Extract path representations [B, 128, d_model]
        path_reps = encoder_hidden[:, 1 : 1 + self.max_path_len, :]

        # Project -> normalize -> transpose -> Conv1d -> softplus
        projected = self.projector_norm(self.projector(path_reps))  # [B, 128, D]
        projected = projected.transpose(1, 2)  # [B, D, 128]
        pred = self.prediction_head(projected)  # [B, 26, 128]
        pred = F.softplus(pred, beta=self.config.softplus_beta)  # [B, 26, 128]

        # Compute attention distillation loss
        loss_dict = self.loss_fn(pred, targets, char_mask)
        total_loss = loss_dict["loss"]

        # Length prediction loss (CLS token, continued from pretraining)
        length_loss_val = None
        if self.config.length_loss_weight > 0 and student_out.length_logits is not None:
            # Length target = number of letter tokens per sample
            letter_idx = input_ids - _LETTER_ID_START
            is_letter = (letter_idx >= 0) & (letter_idx < _NUM_LETTERS)
            length_target = is_letter.sum(dim=1).float()  # [B]

            length_loss = F.smooth_l1_loss(student_out.length_logits, length_target)
            total_loss = total_loss + self.config.length_loss_weight * length_loss
            length_loss_val = length_loss.item()

        if not return_dict:
            return total_loss, pred, targets

        return SwipeAttentionShapingOutput(
            loss=total_loss,
            pred_attention=pred,
            target_attention=targets,
            temporal_loss=loss_dict.get("temporal_loss"),
            char_kl_loss=loss_dict.get("char_kl_loss"),
            valid_huber_loss=loss_dict.get("valid_huber_loss"),
            length_loss=length_loss_val,
        )

    def get_new_params(self) -> list[nn.Parameter]:
        """Get projector + norm + prediction_head parameters (non-encoder)."""
        params = []
        params.extend(self.projector.parameters())
        params.extend(self.projector_norm.parameters())
        params.extend(self.prediction_head.parameters())
        return params
