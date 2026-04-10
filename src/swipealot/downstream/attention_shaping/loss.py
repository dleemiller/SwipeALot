"""Loss functions for attention distillation.

Three orthogonal objectives:
1. Temporal Envelope Loss - WHEN and HOW MUCH total attention occurs (Huber)
2. Character Distribution Loss - WHICH character at each timestep (sum-normalized KL)
3. Valid Character Loss - WHERE to place peaks for valid chars (Huber, masked)

Ported from encodercnn AttentionDistillationLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDistillationLoss(nn.Module):
    """Loss for attention distillation.

    loss = temporal_huber + char_kl_weight * char_kl + valid_huber_weight * valid_huber

    The model outputs non-negative attention values via softplus: [B, 26, 128]
    Target attention is also non-negative: [B, 26, 128] (0 for invalid chars)
    """

    def __init__(
        self,
        huber_beta: float = 0.01,
        char_kl_weight: float = 0.2,
        valid_huber_weight: float = 5.0,
        min_pred_sum: float = 0.01,
    ):
        super().__init__()
        self.huber_beta = huber_beta
        self.char_kl_weight = char_kl_weight
        self.valid_huber_weight = valid_huber_weight
        self.min_pred_sum = min_pred_sum

    def _temporal_envelope_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss on total attention per timestep.

        Captures WHEN attention should occur and HOW MUCH total.

        Args:
            pred: [B, 26, 128] predicted attention
            target: [B, 26, 128] target attention

        Returns:
            Scalar Huber loss on temporal envelopes
        """
        pred_sum = pred.sum(dim=1)  # [B, 128]
        target_sum = target.sum(dim=1)  # [B, 128]
        return F.smooth_l1_loss(pred_sum, target_sum, beta=self.huber_beta)

    def _char_distribution_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Sum-normalized KL divergence per timestep.

        Captures WHICH character should receive attention at each timestep.
        Uses sum-normalization (not softmax) to preserve sparsity.

        NOTE: Computed in float32 because bf16 cannot represent eps=1e-8
        and the log-ratio differences collapse to zero.

        Args:
            pred: [B, 26, 128] predicted attention
            target: [B, 26, 128] target attention

        Returns:
            Scalar KL divergence averaged over valid timesteps
        """
        eps = 1e-8

        # Cast to float32 for numerical stability in KL computation
        pred_f = pred.float()
        target_f = target.float()

        # Normalize target over chars at each timestep
        target_sum = target_f.sum(dim=1, keepdim=True)  # [B, 1, 128]
        target_dist = target_f / (target_sum + eps)  # [B, 26, 128]

        # Normalize prediction with floor for numerical stability
        pred_sum = pred_f.sum(dim=1, keepdim=True).clamp(min=self.min_pred_sum)
        pred_dist = pred_f / pred_sum  # [B, 26, 128]

        # KL divergence: sum_c target[c] * log(target[c] / pred[c])
        kl = target_dist * (torch.log(target_dist + eps) - torch.log(pred_dist + eps))
        kl = kl.sum(dim=1)  # [B, 128]

        # Mask timesteps with no target attention
        mask = (target_sum.squeeze(1) > eps).float()  # [B, 128]
        return (kl * mask).sum() / (mask.sum() + eps)

    def _valid_char_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss on valid characters only.

        Captures WHERE to place attention peaks for each character.
        Only supervises characters that are in the word.

        Args:
            pred: [B, 26, 128] predicted attention
            target: [B, 26, 128] target attention
            char_mask: [B, 26] boolean mask of valid characters

        Returns:
            Scalar Huber loss averaged over valid char-timestep pairs
        """
        mask = char_mask.unsqueeze(-1).float()  # [B, 26, 1]

        diff = torch.abs(pred - target)
        huber = torch.where(
            diff < self.huber_beta,
            0.5 * diff**2 / self.huber_beta,
            diff - 0.5 * self.huber_beta,
        )
        masked_huber = huber * mask  # [B, 26, 128]

        num_valid = mask.sum() * pred.shape[-1]
        return masked_huber.sum() / (num_valid + 1e-8)

    def forward(
        self,
        pred_attention: torch.Tensor,
        target_attention: torch.Tensor,
        char_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute loss.

        Args:
            pred_attention: [B, 26, 128] predicted attention (softplus output)
            target_attention: [B, 26, 128] target attention
            char_mask: [B, 26] mask of valid characters (required for valid_huber)

        Returns:
            Dictionary with 'loss' and component breakdowns
        """
        temporal_loss = self._temporal_envelope_loss(pred_attention, target_attention)
        char_kl_loss = self._char_distribution_loss(pred_attention, target_attention)

        loss = temporal_loss + self.char_kl_weight * char_kl_loss

        result = {
            "temporal_loss": temporal_loss.item(),
            "char_kl_loss": char_kl_loss.item(),
        }

        if char_mask is not None and self.valid_huber_weight > 0:
            valid_huber_loss = self._valid_char_huber_loss(
                pred_attention, target_attention, char_mask
            )
            loss = loss + self.valid_huber_weight * valid_huber_loss
            result["valid_huber_loss"] = valid_huber_loss.item()

        result["loss"] = loss
        return result
