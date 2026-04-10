"""Downstream module: attention shaping via self-distillation."""

from .configuration import SwipeAttentionShapingConfig
from .loss import AttentionDistillationLoss
from .modeling import SwipeAttentionShapingModel

__all__ = [
    "SwipeAttentionShapingConfig",
    "SwipeAttentionShapingModel",
    "AttentionDistillationLoss",
]
