"""Downstream module: distillation bottleneck for TCN training."""

from .configuration_distill import SwipeDistillConfig
from .modeling_distill import SwipeDistillModel, SwipeDistillOutput

__all__ = ["SwipeDistillConfig", "SwipeDistillModel", "SwipeDistillOutput"]
