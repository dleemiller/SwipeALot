"""Training infrastructure: loss, metrics, and HuggingFace trainer.

Note: The custom PyTorch trainer has been archived.
Now using HuggingFace Trainer with minimal SwipeTrainer subclass.
"""

from .checkpoint_utils import prepare_checkpoint_for_hub
from .loss import SwipeLoss
from .metrics import CharacterAccuracy, WordAccuracy
from .trainer import SwipeTrainer, create_compute_metrics_fn

__all__ = [
    "SwipeLoss",
    "CharacterAccuracy",
    "WordAccuracy",
    "SwipeTrainer",
    "create_compute_metrics_fn",
    "prepare_checkpoint_for_hub",
]
