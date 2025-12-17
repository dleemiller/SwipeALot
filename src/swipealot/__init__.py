"""Swipe keyboard transformer model.

Note: SwipeTransformerModel is now available from swipealot.huggingface
The native PyTorch version has been archived.
"""

__version__ = "0.1.0"

# Configuration
from swipealot.config import Config, DataConfig, ModelConfig, TrainingConfig

# Data
from swipealot.data import CharacterTokenizer, SwipeDataset

# Training
from swipealot.training import SwipeLoss, SwipeTrainer

# HuggingFace integration (recommended)
# from swipealot.huggingface import SwipeTransformerModel

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "CharacterTokenizer",
    "SwipeDataset",
    "SwipeTrainer",
    "SwipeLoss",
]
