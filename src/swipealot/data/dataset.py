"""SwipeDataset implementation.

Preprocessing utilities live in `swipealot.data.preprocessing`.
"""

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .preprocessing import (
    normalize_and_compute_features,
    preprocess_raw_path_to_sg_features,
    sample_path_points_with_features,
)
from .tokenizer import CharacterTokenizer

__all__ = [
    "normalize_and_compute_features",
    "sample_path_points_with_features",
    "SwipeDataset",
]


class SwipeDataset(Dataset):
    """Dataset for swipe keyboard data."""

    def __init__(
        self,
        split: str = "train",
        max_path_len: int = 64,
        max_word_len: int = 38,
        tokenizer: CharacterTokenizer | None = None,
        dataset_name: str = "futo-org/swipe.futo.org",
        max_samples: int | None = None,
        path_resample_mode: str = "time",
    ):
        """
        Initialize swipe dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_path_len: Maximum path length (points will be sampled/padded)
            max_word_len: Maximum word length in characters
            tokenizer: Character tokenizer (will be created if None)
            dataset_name: HuggingFace dataset name
            max_samples: Optional limit on number of samples (for debugging)
        """
        self.max_path_len = max_path_len
        self.max_word_len = max_word_len
        self.path_resample_mode = path_resample_mode

        # Load dataset
        print(f"Loading dataset: {dataset_name}, split: {split}")
        if max_samples:
            self.dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
        else:
            self.dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(self.dataset)} samples")

        # Initialize tokenizer (deterministic; no dataset-derived vocab)
        self.tokenizer = tokenizer if tokenizer is not None else CharacterTokenizer()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        # Process swipe path
        data_points = sample["data"]

        # Fast path: preprocess + resample in one pass (vectorized numpy).
        path_features, path_mask = preprocess_raw_path_to_sg_features(
            data_points,
            self.max_path_len,
            resample_mode=self.path_resample_mode,
        )

        # Process word
        word = sample["word"]
        char_tokens = self.tokenizer.encode(word)

        # Add EOS token after the word
        char_tokens = char_tokens + [self.tokenizer.eos_token_id]

        # Pad character tokens (EOS + padding)
        if len(char_tokens) < self.max_word_len:
            char_tokens = char_tokens + [self.tokenizer.pad_token_id] * (
                self.max_word_len - len(char_tokens)
            )
        else:
            # If word+EOS is too long, truncate word and ensure EOS at the end
            char_tokens = char_tokens[: self.max_word_len - 1] + [self.tokenizer.eos_token_id]

        # Create character mask (1 for real tokens + EOS, 0 for padding)
        char_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in char_tokens]

        return {
            "path_coords": torch.tensor(path_features, dtype=torch.float32),  # [max_path_len, 8]
            "char_tokens": torch.tensor(char_tokens, dtype=torch.long),  # [max_word_len]
            "path_mask": torch.tensor(path_mask, dtype=torch.long),  # [max_path_len]
            "char_mask": torch.tensor(char_mask, dtype=torch.long),  # [max_word_len]
            "word": word,  # Keep original word for evaluation
        }
