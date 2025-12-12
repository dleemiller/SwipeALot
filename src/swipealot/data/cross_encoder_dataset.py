"""Dataset for cross-encoder training with hard negative mining."""

import random
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .dataset import normalize_coordinates, sample_path_points
from .negative_mining import load_negative_pool
from .tokenizer import CharacterTokenizer


class CrossEncoderDataset(Dataset):
    """
    Dataset for cross-encoder training with Multiple Negatives Ranking Loss.

    Creates tuples: (path, positive_word, negative_1, ..., negative_n)
    Each sample contains 1 positive pair + N negative pairs.
    """

    def __init__(
        self,
        split: str = "train",
        max_path_len: int = 128,
        max_word_len: int = 48,
        tokenizer: CharacterTokenizer | None = None,
        dataset_name: str = "futo-org/swipe.futo.org",
        negative_pool_path: str | None = None,
        num_negatives: int = 3,
        difficulty_sampling: bool = True,
        max_samples: int | None = None,
    ):
        """
        Initialize cross-encoder dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_path_len: Maximum path length (points will be sampled/padded)
            max_word_len: Maximum word length in characters
            tokenizer: Character tokenizer (will be created if None)
            dataset_name: HuggingFace dataset name
            negative_pool_path: Path to negative pool JSON file
            num_negatives: Number of negatives per positive (default: 3)
            difficulty_sampling: Sample negatives by difficulty score
            max_samples: Optional limit on number of samples (for debugging)
        """
        self.max_path_len = max_path_len
        self.max_word_len = max_word_len
        self.num_negatives = num_negatives
        self.difficulty_sampling = difficulty_sampling

        # Load dataset
        print(f"Loading dataset: {dataset_name}, split: {split}")
        if max_samples:
            self.dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
        else:
            self.dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(self.dataset)} samples")

        # Initialize tokenizer
        if tokenizer is None:
            print("Building tokenizer from dataset...")
            self.tokenizer = CharacterTokenizer.from_dataset(self.dataset)
            print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        else:
            self.tokenizer = tokenizer

        # Load negative pool
        if negative_pool_path:
            print(f"Loading negative pool from: {negative_pool_path}")
            self.negative_pool = load_negative_pool(negative_pool_path)
            print(f"Loaded negatives for {len(self.negative_pool)} words")
        else:
            print("Warning: No negative pool provided. Will use random negatives.")
            self.negative_pool = None

        # Extract all unique words for random negative sampling fallback
        self.all_words = list(set(sample["word"].lower() for sample in self.dataset))
        print(f"Vocabulary: {len(self.all_words)} unique words")

    def _sample_negatives(self, positive_word: str) -> list[str]:
        """
        Sample negative words for a given positive.

        Args:
            positive_word: The correct word (lowercase)

        Returns:
            List of negative words
        """
        positive_word = positive_word.lower()

        # If negative pool exists, use difficulty-based sampling
        if self.negative_pool and positive_word in self.negative_pool:
            negatives_with_scores = self.negative_pool[positive_word]

            if not negatives_with_scores:
                # No negatives found, fallback to random
                return self._random_negatives(positive_word)

            if self.difficulty_sampling:
                # Sample by difficulty score (higher score = higher probability)
                words = [neg for neg, _ in negatives_with_scores]
                scores = [score for _, score in negatives_with_scores]

                # Normalize scores to probabilities
                total_score = sum(scores)
                if total_score > 0:
                    probs = [s / total_score for s in scores]
                else:
                    probs = [1.0 / len(scores)] * len(scores)

                # Sample without replacement
                num_to_sample = min(self.num_negatives, len(words))
                sampled = np.random.choice(words, size=num_to_sample, replace=False, p=probs)
                negatives = list(sampled)
            else:
                # Uniform sampling from pool
                words = [neg for neg, _ in negatives_with_scores]
                num_to_sample = min(self.num_negatives, len(words))
                negatives = random.sample(words, num_to_sample)

            # If not enough negatives, fill with random
            while len(negatives) < self.num_negatives:
                random_neg = random.choice(self.all_words)
                if random_neg != positive_word and random_neg not in negatives:
                    negatives.append(random_neg)

            return negatives[: self.num_negatives]

        else:
            # Fallback to random negatives
            return self._random_negatives(positive_word)

    def _random_negatives(self, positive_word: str) -> list[str]:
        """Sample random negative words."""
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.choice(self.all_words)
            if neg != positive_word and neg not in negatives:
                negatives.append(neg)
        return negatives

    def _process_word(self, word: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process word into char tokens.

        Args:
            word: Word string

        Returns:
            (char_tokens, char_mask) tensors
        """
        char_tokens = self.tokenizer.encode(word.lower())

        # Add EOS token
        char_tokens = char_tokens + [self.tokenizer.eos_token_id]

        # Pad or truncate
        if len(char_tokens) < self.max_word_len:
            char_mask = [1] * len(char_tokens) + [0] * (self.max_word_len - len(char_tokens))
            char_tokens = char_tokens + [self.tokenizer.pad_token_id] * (
                self.max_word_len - len(char_tokens)
            )
        else:
            char_tokens = char_tokens[: self.max_word_len - 1] + [self.tokenizer.eos_token_id]
            char_mask = [1] * self.max_word_len

        return (
            torch.tensor(char_tokens, dtype=torch.long),
            torch.tensor(char_mask, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a training sample with 1 positive + N negatives.

        Returns:
            Dictionary with:
                - path_coords: [max_path_len, 3]
                - path_mask: [max_path_len]
                - positive_word: [max_word_len]
                - positive_mask: [max_word_len]
                - negative_words: [num_negatives, max_word_len]
                - negative_masks: [num_negatives, max_word_len]
                - original_word: str (for debugging)
        """
        sample = self.dataset[idx]

        # Process swipe path (shared across all pairs)
        data_points = sample["data"]
        canvas_width = sample.get("canvas_width", 1.0)
        canvas_height = sample.get("canvas_height", 1.0)

        normalized_points = normalize_coordinates(data_points, canvas_width, canvas_height)
        path_coords, path_mask = sample_path_points(normalized_points, self.max_path_len)

        # Get positive word
        positive_word = sample["word"]
        positive_tokens, positive_mask = self._process_word(positive_word)

        # Sample negative words
        negative_words = self._sample_negatives(positive_word)

        # Process negatives
        negative_tokens_list = []
        negative_masks_list = []
        for neg_word in negative_words:
            neg_tokens, neg_mask = self._process_word(neg_word)
            negative_tokens_list.append(neg_tokens)
            negative_masks_list.append(neg_mask)

        # Stack negatives
        negative_tokens = torch.stack(negative_tokens_list)  # [num_negatives, max_word_len]
        negative_masks = torch.stack(negative_masks_list)  # [num_negatives, max_word_len]

        return {
            "path_coords": torch.tensor(path_coords, dtype=torch.float32),
            "path_mask": torch.tensor(path_mask, dtype=torch.long),
            "positive_word": positive_tokens,
            "positive_mask": positive_mask,
            "negative_words": negative_tokens,
            "negative_masks": negative_masks,
            "original_word": positive_word.lower(),
            "sampled_negatives": negative_words,  # For debugging
        }
