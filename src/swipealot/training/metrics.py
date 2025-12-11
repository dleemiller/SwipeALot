"""Metrics for evaluating swipe keyboard model performance."""

import torch
from torchmetrics import Accuracy


class CharacterAccuracy:
    """Track character-level prediction accuracy using torchmetrics."""

    def __init__(self, vocab_size: int = 128, device: str = "cpu"):
        """
        Initialize metric.

        Args:
            vocab_size: Size of character vocabulary
            device: Device to place metric on
        """
        self.metric = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        ).to(device)
        self.device = device

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics.

        Args:
            predictions: [batch, seq_len, vocab_size] logits
            labels: [batch, seq_len] with -100 for non-masked positions
        """
        # Get predicted tokens
        pred_tokens = predictions.argmax(dim=-1)

        # Flatten for metric computation
        pred_flat = pred_tokens.flatten()
        labels_flat = labels.flatten()

        # Update torchmetrics
        self.metric.update(pred_flat, labels_flat)

    def compute(self) -> float:
        """Compute accuracy."""
        result = self.metric.compute()
        return result.item() if isinstance(result, torch.Tensor) else float(result)

    def reset(self):
        """Reset metrics."""
        self.metric.reset()

    def to(self, device):
        """Move metric to device."""
        self.device = device
        self.metric = self.metric.to(device)
        return self


class WordAccuracy:
    """Track word-level reconstruction accuracy."""

    def __init__(self, tokenizer):
        """
        Initialize metric.

        Args:
            tokenizer: Character tokenizer for decoding
        """
        self.tokenizer = tokenizer
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, original_words: list):
        """
        Update metrics.

        Args:
            predictions: [batch, char_len, vocab_size] character predictions
            original_words: List of original words
        """
        pred_tokens = predictions.argmax(dim=-1)  # [batch, char_len]

        for pred, original in zip(pred_tokens, original_words):
            # Decode and extract word up to EOS token
            pred_ids = pred.cpu().tolist()

            # Find EOS token if present
            eos_id = self.tokenizer.eos_token_id
            if eos_id in pred_ids:
                eos_idx = pred_ids.index(eos_id)
                pred_ids = pred_ids[:eos_idx]  # Exclude EOS from comparison

            pred_word = self.tokenizer.decode(pred_ids)
            pred_word = pred_word.strip()

            # Case-insensitive comparison (tokenizer is case-insensitive)
            if pred_word == original.strip().lower():
                self.correct += 1
            self.total += 1

    def compute(self) -> float:
        """Compute accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        """Reset metrics."""
        self.correct = 0
        self.total = 0
