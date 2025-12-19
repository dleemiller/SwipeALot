from __future__ import annotations

import random
from typing import Any

import torch

from swipealot.text_utils import swipable_length

from ..tokenizer import CharacterTokenizer
from .utils import mask_contiguous_blocks_1d


class MaskedCollator:
    """
    Collator that creates masked versions of characters and paths for MLM-style training.
    """

    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        char_mask_prob: float | tuple[float, float] = 0.15,
        path_mask_prob: float = 0.15,
        mask_path: bool = True,
        mask_vocab_only: bool = False,
        path_mask_block_max_len: int = 32,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Character tokenizer for masking
            char_mask_prob: Probability of masking each character.
                          Can be a float (fixed probability) or tuple (min, max)
                          to randomly sample probability per batch from range.
            path_mask_prob: Probability of masking each path point
            mask_path: Whether to mask path points
            mask_vocab_only: If True, only mask vocabulary tokens (a-z, 0-9),
                           never mask special tokens ([EOS], [PUNC], [UNK])
        """
        self.tokenizer = tokenizer
        self.char_mask_prob = char_mask_prob
        self.path_mask_prob = path_mask_prob
        self.mask_path = mask_path
        self.mask_vocab_only = mask_vocab_only
        self.path_mask_block_max_len = path_mask_block_max_len

        if isinstance(char_mask_prob, (tuple, list)):
            if len(char_mask_prob) != 2:
                raise ValueError("char_mask_prob range must have exactly 2 values (min, max)")
            self.char_mask_prob_min, self.char_mask_prob_max = char_mask_prob
            self.use_random_mask_prob = True
        else:
            self.use_random_mask_prob = False

        if self.mask_vocab_only:
            self.vocab_start_id = len(tokenizer.special_tokens)
            self.vocab_end_id = tokenizer.vocab_size

    def mask_characters(self, char_tokens: torch.Tensor, char_mask: torch.Tensor) -> tuple:
        batch_size, seq_len = char_tokens.shape
        masked_tokens = char_tokens.clone()
        labels = torch.full_like(char_tokens, -100)

        if self.use_random_mask_prob:
            char_mask_prob = random.uniform(self.char_mask_prob_min, self.char_mask_prob_max)
        else:
            char_mask_prob = self.char_mask_prob

        mask_decisions = torch.rand(batch_size, seq_len) < char_mask_prob
        mask_decisions = mask_decisions & (char_mask == 1)

        if self.mask_vocab_only:
            is_vocab_token = (char_tokens >= self.vocab_start_id) & (
                char_tokens < self.vocab_end_id
            )
            mask_decisions = mask_decisions & is_vocab_token

        labels[mask_decisions] = char_tokens[mask_decisions]

        bert_probs = torch.rand(batch_size, seq_len)

        use_mask_token = mask_decisions & (bert_probs < 0.8)
        masked_tokens[use_mask_token] = self.tokenizer.mask_token_id

        use_random_token = mask_decisions & (bert_probs >= 0.8) & (bert_probs < 0.9)
        num_random = int(use_random_token.sum().item())
        if num_random > 0:
            random_tokens = torch.randint(
                len(self.tokenizer.special_tokens),
                self.tokenizer.vocab_size,
                (num_random,),
                dtype=char_tokens.dtype,
            )
            masked_tokens[use_random_token] = random_tokens

        return masked_tokens, labels

    def mask_path_points(self, path_coords: torch.Tensor, path_mask: torch.Tensor) -> tuple:
        batch_size, seq_len, _ = path_coords.shape
        masked_coords = path_coords.clone()
        labels = path_coords.clone()

        mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for b in range(batch_size):
            n_valid = int(path_mask[b].sum().item())
            if n_valid == 0 or self.path_mask_prob <= 0.0:
                continue
            n_to_mask = int(round(float(self.path_mask_prob) * n_valid))
            if n_to_mask <= 0:
                continue
            mask_indices[b] = mask_contiguous_blocks_1d(
                path_mask[b],
                n_to_mask,
                max_block_len=self.path_mask_block_max_len,
                rng=random,
            )

        masked_coords[mask_indices.bool()] = 0.0
        return masked_coords, labels, mask_indices

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        path_coords = torch.stack([item["path_coords"] for item in batch])
        char_tokens = torch.stack([item["char_tokens"] for item in batch])
        path_mask = torch.stack([item["path_mask"] for item in batch])
        char_mask = torch.stack([item["char_mask"] for item in batch])

        masked_char_tokens, char_labels = self.mask_characters(char_tokens, char_mask)

        if self.mask_path:
            masked_path_coords, path_labels, path_mask_indices = self.mask_path_points(
                path_coords, path_mask
            )
        else:
            masked_path_coords = path_coords
            path_labels = None
            path_mask_indices = None

        batch_size = int(path_coords.shape[0])
        cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
        sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
        attention_mask = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=1)

        result: dict[str, Any] = {
            "path_coords": masked_path_coords,
            "input_ids": masked_char_tokens,
            "char_labels": char_labels,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "words": [item["word"] for item in batch],
        }

        if self.mask_path:
            result["path_labels"] = path_labels
            result["path_mask_indices"] = path_mask_indices

        max_len = int(char_tokens.shape[1])
        result["length_target"] = torch.tensor(
            [swipable_length(item["word"], max_len=max_len) for item in batch], dtype=torch.long
        )
        result["length_supervise_mask"] = torch.ones(len(batch), dtype=torch.long)

        return result  # type: ignore[return-value]
