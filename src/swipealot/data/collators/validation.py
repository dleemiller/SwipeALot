from __future__ import annotations

from typing import Any

import torch

from swipealot.text_utils import swipable_length

from ..tokenizer import CharacterTokenizer


class ValidationCollator:
    """
    Collator for validation that doesn't apply masking.

    Evaluates the model's ability to predict all character positions
    from the full unmasked input, giving true reconstruction accuracy.
    """

    def __init__(self, tokenizer: CharacterTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        path_coords = torch.stack([item["path_coords"] for item in batch])
        char_tokens = torch.stack([item["char_tokens"] for item in batch])
        path_mask = torch.stack([item["path_mask"] for item in batch])
        char_mask = torch.stack([item["char_mask"] for item in batch])

        batch_size = int(char_tokens.shape[0])

        char_labels = char_tokens.clone()
        char_labels[char_mask == 0] = -100

        masked_char_tokens = char_tokens.clone()
        masked_char_tokens[char_mask == 1] = self.tokenizer.mask_token_id

        cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
        sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
        attention_mask = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=1)

        words = [item["word"] for item in batch]
        max_len = int(char_tokens.shape[1])
        length_target = torch.tensor(
            [swipable_length(w, max_len=max_len) for w in words], dtype=torch.long
        )

        return {
            "path_coords": path_coords,
            "input_ids": masked_char_tokens,
            "char_labels": char_labels,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "words": words,
            "length_target": length_target,
            "length_supervise_mask": torch.ones(batch_size, dtype=torch.long),
        }
