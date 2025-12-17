"""Test independent masking probabilities for path and text using synthetic data."""

import torch
from torch.utils.data import DataLoader

from swipealot.data import CharacterTokenizer, MaskedCollator


def _make_sample(tokenizer: CharacterTokenizer, word: str, path_len: int, char_len: int):
    path_coords = torch.randn(path_len, 6)
    path_mask = torch.ones(path_len, dtype=torch.long)

    token_ids = tokenizer.encode(word) + [tokenizer.eos_token_id]
    token_ids = token_ids[: char_len - 1] + [tokenizer.eos_token_id]
    token_ids = token_ids + [tokenizer.pad_token_id] * (char_len - len(token_ids))

    char_mask = torch.tensor([1 if t != tokenizer.pad_token_id else 0 for t in token_ids])

    return {
        "path_coords": path_coords,
        "path_mask": path_mask,
        "char_tokens": torch.tensor(token_ids, dtype=torch.long),
        "char_mask": char_mask,
        "word": word,
    }


def test_independent_masking():
    torch.manual_seed(0)
    tokenizer = CharacterTokenizer()

    samples = [
        _make_sample(tokenizer, word, path_len=32, char_len=16)
        for word in ["hello", "world", "keyboard", "swipe", "model", "test"] * 6
    ]

    configs = [
        {"char": 0.15, "path": 0.15},
        {"char": 0.30, "path": 0.15},
        {"char": 0.15, "path": 0.30},
        {"char": 0.50, "path": 0.05},
        {"char": 0.0, "path": 0.20},
    ]

    for cfg in configs:
        collator = MaskedCollator(
            tokenizer=tokenizer,
            char_mask_prob=cfg["char"],
            path_mask_prob=cfg["path"],
            mask_path=True,
        )
        loader = DataLoader(samples, batch_size=16, shuffle=False, collate_fn=collator)
        batch = next(iter(loader))

        char_labels = batch["char_labels"]
        char_mask = batch["char_mask"]
        path_mask_indices = batch["path_mask_indices"]
        path_mask = batch["path_mask"]

        valid_char_positions = char_mask == 1
        char_masked = (char_labels != -100).sum().item()
        char_total_valid = valid_char_positions.sum().item()
        char_masked_pct = 100 * char_masked / char_total_valid if char_total_valid > 0 else 0

        valid_path_positions = path_mask == 1
        path_masked = path_mask_indices.sum().item()
        path_total_valid = valid_path_positions.sum().item()
        path_masked_pct = 100 * path_masked / path_total_valid if path_total_valid > 0 else 0

        # Allow a loose tolerance; probabilities are stochastic
        assert abs(char_masked_pct - cfg["char"] * 100) < 10
        assert abs(path_masked_pct - cfg["path"] * 100) < 10


if __name__ == "__main__":
    test_independent_masking()
