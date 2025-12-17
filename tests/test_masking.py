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


def test_path_masking_produces_contiguous_blocks():
    tokenizer = CharacterTokenizer()
    sample = _make_sample(tokenizer, "hello", path_len=32, char_len=16)
    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=0.0,
        path_mask_prob=0.5,
        mask_path=True,
        path_mask_block_max_len=32,
    )

    batch = collator([sample])
    mask = batch["path_mask_indices"][0].tolist()

    # Ensure we masked something.
    assert sum(mask) > 0

    # Compute maximum contiguous run of masked points; with block masking it should be >= 2.
    max_run = 0
    cur = 0
    for v in mask:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    assert max_run >= 2


def test_path_masking_respects_max_block_len_cap():
    tokenizer = CharacterTokenizer()
    sample = _make_sample(tokenizer, "hello", path_len=128, char_len=16)
    cap = 32
    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=0.0,
        path_mask_prob=0.9,
        mask_path=True,
        path_mask_block_max_len=cap,
    )

    batch = collator([sample])
    mask = batch["path_mask_indices"][0].tolist()
    assert sum(mask) > 0

    max_run = 0
    cur = 0
    for v in mask:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    assert max_run <= cap


if __name__ == "__main__":
    test_independent_masking()
