"""Tests for masking and attention mask construction in collators."""

import torch

from swipealot.data import (
    CharacterTokenizer,
    MaskedCollator,
    PairwiseMaskedCollator,
    ValidationCollator,
)
from swipealot.huggingface import SwipeProcessor


def _sample(tokenizer: CharacterTokenizer, word: str, path_len: int, char_len: int) -> dict:
    token_ids = tokenizer.encode(word) + [tokenizer.eos_token_id]
    token_ids = token_ids[: char_len - 1] + [tokenizer.eos_token_id]
    token_ids = token_ids + [tokenizer.pad_token_id] * (char_len - len(token_ids))
    char_mask = torch.tensor(
        [1 if t != tokenizer.pad_token_id else 0 for t in token_ids], dtype=torch.long
    )
    return {
        "path_coords": torch.randn(path_len, 6),
        "path_mask": torch.ones(path_len, dtype=torch.long),
        "char_tokens": torch.tensor(token_ids, dtype=torch.long),
        "char_mask": char_mask,
        "word": word,
    }


def test_masked_collator_attention_and_labels():
    tokenizer = CharacterTokenizer()
    sample = _sample(tokenizer, "hello", path_len=5, char_len=7)
    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=0.5,
        path_mask_prob=0.5,
        mask_path=True,
    )

    batch = collator([sample, sample])

    seq_len = 1 + 5 + 1 + 7  # CLS + path + SEP + chars
    assert batch["attention_mask"].shape == (2, seq_len)
    # CLS and SEP should always be attended
    assert torch.all(batch["attention_mask"][:, 0] == 1)
    assert torch.all(batch["attention_mask"][:, 1 + 5] == 1)

    # Path and char portions mirror the masks
    assert torch.all(batch["attention_mask"][:, 1 : 1 + 5] == batch["path_mask"])
    assert torch.all(batch["attention_mask"][:, -7:] == batch["char_mask"])

    # Path masking outputs exist and align in shape
    assert batch["path_labels"].shape == (2, 5, 6)
    assert batch["path_mask_indices"].shape == (2, 5)

    # Character labels should ignore padding
    pad_positions = batch["char_mask"] == 0
    assert torch.all(batch["char_labels"][pad_positions] == -100)


def test_pairwise_collator_shapes():
    tokenizer = CharacterTokenizer()
    sample = _sample(tokenizer, "swipe", path_len=4, char_len=6)
    collator = PairwiseMaskedCollator(
        tokenizer=tokenizer,
        mask_path=True,
        modality_prob=0.0,  # force inverted mode for determinism in shape expectations
        zero_attention_prob=0.0,
    )

    batch = collator([sample])

    # Two views are produced
    assert batch["path_coords"].shape[0] == 2
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == (2, 1 + 4 + 1 + 6)

    # Pair metadata matches views
    assert batch["pair_ids"].tolist() == [0, 0]
    assert batch["gradient_mask"].shape == (2,)
    assert set(batch.keys()) >= {
        "path_coords",
        "input_ids",
        "char_labels",
        "attention_mask",
        "path_labels",
        "path_mask_indices",
        "pair_ids",
        "gradient_mask",
    }


def test_pairwise_collator_modality_mode_masks_path_and_text():
    tokenizer = CharacterTokenizer()
    sample = _sample(tokenizer, "modal", path_len=3, char_len=5)
    collator = PairwiseMaskedCollator(
        tokenizer=tokenizer,
        mask_path=True,
        modality_prob=1.0,  # force modality mode
        zero_attention_prob=0.0,
    )

    batch = collator([sample])

    # View A (text masked) should have char_mask_indices all ones, path_mask_indices zeros
    path_mask_a = batch["path_mask_indices"][0]
    char_mask_a = batch["char_labels"][0] != -100
    assert torch.all(path_mask_a == 0)
    assert torch.all(char_mask_a == 1)

    # View B (path masked) should invert: path masked, text labels ignored
    path_mask_b = batch["path_mask_indices"][1]
    char_mask_b = batch["char_labels"][1] != -100
    assert torch.all(path_mask_b == batch["path_mask"][1])
    assert torch.all(char_mask_b == 0)


def test_validation_collator_attention_and_labels():
    tokenizer = CharacterTokenizer()
    sample = _sample(tokenizer, "test", path_len=3, char_len=5)
    collator = ValidationCollator(tokenizer)

    batch = collator([sample])

    seq_len = 1 + 3 + 1 + 5
    assert batch["attention_mask"].shape == (1, seq_len)

    # CLS + path + SEP are attended; chars follow char_mask
    assert torch.all(batch["attention_mask"][0, 0 : 1 + 3 + 1] == 1)
    assert torch.all(batch["attention_mask"][0, -5:] == batch["char_mask"])

    # Labels should mask padding positions
    pad_positions = batch["char_mask"] == 0
    assert torch.all(batch["char_labels"][pad_positions] == -100)


def test_processor_attention_when_text_missing():
    tokenizer = CharacterTokenizer()
    processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=4, max_char_len=6)

    path_coords = [[0.1, 0.2, 0.0], [0.2, 0.3, 0.1]]
    inputs = processor(path_coords=path_coords, text=None, return_tensors="pt")

    # Input IDs should be all PAD when text is None
    assert torch.all(inputs["input_ids"] == tokenizer.pad_token_id)
    # Char portion of attention mask should be zeros
    char_attn = inputs["attention_mask"][0, -(processor.max_char_len) :]
    assert torch.all(char_attn == 0)
    # Path portion attends; length matches 1+path+1+chars
    assert (
        inputs["attention_mask"].shape[1] == 1 + processor.max_path_len + 1 + processor.max_char_len
    )


def test_pairwise_collator_zero_attention_prob():
    tokenizer = CharacterTokenizer()
    sample = _sample(tokenizer, "modal", path_len=3, char_len=5)
    collator = PairwiseMaskedCollator(
        tokenizer=tokenizer,
        mask_path=True,
        modality_prob=1.0,
        zero_attention_prob=1.0,  # always zero attention in modality mode
    )

    batch = collator([sample])

    # View A: text attention zeroed
    attn_a = batch["attention_mask"][0]
    assert attn_a[-5:].sum() == 0
    assert attn_a[: 1 + 3 + 1].sum() == 1 + 3 + 1

    # View B: path attention zeroed
    attn_b = batch["attention_mask"][1]
    path_segment = attn_b[1 : 1 + 3]
    assert path_segment.sum() == 0
    # text attended
    assert attn_b[-5:].sum() == 5


def test_pairwise_collator_accepts_custom_inverted_probs():
    tokenizer = CharacterTokenizer()
    custom_collator = PairwiseMaskedCollator(
        tokenizer=tokenizer,
        mask_path=True,
        modality_prob=0.0,
        zero_attention_prob=0.0,
        inverted_char_prob_heavy=0.9,
        inverted_path_prob_heavy=0.8,
        inverted_char_prob_light=0.3,
        inverted_path_prob_light=0.25,
    )

    assert custom_collator.pairwise_inverted_char_prob_heavy == 0.9
    assert custom_collator.pairwise_inverted_path_prob_heavy == 0.8
    assert custom_collator.pairwise_inverted_char_prob_light == 0.3
    assert custom_collator.pairwise_inverted_path_prob_light == 0.25
