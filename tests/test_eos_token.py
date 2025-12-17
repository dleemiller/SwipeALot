"""EOS token handling tests."""

import torch

from swipealot.data import CharacterTokenizer, MaskedCollator


def _build_sample(tokenizer: CharacterTokenizer, word: str, max_char_len: int):
    tokens = tokenizer.encode(word) + [tokenizer.eos_token_id]
    tokens = tokens[: max_char_len - 1] + [tokenizer.eos_token_id]
    tokens = tokens + [tokenizer.pad_token_id] * (max_char_len - len(tokens))
    char_mask = torch.tensor([1 if t != tokenizer.pad_token_id else 0 for t in tokens])

    path_len = 4
    return {
        "path_coords": torch.zeros(path_len, 3),
        "path_mask": torch.ones(path_len, dtype=torch.long),
        "char_tokens": torch.tensor(tokens, dtype=torch.long),
        "char_mask": char_mask,
        "word": word,
    }


def test_eos_token_added_and_preserved():
    tokenizer = CharacterTokenizer()
    max_char_len = 10

    for word in ["hello", "hi", "a", "test"]:
        sample = _build_sample(tokenizer, word, max_char_len)
        tokens = sample["char_tokens"]
        char_mask = sample["char_mask"]

        # EOS should be present before padding starts
        valid_tokens = tokens[char_mask == 1]
        assert tokenizer.eos_token_id in valid_tokens.tolist()

        # Decoding up to EOS should recover the original lowercase word
        eos_idx = (valid_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
        decoded = tokenizer.decode(valid_tokens[: eos_idx + 1].tolist())
        assert decoded == word.lower()


def test_eos_in_masking_labels():
    tokenizer = CharacterTokenizer()
    max_char_len = 8

    samples = [_build_sample(tokenizer, word, max_char_len) for word in ["hello", "world"]]

    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=1.0,  # mask all valid tokens, including EOS
        path_mask_prob=0.0,
        mask_path=False,
    )

    batch = collator(samples)
    labels = batch["char_labels"]

    # Every sequence should include EOS in the masked labels
    for seq_labels in labels:
        assert tokenizer.eos_token_id in seq_labels.tolist()

