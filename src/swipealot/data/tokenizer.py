"""Character tokenizer and vocabulary utilities for swipe keyboard dataset."""

import hashlib

import torch


class CharacterTokenizer:
    """Character-level tokenizer for swipe keyboard words."""

    def __init__(self, vocab: set | None = None):
        """
        Initialize tokenizer with vocabulary.

        Args:
            vocab: Optional set of characters. If None, will use printable ASCII.
        """
        # Special tokens
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.eos_token = "[EOS]"  # End of word token

        self.special_tokens = [
            self.pad_token,  # 0
            self.cls_token,  # 1
            self.sep_token,  # 2
            self.mask_token,  # 3
            self.unk_token,  # 4
            self.eos_token,  # 5
        ]

        # Build vocabulary
        if vocab is None:
            # Use printable ASCII + common characters
            chars = set()
            for i in range(32, 127):  # Printable ASCII
                chars.add(chr(i))
        else:
            chars = vocab

        self.char_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        for idx, char in enumerate(sorted(chars), start=len(self.special_tokens)):
            self.char_to_id[char] = idx

        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

    @classmethod
    def from_dataset(cls, dataset) -> "CharacterTokenizer":
        """Build tokenizer vocabulary from dataset (case-insensitive)."""
        chars = set()
        for sample in dataset:
            for char in sample["word"].lower():  # Convert to lowercase
                chars.add(char)
        return cls(vocab=chars)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (case-insensitive)."""
        unk_id = self.char_to_id[self.unk_token]
        return [self.char_to_id.get(char, unk_id) for char in text.lower()]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text, stopping at EOS token."""
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                # Stop at EOS token
                if char == self.eos_token:
                    break
                # Skip other special tokens except for debugging
                if char not in self.special_tokens or char == " ":
                    chars.append(char)
        return "".join(chars)

    @property
    def pad_token_id(self) -> int:
        return self.char_to_id[self.pad_token]

    @property
    def cls_token_id(self) -> int:
        return self.char_to_id[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self.char_to_id[self.sep_token]

    @property
    def mask_token_id(self) -> int:
        return self.char_to_id[self.mask_token]

    @property
    def unk_token_id(self) -> int:
        return self.char_to_id[self.unk_token]

    @property
    def eos_token_id(self) -> int:
        return self.char_to_id[self.eos_token]


def vocab_hash(tokenizer: CharacterTokenizer) -> str:
    """Stable hash of the tokenizer's id->token mapping (includes specials)."""
    ordered_tokens = [tokenizer.id_to_char[i] for i in range(tokenizer.vocab_size)]
    joined = "\n".join(ordered_tokens).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def compute_char_frequency_weights(
    tokenizer: CharacterTokenizer,
    dataset,
    max_samples: int | None = None,
    weight_exponent: float = 1.0,
):
    """Compute inverse log frequency weights for characters.

    Args:
        tokenizer: CharacterTokenizer used for encoding
        dataset: HF dataset or iterable of samples with a 'word' field
        max_samples: Optional cap on samples to scan

    Returns:
        torch.Tensor of shape [vocab_size] with weights normalized to mean=1.
        Padding token weight is set to the non-pad mean (not zero) so min>0.
    """
    counts = torch.ones(tokenizer.vocab_size, dtype=torch.float)  # start at 1 for smoothing

    # Collect all token IDs first for vectorized counting
    all_token_ids = []
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        # Encode lowercase characters and append EOS (matches training labels)
        token_ids = tokenizer.encode(sample["word"]) + [tokenizer.eos_token_id]
        all_token_ids.extend(token_ids)

    # Use bincount for efficient vectorized counting
    if all_token_ids:
        token_tensor = torch.tensor(all_token_ids, dtype=torch.long)
        bincount_result = torch.bincount(token_tensor, minlength=tokenizer.vocab_size).float()
        counts = counts + bincount_result

    # Padding is never a supervised label, but keep a finite weight
    pad_id = tokenizer.pad_token_id
    counts[pad_id] = counts[pad_id]  # leave smoothing value as-is

    # Inverse log weighting; add 1 inside log to avoid div-by-zero
    weights = 1.0 / torch.log1p(counts)

    # Use non-pad mean for pad token to avoid zero/inf
    non_pad_mask = torch.ones_like(weights, dtype=torch.bool)
    non_pad_mask[pad_id] = False
    non_pad_mean = weights[non_pad_mask].mean().clamp_min(1e-8)
    weights[pad_id] = non_pad_mean

    # Optional tempering (e.g., exponent <1 flattens extremes)
    if weight_exponent != 1.0:
        weights = torch.pow(weights, weight_exponent)

    # Normalize to keep loss scale stable (mean of non-pad tokens -> 1)
    non_pad_mean = weights[non_pad_mask].mean().clamp_min(1e-8)
    weights[pad_id] = non_pad_mean
    weights = weights / non_pad_mean

    return weights
