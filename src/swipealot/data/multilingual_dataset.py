"""SwipeMultilingualDataset — adapter for futo-org/swipe-multilingual-compiled.

Produces the same output dict as SwipeDataset so it can be used with the
existing collators and trainer unchanged.

Excluded sources by default: ``yandex``, ``indic_swipe``.
"""

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .preprocessing import preprocess_raw_path_to_sg_features
from .tokenizer import CharacterTokenizer

__all__ = ["SwipeMultilingualDataset"]

_DEFAULT_EXCLUDE_SOURCES = frozenset({"yandex", "indic_swipe"})
_DATASET_NAME = "futo-org/swipe-multilingual-compiled"


class SwipeMultilingualDataset(Dataset):
    """Adapter for the swipe-multilingual-compiled HuggingFace dataset.

    Converts the ``points_x / points_y / points_t`` column format into
    the ``{"x", "y", "t"}`` dict list expected by
    ``preprocess_raw_path_to_sg_features``, then returns the same dict
    structure as ``SwipeDataset``.
    """

    def __init__(
        self,
        split: str = "train",
        max_path_len: int = 128,
        max_word_len: int = 48,
        tokenizer: CharacterTokenizer | None = None,
        exclude_sources: frozenset[str] | set[str] = _DEFAULT_EXCLUDE_SOURCES,
        max_samples: int | None = None,
        path_resample_mode: str = "time",
    ):
        self.max_path_len = max_path_len
        self.max_word_len = max_word_len
        self.path_resample_mode = path_resample_mode
        self.tokenizer = tokenizer if tokenizer is not None else CharacterTokenizer()

        print(f"Loading {_DATASET_NAME} (swipes), split: {split}")
        ds = load_dataset(_DATASET_NAME, "swipes", split=split)

        if exclude_sources:
            ds = ds.filter(lambda r: r["source"] not in exclude_sources)
            print(f"After excluding {set(exclude_sources)}: {len(ds):,} samples")

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.dataset = ds
        print(f"Final dataset size: {len(self.dataset):,} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        xs = sample["points_x"]
        ys = sample["points_y"]
        ts = sample["points_t"]
        data_points = [{"x": x, "y": y, "t": t} for x, y, t in zip(xs, ys, ts, strict=True)]

        path_features, path_mask = preprocess_raw_path_to_sg_features(
            data_points,
            self.max_path_len,
            resample_mode=self.path_resample_mode,
        )

        word = sample["word"]
        char_tokens = self.tokenizer.encode(word)
        char_tokens = char_tokens + [self.tokenizer.eos_token_id]

        if len(char_tokens) < self.max_word_len:
            char_tokens = char_tokens + [self.tokenizer.pad_token_id] * (
                self.max_word_len - len(char_tokens)
            )
        else:
            char_tokens = char_tokens[: self.max_word_len - 1] + [self.tokenizer.eos_token_id]

        char_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in char_tokens]

        return {
            "path_coords": torch.tensor(path_features, dtype=torch.float32),
            "char_tokens": torch.tensor(char_tokens, dtype=torch.long),
            "path_mask": torch.tensor(path_mask, dtype=torch.long),
            "char_mask": torch.tensor(char_mask, dtype=torch.long),
            "word": word,
        }
