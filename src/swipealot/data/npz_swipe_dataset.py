"""Load preprocessed NPZ shards in SwipeDataset-compatible format."""

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import CharacterTokenizer


class NPZSwipeDataset(Dataset):
    """Load preprocessed NPZ shards and return samples matching SwipeDataset format.

    NPZ shards contain path_features [N, 8, max_path_len] (float16),
    path_mask [N, max_path_len], and words [N].  This dataset transposes
    path_features to [max_path_len, 8] float32 and tokenizes words, so the
    output dict is identical to SwipeDataset.__getitem__.
    """

    def __init__(
        self,
        npz_paths: str | list[str],
        tokenizer: CharacterTokenizer,
        max_word_len: int = 48,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_word_len = max_word_len

        files = self._resolve_paths(npz_paths)
        if not files:
            raise FileNotFoundError(f"No NPZ files found for: {npz_paths}")

        all_features = []
        all_path_mask = []
        all_words = []

        for fpath in sorted(files):
            data = np.load(fpath, allow_pickle=True)
            all_features.append(data["path_features"])
            all_path_mask.append(data["path_mask"])
            all_words.append(data["words"])

        self.path_features = np.concatenate(all_features, axis=0)  # [N, 8, L] float16
        self.path_mask = np.concatenate(all_path_mask, axis=0)  # [N, L]
        self.words = np.concatenate(all_words, axis=0)  # [N] object

        if max_samples is not None:
            self.path_features = self.path_features[:max_samples]
            self.path_mask = self.path_mask[:max_samples]
            self.words = self.words[:max_samples]

    def __len__(self) -> int:
        return len(self.path_features)

    def __getitem__(self, idx: int) -> dict:
        # [8, L] float16 -> [L, 8] float32
        path_coords = torch.from_numpy(self.path_features[idx].astype(np.float32)).T

        path_mask = torch.from_numpy(self.path_mask[idx].copy()).long()

        word = str(self.words[idx])

        # Tokenize word (same logic as SwipeDataset.__getitem__)
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
            "path_coords": path_coords,  # [max_path_len, 8]
            "char_tokens": torch.tensor(char_tokens, dtype=torch.long),  # [max_word_len]
            "path_mask": path_mask,  # [max_path_len]
            "char_mask": torch.tensor(char_mask, dtype=torch.long),  # [max_word_len]
            "word": word,
        }

    @staticmethod
    def _resolve_paths(data_paths: str | list[str]) -> list[str]:
        """Resolve data_paths to a flat list of .npz file paths."""
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        files: list[str] = []
        for p in data_paths:
            path = Path(p)
            if path.is_dir():
                files.extend(str(f) for f in sorted(path.rglob("*.npz")))
            elif "*" in p or "?" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))
            elif path.is_file() and path.suffix == ".npz":
                files.append(str(path))
        return files
