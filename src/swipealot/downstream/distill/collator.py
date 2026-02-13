"""Collator for distillation training.

Produces path features + masked text input + CTC targets from the dataset.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from swipealot.data.preprocessing import preprocess_raw_path_to_features


def _to_raw_dict_path(data) -> list[dict[str, float]]:
    """Accept either raw dict points or Nx3 numeric points and return dicts with x/y/t."""
    if data is None:
        return []
    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
        return data
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected path as Nx3 or list-of-dicts, got shape={arr.shape}")
    if arr.shape[1] == 2:
        t = np.zeros((arr.shape[0],), dtype=np.float32)
        arr = np.concatenate([arr.astype(np.float32), t[:, None]], axis=1)
    return [{"x": float(p[0]), "y": float(p[1]), "t": float(p[2])} for p in arr]


def _word_to_ctc_target(word: str) -> tuple[list[int], int]:
    """Convert word to CTC target indices (0=a, 1=b, ..., 25=z).

    Returns:
        (target_indices, length)
    """
    indices = []
    for ch in word.lower():
        if "a" <= ch <= "z":
            indices.append(ord(ch) - ord("a"))
    return indices, len(indices)


def _encode_ctc_batch(words: list[str], max_label_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode words as padded CTC targets.

    Returns:
        (labels [B, max_label_len], label_lengths [B])
    """
    all_targets = []
    all_lengths = []
    for word in words:
        indices, length = _word_to_ctc_target(word)
        padded = indices[:max_label_len]
        length = min(length, max_label_len)
        padded = padded + [0] * (max_label_len - len(padded))
        all_targets.append(padded)
        all_lengths.append(length)

    labels = torch.tensor(all_targets, dtype=torch.long)
    label_lengths = torch.tensor(all_lengths, dtype=torch.long)
    return labels, label_lengths


class HFToWordDataset(Dataset):
    """Wraps an HF dataset to provide {word, data} items."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {"word": item["word"], "data": item["data"]}


class SwipeDistillCollator:
    """Build batches for distillation fine-tuning.

    Expects items with keys:
      - `word`: string
      - `data`: path, either list-of-dicts (x/y/t) or Nx3 numeric list/array
    """

    def __init__(
        self,
        *,
        processor,
        resample_mode: str | None = None,
        max_label_len: int = 32,
    ):
        self.processor = processor
        self.resample_mode = resample_mode
        self.max_label_len = max_label_len

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        words = [it["word"] for it in items]
        paths = [it["data"] for it in items]

        # Encode text (for encoder input, will be masked in model)
        enc = self.processor.encode_text(words, return_tensors="pt")

        max_path_len = int(self.processor.max_path_len)
        path_input_dim = int(getattr(self.processor, "path_input_dim", 8))

        all_path_coords = []  # [B, P, D] for encoder input
        all_path_features = []  # [B, D, P] channel-first for concat

        for p in paths:
            raw = _to_raw_dict_path(p)
            feats, _ = preprocess_raw_path_to_features(
                raw,
                max_path_len,
                resample_mode=(
                    self.resample_mode or str(getattr(self.processor, "path_resample_mode", "time"))
                ),
            )
            # feats: [P, D] where D=8 (x, y, dx, dy, d2x, d2y, speed, curvature)
            # Pad or truncate to path_input_dim
            if feats.shape[1] < path_input_dim:
                pad = np.zeros((feats.shape[0], path_input_dim - feats.shape[1]), dtype=np.float32)
                feats = np.concatenate([feats, pad], axis=1)
            elif feats.shape[1] > path_input_dim:
                feats = feats[:, :path_input_dim]

            all_path_coords.append(feats)  # [P, D]
            all_path_features.append(feats.T)  # [D, P] channel-first

        path_coords = torch.from_numpy(np.stack(all_path_coords)).float()  # [B, P, D]
        path_features = torch.from_numpy(np.stack(all_path_features)).float()  # [B, D, P]

        labels, label_lengths = _encode_ctc_batch(words, self.max_label_len)

        return {
            "path_coords": path_coords,
            "path_features": path_features,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "label_lengths": label_lengths,
            "words": words,
        }


class NPZDistillCollator:
    """Collator for NPZ data that already has path_features.

    Expects items with keys:
      - `path_features`: [8, 128] tensor (channel-first)
      - `word`: string
    """

    def __init__(self, *, processor, max_label_len: int = 32):
        self.processor = processor
        self.max_label_len = max_label_len

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        words = [it["word"] for it in items]
        features = [it["path_features"] for it in items]

        # Encode text
        enc = self.processor.encode_text(words, return_tensors="pt")

        # Stack path features [B, 8, 128]
        path_features = torch.stack(features)  # already [B, 8, 128]

        # path_coords for encoder: transpose to [B, 128, 8]
        path_coords = path_features.transpose(1, 2)

        labels, label_lengths = _encode_ctc_batch(words, self.max_label_len)

        return {
            "path_coords": path_coords,
            "path_features": path_features,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "label_lengths": label_lengths,
            "words": words,
        }
