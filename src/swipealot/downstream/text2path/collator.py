"""Collator for text→path training.

This lives under `downstream/` so it can be dropped for lightweight encoder-only distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

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
    out: list[dict[str, float]] = []
    for p in arr:
        out.append({"x": float(p[0]), "y": float(p[1]), "t": float(p[2])})
    return out


@dataclass(frozen=True)
class TextToPathBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels_xy: torch.Tensor
    labels_mask: torch.Tensor


class SwipeTextToPathCollator:
    """Build batches for text→path fine-tuning.

    Expects items with keys:
      - `word`: string
      - `data`: path, either list-of-dicts (x/y/t) or Nx3 numeric list/array
    """

    def __init__(self, *, processor, resample_mode: str | None = None):
        self.processor = processor
        self.resample_mode = resample_mode

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        words = [it["word"] for it in items]
        paths = [it["data"] for it in items]

        enc = self.processor.encode_text(words, return_tensors="pt")

        max_path_len = int(self.processor.max_path_len)
        xy = []
        mask = []
        for p in paths:
            raw = _to_raw_dict_path(p)
            feats, m = preprocess_raw_path_to_features(
                raw,
                max_path_len,
                resample_mode=(
                    self.resample_mode or str(getattr(self.processor, "path_resample_mode", "time"))
                ),
            )
            xy.append(feats[:, :2])
            mask.append(m)

        labels_xy = torch.from_numpy(np.stack(xy)).float()
        labels_mask = torch.from_numpy(np.stack(mask)).long()

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels_xy": labels_xy,
            "labels_mask": labels_mask,
        }
