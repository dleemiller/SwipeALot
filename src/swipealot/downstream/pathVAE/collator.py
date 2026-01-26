"""Collator for text-to-path CVAE training."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from swipealot.data.masking_policies import reverse_char_tokens, reverse_path_coords
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
class TextToPathCVAEBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels_xy: torch.Tensor
    labels_mask: torch.Tensor


class SwipeTextToPathCVAECollator:
    """Build batches for text-to-path CVAE fine-tuning.

    Expects items with keys:
      - `word`: string
      - `data`: path, either list-of-dicts (x/y/t) or Nx3 numeric list/array
    """

    def __init__(self, *, processor, resample_mode: str | None = None, reverse_prob: float = 0.0):
        self.processor = processor
        self.resample_mode = resample_mode
        self.reverse_prob = float(reverse_prob)

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        words = [it["word"] for it in items]
        paths = [it["data"] for it in items]

        reverse_flags = [
            self.reverse_prob > 0.0 and random.random() < self.reverse_prob for _ in items
        ]

        enc = self.processor.encode_text(words, return_tensors="pt")
        if "attention_mask" not in enc or "input_ids" not in enc:
            raise ValueError("Processor did not return input_ids/attention_mask for text encoding.")

        pad_id = int(self.processor.tokenizer.pad_token_id)
        char_mask = enc["input_ids"].ne(pad_id).long()
        if self.reverse_prob > 0.0:
            eos_id = int(self.processor.tokenizer.eos_token_id)
            for i, do_reverse in enumerate(reverse_flags):
                if do_reverse:
                    tokens, _ = reverse_char_tokens(
                        enc["input_ids"][i], char_mask[i], eos_id=eos_id
                    )
                    enc["input_ids"][i] = tokens

        max_path_len = int(self.processor.max_path_len)
        xy = []
        mask = []
        input_paths = []
        for idx, p in enumerate(paths):
            raw = _to_raw_dict_path(p)
            feats, m = preprocess_raw_path_to_features(
                raw,
                max_path_len,
                resample_mode=(
                    self.resample_mode or str(getattr(self.processor, "path_resample_mode", "time"))
                ),
            )
            if reverse_flags[idx]:
                feats_t = torch.from_numpy(feats).float()
                mask_t = torch.from_numpy(m).long()
                feats = reverse_path_coords(feats_t, mask_t).cpu().numpy()
            xy.append(feats[:, :2])
            mask.append(m)
            input_feats = np.zeros_like(feats)
            valid = np.where(m == 1)[0]
            if valid.size > 0:
                first_idx = int(valid[0])
                last_idx = int(valid[-1])
                input_feats[first_idx] = feats[first_idx]
                if last_idx != first_idx:
                    input_feats[last_idx] = feats[last_idx]
            input_paths.append(input_feats)

        labels_xy = torch.from_numpy(np.stack(xy)).float()
        labels_mask = torch.from_numpy(np.stack(mask)).long()
        path_coords = torch.from_numpy(np.stack(input_paths)).float()
        has_path = labels_mask.sum(dim=1, keepdim=True) > 0
        path_attention_mask = torch.where(has_path, torch.ones_like(labels_mask), labels_mask)
        cls_mask = torch.ones((path_attention_mask.shape[0], 1), dtype=path_attention_mask.dtype)
        sep_mask = torch.ones((path_attention_mask.shape[0], 1), dtype=path_attention_mask.dtype)
        attention_mask = torch.cat([cls_mask, path_attention_mask, sep_mask, char_mask], dim=1)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": attention_mask,
            "path_coords": path_coords,
            "labels_xy": labels_xy,
            "labels_mask": labels_mask,
        }
