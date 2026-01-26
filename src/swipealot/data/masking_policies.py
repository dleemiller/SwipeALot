"""Masking policies and helpers for swipe training collators."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from .collators.utils import mask_contiguous_blocks_1d


def prob_from_cfg(value: Any, rng: random.Random | None = None) -> float:
    rng = rng or random
    if isinstance(value, (tuple, list)):
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def create_inverted_masks(
    *,
    path_mask: torch.Tensor,
    char_mask: torch.Tensor,
    heavy: bool,
    mask_path: bool,
    path_mask_block_max_len: int,
    inverted_path_prob_heavy: Any,
    inverted_path_prob_light: Any,
    inverted_char_prob_heavy: Any,
    inverted_char_prob_light: Any,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = rng or random
    path_len = int(path_mask.shape[0])
    char_len = int(char_mask.shape[0])

    if heavy:
        path_mask_prob = prob_from_cfg(inverted_path_prob_heavy, rng=rng)
        text_mask_prob = prob_from_cfg(inverted_char_prob_heavy, rng=rng)
    else:
        path_mask_prob = prob_from_cfg(inverted_path_prob_light, rng=rng)
        text_mask_prob = prob_from_cfg(inverted_char_prob_light, rng=rng)

    path_mask_indices = torch.zeros(path_len, dtype=torch.long)
    if mask_path and path_mask_prob > 0.0:
        n_valid = int(path_mask.sum().item())
        n_to_mask = int(round(float(path_mask_prob) * n_valid))
        if n_to_mask > 0:
            path_mask_indices = mask_contiguous_blocks_1d(
                path_mask,
                n_to_mask,
                max_block_len=path_mask_block_max_len,
                rng=rng,
            )

    char_mask_indices = torch.zeros(char_len, dtype=torch.long)
    for i in range(char_len):
        if char_mask[i] == 0:
            continue
        if rng.random() < text_mask_prob:
            char_mask_indices[i] = 1

    return path_mask_indices, char_mask_indices


def create_modality_masks(
    *,
    path_mask: torch.Tensor,
    char_mask: torch.Tensor,
    mask_path_modality: bool,
    mask_path: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    path_len = int(path_mask.shape[0])
    char_len = int(char_mask.shape[0])

    if mask_path_modality:
        path_mask_indices = (
            path_mask.clone() if mask_path else torch.zeros(path_len, dtype=torch.long)
        )
        char_mask_indices = torch.zeros(char_len, dtype=torch.long)
    else:
        path_mask_indices = torch.zeros(path_len, dtype=torch.long)
        char_mask_indices = torch.zeros(char_len, dtype=torch.long)
        for i in range(char_len):
            if char_mask[i] == 0:
                continue
            char_mask_indices[i] = 1

    return path_mask_indices, char_mask_indices


def create_half_mask(
    *,
    path_mask: torch.Tensor,
    side: str,
    mask_path: bool,
    prob: Any,
    path_mask_block_max_len: int,
    rng: random.Random | None = None,
) -> torch.Tensor:
    rng = rng or random
    path_len = int(path_mask.shape[0])
    half = path_len // 2

    path_mask_indices = torch.zeros(path_len, dtype=torch.long)
    if mask_path:
        half_mask = path_mask.clone()
        if side == "right":
            half_mask[:half] = 0
        elif side == "left":
            half_mask[half:] = 0
        else:
            raise ValueError(f"Unknown side: {side!r}")

        n_valid = int(half_mask.sum().item())
        if n_valid > 0:
            mask_prob = prob_from_cfg(prob, rng=rng)
            n_to_mask = int(round(float(mask_prob) * n_valid))
            if n_to_mask > 0:
                path_mask_indices = mask_contiguous_blocks_1d(
                    half_mask,
                    n_to_mask,
                    max_block_len=path_mask_block_max_len,
                    rng=rng,
                )
    return path_mask_indices


def complement_mask(path_mask: torch.Tensor, masked_indices: torch.Tensor) -> torch.Tensor:
    complement = path_mask.clone()
    complement[masked_indices.bool()] = 0
    return complement


def reverse_path_coords(path_coords: torch.Tensor, path_mask: torch.Tensor) -> torch.Tensor:
    valid_len = int(path_mask.sum().item())
    if valid_len <= 1:
        return path_coords

    coords = path_coords.clone()
    segment = coords[:valid_len]
    dim = int(segment.shape[-1])

    if dim >= 6:
        x = segment[:, 0].flip(0)
        y = segment[:, 1].flip(0)
        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        if valid_len > 1:
            dx[1:] = x[1:] - x[:-1]
            dy[1:] = y[1:] - y[:-1]
        ds = torch.hypot(dx, dy)

        log_dt = segment[:, 5]
        dt = torch.expm1(torch.clamp(log_dt, min=0.0))
        dt_rev = dt.flip(0)
        dt_rev[0] = 0.0
        log_dt_rev = torch.log1p(torch.clamp(dt_rev, min=0.0))

        segment_rev = segment.flip(0).clone()
        segment_rev[:, 0] = x
        segment_rev[:, 1] = y
        segment_rev[:, 2] = dx
        segment_rev[:, 3] = dy
        segment_rev[:, 4] = ds
        segment_rev[:, 5] = log_dt_rev
        coords[:valid_len] = segment_rev
        return coords

    coords[:valid_len] = segment.flip(0)
    return coords


def reverse_char_tokens(
    char_tokens: torch.Tensor,
    char_mask: torch.Tensor,
    *,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_len = int(char_mask.sum().item())
    if valid_len <= 1:
        return char_tokens, char_mask

    tokens = char_tokens.clone()
    valid = tokens[:valid_len]
    if int(valid[-1].item()) == int(eos_id) and valid_len > 1:
        core = valid[:-1].flip(0)
        new_valid = torch.cat([core, valid[-1:]], dim=0)
    else:
        new_valid = valid.flip(0)
    tokens[:valid_len] = new_valid
    return tokens, char_mask


@dataclass
class MaskingStats:
    counts: dict[str, int] = field(default_factory=dict)
    path_mask_frac_sum: dict[str, float] = field(default_factory=dict)
    char_mask_frac_sum: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        *,
        mode: str,
        path_mask_indices: torch.Tensor,
        path_mask: torch.Tensor,
        char_mask_indices: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> None:
        valid_path = float(path_mask.sum().item())
        valid_char = float(char_mask.sum().item())
        path_frac = float(path_mask_indices.sum().item()) / valid_path if valid_path > 0 else 0.0
        char_frac = float(char_mask_indices.sum().item()) / valid_char if valid_char > 0 else 0.0

        self.counts[mode] = self.counts.get(mode, 0) + 1
        self.path_mask_frac_sum[mode] = self.path_mask_frac_sum.get(mode, 0.0) + path_frac
        self.char_mask_frac_sum[mode] = self.char_mask_frac_sum.get(mode, 0.0) + char_frac

    def summarize(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for mode, count in self.counts.items():
            if count <= 0:
                continue
            summary[mode] = {
                "count": float(count),
                "path_mask_frac_mean": self.path_mask_frac_sum.get(mode, 0.0) / count,
                "char_mask_frac_mean": self.char_mask_frac_sum.get(mode, 0.0) / count,
            }
        return summary

    def reset(self) -> None:
        self.counts.clear()
        self.path_mask_frac_sum.clear()
        self.char_mask_frac_sum.clear()
