#!/usr/bin/env python3
"""Batch keyboard-proximity augmentation: flip characters to adjacent QWERTY keys.

Randomly replaces characters with their keyboard neighbors (1 flip per 3 chars),
generates augmented swipe paths via attention-guided reconstruction, and appends
results to a JSONL file.

Architecture:
  - Single model on GPU handles batched forward passes
  - ProcessPoolExecutor parallelizes CPU-bound work (curve fitting, post-processing)

Usage:
  uv run keyboard-augment-batch \\
      --checkpoint checkpoints/base_20251228_145048/final \\
      --output augmented_keyboard.jsonl \\
      -n 1000 --blend 0.5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.interpolate import interp1d
from transformers import AutoModel, AutoProcessor

from swipealot.analysis.attention_capture import get_all_layer_attentions
from swipealot.analysis.attention_extractor import extract_path_to_char_attention
from swipealot.analysis.attention_visualizer import (
    _filter_path_points,
    _infer_time_axis,
    _pool_layer_attentions,
)

# ---------------------------------------------------------------------------
# QWERTY adjacency map
# ---------------------------------------------------------------------------

QWERTY_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]
# Row offsets (staggered keyboard): row 0 = 0.0, row 1 = 0.5, row 2 = 1.0
ROW_OFFSETS = [0.0, 0.5, 1.0]


def build_adjacency() -> dict[str, list[str]]:
    """Build adjacency map: for each key, list of touching keys (8-connected with stagger)."""
    positions: dict[str, tuple[float, float]] = {}
    for row_idx, row in enumerate(QWERTY_ROWS):
        offset = ROW_OFFSETS[row_idx]
        for col_idx, ch in enumerate(row):
            positions[ch] = (row_idx, col_idx + offset)

    adj: dict[str, list[str]] = {}
    for ch, (r, c) in positions.items():
        neighbors = []
        for other, (r2, c2) in positions.items():
            if other == ch:
                continue
            dist = math.sqrt((r - r2) ** 2 + (c - c2) ** 2)
            if dist <= 1.2:
                neighbors.append(other)
        adj[ch] = sorted(neighbors)
    return adj


ADJACENCY = build_adjacency()


def get_adjacent_keys(char: str) -> list[str]:
    """Return list of adjacent keys for a given character."""
    return ADJACENCY.get(char.lower(), [])


# ---------------------------------------------------------------------------
# Helpers (same as misspelling_batch.py)
# ---------------------------------------------------------------------------


def _apply_temperature(
    attn: torch.Tensor,
    path_mask: torch.Tensor,
    temperature: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if temperature == 1.0:
        return attn
    masked = attn * path_mask.view(1, 1, -1).to(attn.dtype)
    row_sum = masked.sum(dim=-1, keepdim=True)
    safe_sum = row_sum.clamp_min(eps)
    p = masked / safe_sum
    logp = torch.log(p.clamp_min(eps))
    p_t = torch.softmax(logp / float(temperature), dim=-1)
    return p_t * row_sum


def _fit_curve(
    t: np.ndarray,
    y: np.ndarray,
    mu_init: float,
    sigma_init: float,
    amp_init: float,
    family: str,
    *,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    eps = 1e-6
    mu_init = float(np.clip(mu_init, eps, 1.0 - eps))
    sigma_init = float(max(sigma_init, eps))
    amp_init = float(max(amp_init, eps))

    mu_raw = torch.tensor(np.log(mu_init / (1.0 - mu_init)), device=device, requires_grad=True)
    log_sigma_param = torch.tensor(np.log(sigma_init), device=device, requires_grad=True)
    log_amp = torch.tensor(np.log(amp_init), device=device, requires_grad=True)

    params = [mu_raw, log_sigma_param, log_amp]
    log_nu = None
    if family == "student":
        log_nu = torch.tensor(np.log(4.0), device=device, requires_grad=True)
        params.append(log_nu)

    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(int(steps)):
        mu = torch.sigmoid(mu_raw)
        sigma = torch.exp(log_sigma_param).clamp_min(1e-4)
        amp = torch.exp(log_amp)
        if family == "gaussian":
            pred = amp * torch.exp(-0.5 * ((t_t - mu) / sigma) ** 2)
        elif family == "student":
            nu = torch.exp(log_nu).clamp_min(1e-4) + 1.0
            z2 = ((t_t - mu) / sigma) ** 2
            pred = amp * torch.pow(1.0 + (z2 / nu), -(nu + 1.0) * 0.5)
        else:
            raise ValueError(f"Unknown fit family: {family}")
        loss = torch.mean((pred - y_t) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    mu = float(torch.sigmoid(mu_raw).item())
    sigma = float(torch.exp(log_sigma_param).clamp_min(1e-4).item())
    amp = float(torch.exp(log_amp).item())
    nu = (
        float((torch.exp(log_nu).clamp_min(1e-4) + 1.0).item())
        if log_nu is not None
        else float("nan")
    )
    return {"mu": mu, "sigma": sigma, "amp": amp, "nu": nu}


def _eval_curve(
    t: np.ndarray,
    params: dict[str, float],
    family: str,
    eps: float = 1e-6,
) -> np.ndarray:
    mu = params["mu"]
    sigma = max(params["sigma"], eps)
    amp = params["amp"]
    if family == "gaussian":
        z = (t - mu) / sigma
        return amp * np.exp(-0.5 * z**2)
    if family == "student":
        z2 = ((t - mu) / sigma) ** 2
        nu = max(params.get("nu", 2.0), eps)
        return amp * np.power(1.0 + (z2 / nu), -(nu + 1.0) * 0.5)
    raise ValueError(f"Unknown fit family: {family}")


# ---------------------------------------------------------------------------
# Quality checks (reused from misspelling_batch.py)
# ---------------------------------------------------------------------------


def check_quality(
    word: str,
    changed_indices: list[int],
    fitted: list[dict[str, float]],
    peak_path_indices: list[int],
    pooled_valid: np.ndarray,
    t_norm: np.ndarray,
    fit_family: str,
    n_chars: int,
    n_valid: int,
    mask_start: int,
    mask_end: int,
) -> list[str]:
    problems = []

    # Peak ordering
    for ci in range(1, n_chars):
        if peak_path_indices[ci] < peak_path_indices[ci - 1]:
            problems.append(
                f"Peak order: '{word[ci - 1]}' pos {peak_path_indices[ci - 1]} "
                f">= '{word[ci]}' pos {peak_path_indices[ci]}"
            )

    # Boundary chars
    first_changed = changed_indices[0]
    last_changed = changed_indices[-1]
    boundary_chars: list[int] = []
    if first_changed > 0:
        boundary_chars.append(first_changed - 1)
    if last_changed < n_chars - 1:
        boundary_chars.append(last_changed + 1)

    max_amp = max(f["amp"] for f in fitted) if fitted else 0.0

    for ci in boundary_chars:
        y = pooled_valid[ci, :]
        fit_y = _eval_curve(t_norm, fitted[ci], fit_family)
        ss_res = float(np.sum((y - fit_y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        if r2 < 0.3:
            problems.append(f"Poor fit R²={r2:.3f} for boundary '{word[ci]}' (idx {ci})")

        mu = fitted[ci]["mu"]
        if ci > 0 and ci < n_chars - 1:
            if mu < 0.005 or mu > 0.995:
                problems.append(f"Saturated μ={mu:.4f} for boundary '{word[ci]}' (idx {ci})")

        amp = fitted[ci]["amp"]
        if max_amp > 0 and amp / max_amp < 0.05:
            problems.append(
                f"Low amplitude for boundary '{word[ci]}' (idx {ci}): {amp / max_amp:.1%} of max"
            )

    # Mask too narrow
    if (mask_end - mask_start) < 5:
        problems.append(f"Mask too narrow: {mask_end - mask_start} positions")

    # Mask too large
    mask_fraction = (mask_end - mask_start) / max(n_valid, 1)
    if mask_fraction > 0.80:
        problems.append(f"Mask too large: {mask_fraction:.0%} of valid path")

    return problems


# ---------------------------------------------------------------------------
# Convert augmented 6D path back to raw {x, y, t} at original timestamps
# ---------------------------------------------------------------------------


def augmented_path_to_raw(
    augmented_path: np.ndarray,
    path_mask: np.ndarray,
    original_data: list[dict],
) -> list[dict[str, float]]:
    """Convert augmented 128-point 6D path back to {x, y, t} dicts at original timestamps."""
    valid = path_mask.astype(bool)

    aug_x = augmented_path[valid, 0]
    aug_y = augmented_path[valid, 1]
    aug_log_dt = augmented_path[valid, 5]
    aug_dt = np.expm1(aug_log_dt)
    aug_dt = np.clip(aug_dt, 0.0, None)
    aug_time_rel = np.cumsum(aug_dt)

    t_orig = np.array([p["t"] for p in original_data], dtype=np.float64)
    t0 = t_orig[0]
    t_orig_rel = t_orig - t0

    aug_t_span = aug_time_rel[-1] if len(aug_time_rel) > 0 else 1.0
    aug_t_norm = aug_time_rel / max(aug_t_span, 1e-12)

    orig_t_span = t_orig_rel[-1] if len(t_orig_rel) > 0 else 1.0
    orig_t_norm = t_orig_rel / max(orig_t_span, 1e-12)

    x_interp = interp1d(aug_t_norm, aug_x, kind="linear", fill_value="extrapolate")
    y_interp = interp1d(aug_t_norm, aug_y, kind="linear", fill_value="extrapolate")

    new_x = np.clip(x_interp(orig_t_norm), 0.0, 1.0)
    new_y = np.clip(y_interp(orig_t_norm), 0.0, 1.0)

    return [
        {"x": float(new_x[i]), "y": float(new_y[i]), "t": float(t_orig[i])}
        for i in range(len(t_orig))
    ]


# ---------------------------------------------------------------------------
# Valid position filtering (pre-quality)
# ---------------------------------------------------------------------------


def get_valid_flip_positions(
    word: str,
    fitted: list[dict[str, float]],
    peak_path_indices: list[int],
    pooled_valid: np.ndarray,
    t_norm: np.ndarray,
    fit_family: str,
    n_chars: int,
    n_valid: int,
    path_len: int,
    adjacency: dict[str, list[str]],
) -> list[int]:
    """Return indices of characters that can be safely flipped."""
    max_amp = max(f["amp"] for f in fitted) if fitted else 0.0
    valid_positions: list[int] = []

    for i in range(n_chars):
        ch = word[i].lower()

        # Must have adjacent keys
        if ch not in adjacency or not adjacency[ch]:
            continue

        # Check boundary chars (i-1 and i+1) for fit quality
        boundary_ok = True
        for bi in [i - 1, i + 1]:
            if bi < 0 or bi >= n_chars:
                continue

            # R² check
            y = pooled_valid[bi, :]
            fit_y = _eval_curve(t_norm, fitted[bi], fit_family)
            ss_res = float(np.sum((y - fit_y) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            if r2 < 0.3:
                boundary_ok = False
                break

            # Saturated μ (exempt first/last chars)
            mu = fitted[bi]["mu"]
            if bi > 0 and bi < n_chars - 1:
                if mu < 0.005 or mu > 0.995:
                    boundary_ok = False
                    break

            # Low amplitude
            amp = fitted[bi]["amp"]
            if max_amp > 0 and amp / max_amp < 0.05:
                boundary_ok = False
                break

        if not boundary_ok:
            continue

        # Check mask region size
        left = 0 if i == 0 else peak_path_indices[i - 1]
        right = path_len if i == n_chars - 1 else peak_path_indices[i + 1]
        left = max(0, min(left, path_len - 1))
        right = max(left + 1, min(right, path_len))
        if (right - left) < 5:
            continue

        valid_positions.append(i)

    return valid_positions


# ---------------------------------------------------------------------------
# Flip selection logic
# ---------------------------------------------------------------------------


def select_flips(
    valid_positions: list[int],
    word: str,
    adjacency: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Select which characters to flip from the pre-validated positions.

    Rules:
    - max_flips = len(word) // 3, minimum 1 (for words >= 3 chars)
    - Randomly choose from valid_positions
    - Positions must be at least 2 apart (no adjacent flips)
    - For each position, pick a random adjacent key

    Returns: list of (index, new_char) tuples, sorted by index
    """
    if not valid_positions:
        return []

    max_flips = max(1, len(word) // 3)

    # Shuffle and greedily pick non-adjacent positions
    shuffled = list(valid_positions)
    rng.shuffle(shuffled)

    selected: list[int] = []
    for pos in shuffled:
        if len(selected) >= max_flips:
            break
        # Check at least 2 apart from all already selected
        if any(abs(pos - s) < 2 for s in selected):
            continue
        selected.append(pos)

    selected.sort()

    # For each selected position, pick a random adjacent key
    flips: list[tuple[int, str]] = []
    for pos in selected:
        neighbors = adjacency.get(word[pos].lower(), [])
        if not neighbors:
            continue
        replacement = neighbors[rng.integers(len(neighbors))]
        flips.append((pos, replacement))

    return flips


# ---------------------------------------------------------------------------
# Multi-region mask computation
# ---------------------------------------------------------------------------


def compute_mask_regions(
    flips: list[tuple[int, str]],
    peak_path_indices: list[int],
    n_chars: int,
    path_len: int,
) -> list[tuple[int, int]]:
    """Compute mask regions for each flip position.

    For flip at position i:
    - Left boundary = peak of char at i-1 (or 0 for first char)
    - Right boundary = peak of char at i+1 (or path_len for last char)
    """
    regions: list[tuple[int, int]] = []
    for pos, _ in flips:
        left = 0 if pos == 0 else peak_path_indices[pos - 1]
        right = path_len if pos == n_chars - 1 else peak_path_indices[pos + 1]
        left = max(0, min(left, path_len - 1))
        right = max(left + 1, min(right, path_len))
        regions.append((left, right))
    return regions


# ---------------------------------------------------------------------------
# Batched input collation
# ---------------------------------------------------------------------------


def collate_inputs(
    inputs_list: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pad and stack a list of single-sample processor outputs into a batch."""
    path_coords = torch.cat([inp["path_coords"] for inp in inputs_list], dim=0)

    max_ids = max(inp["input_ids"].shape[1] for inp in inputs_list)
    input_ids = torch.cat(
        [F.pad(inp["input_ids"], (0, max_ids - inp["input_ids"].shape[1])) for inp in inputs_list],
        dim=0,
    )

    max_mask = max(inp["attention_mask"].shape[1] for inp in inputs_list)
    attention_mask = torch.cat(
        [
            F.pad(inp["attention_mask"], (0, max_mask - inp["attention_mask"].shape[1]))
            for inp in inputs_list
        ],
        dim=0,
    )

    return {
        "path_coords": path_coords.to(device),
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


# ---------------------------------------------------------------------------
# CPU worker: curve fitting + flip selection (runs in ProcessPoolExecutor)
# ---------------------------------------------------------------------------

_CPU_DEVICE = torch.device("cpu")


def analyze_sample(
    word: str,
    pooled_valid: np.ndarray,
    valid_indices: np.ndarray,
    t_norm: np.ndarray,
    n_valid: int,
    path_len: int,
    fit_steps: int,
    fit_lr: float,
    rng_seed: int,
) -> dict | None:
    """CPU-only: fit curves, validate positions, select flips, compute masks.

    Returns dict with augmented_word, flips, mask_regions, or None on failure.
    """
    n_chars = len(word)
    rng = np.random.default_rng(rng_seed)
    fit_family = "student"

    # Fit peaks
    fitted: list[dict[str, float]] = []
    peak_path_indices: list[int] = []

    for ci in range(n_chars):
        y = pooled_valid[ci, :]
        mu_seed = 0.5 if n_chars == 1 else ci / float(n_chars - 1)
        sigma_seed = 0.5 / max(n_chars, 1)
        amp_seed = float(y.max()) if y.size else 0.0

        fit = _fit_curve(
            t_norm,
            y,
            mu_seed,
            sigma_seed,
            amp_seed,
            fit_family,
            steps=fit_steps,
            lr=fit_lr,
            device=_CPU_DEVICE,
        )
        fitted.append(fit)

        fit_y = _eval_curve(t_norm, fit, fit_family)
        if np.any(np.isfinite(fit_y)):
            local_peak = int(np.nanargmax(fit_y))
        else:
            local_peak = int(np.argmin(np.abs(t_norm - float(np.clip(fit["mu"], 0.0, 1.0)))))
        peak_path_indices.append(int(valid_indices[local_peak]))

    # Peak ordering check
    for ci in range(1, n_chars):
        if peak_path_indices[ci] < peak_path_indices[ci - 1]:
            return "peak_order"

    # Valid flip positions
    valid_positions = get_valid_flip_positions(
        word,
        fitted,
        peak_path_indices,
        pooled_valid,
        t_norm,
        fit_family,
        n_chars,
        n_valid,
        path_len,
        ADJACENCY,
    )
    if not valid_positions:
        return "no_valid_pos"

    flips = select_flips(valid_positions, word, ADJACENCY, rng)
    if not flips:
        return "no_flips"

    # Build modified word
    augmented_word = list(word)
    for pos, rep in flips:
        augmented_word[pos] = rep
    augmented_word_str = "".join(augmented_word)

    mask_regions = compute_mask_regions(flips, peak_path_indices, n_chars, path_len)

    # Total mask fraction check
    total_masked = sum(end - start for start, end in mask_regions)
    if total_masked / max(n_valid, 1) > 0.80:
        return "mask_fraction"

    return {
        "augmented_word": augmented_word_str,
        "flips": flips,
        "mask_regions": mask_regions,
    }


# ---------------------------------------------------------------------------
# CPU worker: post-processing (blend + smooth + convert)
# ---------------------------------------------------------------------------


def postprocess_sample(
    original_path: np.ndarray,
    reconstructed: np.ndarray,
    path_mask_np: np.ndarray,
    path_data: list[dict],
    mask_regions: list[tuple[int, int]],
    word: str,
    augmented_word: str,
    flips: list[tuple[int, str]],
    sample_idx: int,
) -> dict:
    """CPU-only: convert original and reconstructed paths to raw, build JSONL row.

    Stores both original and fully reconstructed paths so blending can be done
    as a separate post-processing step without re-running inference.
    """
    # Fully reconstructed path: splice reconstructed regions into original
    recon_path = original_path.copy()
    for region_start, region_end in mask_regions:
        recon_path[region_start:region_end, :5] = reconstructed[region_start:region_end, :5]

    raw_data = augmented_path_to_raw(original_path, path_mask_np, path_data)
    raw_augmented = augmented_path_to_raw(recon_path, path_mask_np, path_data)

    flip_info = [{"index": pos, "original": word[pos], "replacement": rep} for pos, rep in flips]
    region_info = [[s, e] for s, e in mask_regions]

    return {
        "word": word,
        "augmented": augmented_word,
        "data": raw_data,
        "augmented_data": raw_augmented,
        "dataset_index": sample_idx,
        "flips": flip_info,
        "mask_regions": region_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _worker_init():
    """Disable CUDA in worker processes to avoid CUDA initialization errors."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch keyboard-proximity augmentation with QWERTY adjacency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to HuggingFace checkpoint directory"
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file (appended to)")
    parser.add_argument(
        "-n", type=int, default=100, help="Number of augmentations to generate (default: 100)"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--random", action="store_true", default=True, help="Sample words randomly (default)"
    )
    mode_group.add_argument(
        "--sequential", action="store_true", help="Process words sequentially through dataset"
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="Transformer layers for attention extraction (default: 4 5 6)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["max", "mean", "sum", "logsumexp"],
        default="mean",
        help="How to aggregate attention across heads (default: mean)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Attention temperature (default: 1.0)"
    )
    parser.add_argument(
        "--fit-steps", type=int, default=200, help="Optimization steps per character (default: 200)"
    )
    parser.add_argument(
        "--fit-lr", type=float, default=0.1, help="Learning rate for curve fitting (default: 0.1)"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="futo-org/swipe.futo.org",
        help="HuggingFace dataset name (default: futo-org/swipe.futo.org)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit dataset to this many samples (default: use entire split)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        help="Skip words shorter than this (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU workers for parallel processing (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Samples per GPU batch (default: 4 * workers)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    sequential_mode = args.sequential
    n_workers = max(1, args.workers)
    batch_size = args.batch_size if args.batch_size > 0 else n_workers * 4
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model + processor ---
    print("1. Loading model and processor...")
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.eval()
    print(f"   {model.config.n_layers} layers, {model.config.n_heads} heads")

    # --- Load dataset ---
    if args.num_samples is not None:
        split_str = f"{args.dataset_split}[:{args.num_samples}]"
        print(f"\n2. Loading dataset ({args.dataset_split}, first {args.num_samples} samples)...")
    else:
        split_str = args.dataset_split
        print(f"\n2. Loading dataset ({args.dataset_split}, full split)...")
    dataset = load_dataset(args.dataset_name, split=split_str)
    n_dataset = len(dataset)
    print(f"   {n_dataset} samples loaded")

    # --- Main batch loop ---
    generated = 0
    skipped_short = 0
    skipped_quality = 0
    skip_reasons: dict[str, int] = {}
    attempted = 0
    seq_idx = 0

    mode_label = "sequential" if sequential_mode else "random"
    pool_label = f"{n_workers} workers" if n_workers > 1 else "1 worker"
    print(
        f"\n3. Generating {args.n} augmentations ({mode_label}, {pool_label}, batch={batch_size})..."
    )

    pool = (
        ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
        )
        if n_workers > 1
        else None
    )

    try:
        with open(output_path, "a", encoding="utf-8") as out_f:
            while generated < args.n:
                # ===== Collect candidate batch =====
                candidates = []  # list of (word, path_data, sample_idx)
                while len(candidates) < batch_size:
                    if sequential_mode:
                        if seq_idx >= n_dataset:
                            break
                        sample_idx = seq_idx
                        seq_idx += 1
                    else:
                        sample_idx = int(rng.integers(n_dataset))

                    attempted += 1
                    example = dataset[sample_idx]
                    word = example["word"].lower()
                    if len(word) < args.min_word_length:
                        skipped_short += 1
                        continue
                    candidates.append((word, example["data"], sample_idx))

                if not candidates:
                    if sequential_mode:
                        print(f"   Dataset exhausted at index {seq_idx}")
                    break

                # ===== GPU: batched attention forward pass =====
                inputs_list = []
                for word, path_data, _ in candidates:
                    inputs_list.append(
                        processor(path_coords=path_data, text=word, return_tensors="pt")
                    )

                batched = collate_inputs(inputs_list, device)
                with torch.no_grad():
                    _, attentions = get_all_layer_attentions(model, batched)

                # ===== Per-sample attention extraction (main thread, fast) =====
                sample_data = []  # data needed for CPU analysis
                for i, (word, path_data, sample_idx) in enumerate(candidates):
                    try:
                        n_chars = len(word)
                        path_len = inputs_list[i]["path_coords"].shape[1]
                        char_len = inputs_list[i]["input_ids"].shape[1]

                        attn_mask = batched["attention_mask"][i]
                        path_mask_tensor = attn_mask[1 : 1 + path_len]
                        path_mask_np = path_mask_tensor.detach().cpu().numpy()

                        layer_attentions: dict[int, np.ndarray] = {}
                        for layer_idx in args.layers:
                            if layer_idx < 0 or layer_idx >= len(attentions):
                                continue
                            attn = attentions[layer_idx][i : i + 1]  # (1, H, S, S)
                            c2p = extract_path_to_char_attention(
                                attn,
                                path_len=path_len,
                                char_len=char_len,
                                aggregation=args.aggregation,
                            )
                            if args.temperature != 1.0:
                                c2p = _apply_temperature(c2p, path_mask_tensor, args.temperature)
                            c2p = c2p[0]  # (char_len, path_len)
                            layer_attentions[layer_idx] = c2p[:n_chars].detach().cpu().numpy()

                        if not layer_attentions:
                            sample_data.append(None)
                            skipped_quality += 1
                            continue

                        attention_stack = np.stack(list(layer_attentions.values()), axis=0)
                        pooled_attention = _pool_layer_attentions(attention_stack, "mean")

                        path_coords_np = inputs_list[i]["path_coords"][0].numpy()
                        _, valid_indices = _filter_path_points(path_coords_np, path_mask_np)
                        pooled_valid = pooled_attention[:, valid_indices]
                        path_coords_filtered = path_coords_np[valid_indices]

                        times, _ = _infer_time_axis(path_coords_filtered)
                        n_valid = len(valid_indices)
                        if n_valid < 2:
                            sample_data.append(None)
                            skipped_quality += 1
                            continue

                        t_min = float(times[0])
                        t_span = float(times[-1] - times[0])
                        t_norm = (times - t_min) / max(t_span, 1e-6)

                        sample_data.append(
                            {
                                "word": word,
                                "path_data": path_data,
                                "sample_idx": sample_idx,
                                "pooled_valid": pooled_valid,
                                "valid_indices": valid_indices,
                                "t_norm": t_norm,
                                "n_valid": n_valid,
                                "path_len": path_len,
                                "path_mask_np": path_mask_np,
                                "original_path": inputs_list[i]["path_coords"][0].numpy().copy(),
                            }
                        )
                    except Exception as e:
                        print(f"   Error extracting attention for '{word}': {e}", file=sys.stderr)
                        sample_data.append(None)
                        skipped_quality += 1

                # Free GPU attention memory
                del attentions, batched

                # ===== CPU: parallel curve fitting + flip selection =====
                valid_data = [(idx, sd) for idx, sd in enumerate(sample_data) if sd is not None]
                if not valid_data:
                    continue

                if pool is not None:
                    futures = []
                    for _, sd in valid_data:
                        seed_i = int(rng.integers(2**63))
                        futures.append(
                            pool.submit(
                                analyze_sample,
                                sd["word"],
                                sd["pooled_valid"],
                                sd["valid_indices"],
                                sd["t_norm"],
                                sd["n_valid"],
                                sd["path_len"],
                                args.fit_steps,
                                args.fit_lr,
                                seed_i,
                            )
                        )
                    analyses = [f.result() for f in futures]
                else:
                    analyses = []
                    for _, sd in valid_data:
                        seed_i = int(rng.integers(2**63))
                        analyses.append(
                            analyze_sample(
                                sd["word"],
                                sd["pooled_valid"],
                                sd["valid_indices"],
                                sd["t_norm"],
                                sd["n_valid"],
                                sd["path_len"],
                                args.fit_steps,
                                args.fit_lr,
                                seed_i,
                            )
                        )

                # Pair valid analyses with their sample data
                recon_items = []
                for (_orig_idx, sd), analysis in zip(valid_data, analyses, strict=False):
                    if isinstance(analysis, str):
                        skip_reasons[analysis] = skip_reasons.get(analysis, 0) + 1
                        continue
                    recon_items.append((sd, analysis))

                if not recon_items:
                    continue

                # ===== GPU: batched reconstruction forward pass =====
                recon_inputs_list = []
                for sd, analysis in recon_items:
                    recon_inputs_list.append(
                        processor(
                            path_coords=sd["path_data"],
                            text=analysis["augmented_word"],
                            return_tensors="pt",
                        )
                    )

                recon_batched = collate_inputs(recon_inputs_list, device)

                # Apply masks per sample
                for j, (_sd, analysis) in enumerate(recon_items):
                    for region_start, region_end in analysis["mask_regions"]:
                        recon_batched["path_coords"][j, region_start:region_end, :] = 0.0

                with torch.no_grad():
                    outputs = model(
                        path_coords=recon_batched["path_coords"],
                        input_ids=recon_batched["input_ids"],
                        attention_mask=recon_batched["attention_mask"],
                        return_dict=True,
                    )

                recon = getattr(outputs, "path_logits", None)
                if recon is None:
                    recon = getattr(outputs, "path_coords_pred", None)
                if recon is None:
                    skipped_quality += len(recon_items)
                    del recon_batched, outputs
                    continue

                recon_np = recon.detach().cpu().numpy()  # (B, 128, D)
                del recon_batched, outputs, recon

                # ===== CPU: parallel post-processing =====
                if pool is not None:
                    futures = []
                    for j, (sd, analysis) in enumerate(recon_items):
                        futures.append(
                            pool.submit(
                                postprocess_sample,
                                sd["original_path"],
                                recon_np[j],
                                sd["path_mask_np"],
                                sd["path_data"],
                                analysis["mask_regions"],
                                sd["word"],
                                analysis["augmented_word"],
                                analysis["flips"],
                                sd["sample_idx"],
                            )
                        )
                    rows = [f.result() for f in futures]
                else:
                    rows = []
                    for j, (sd, analysis) in enumerate(recon_items):
                        rows.append(
                            postprocess_sample(
                                sd["original_path"],
                                recon_np[j],
                                sd["path_mask_np"],
                                sd["path_data"],
                                analysis["mask_regions"],
                                sd["word"],
                                analysis["augmented_word"],
                                analysis["flips"],
                                sd["sample_idx"],
                            )
                        )

                for row in rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    generated += 1
                    if generated >= args.n:
                        break

                out_f.flush()
                total_skipped = skipped_short + skipped_quality + sum(skip_reasons.values())
                reasons_str = ", ".join(f"{v} {k}" for k, v in sorted(skip_reasons.items()))
                print(
                    f"   [{generated}/{args.n}] "
                    f"(attempted {attempted}, "
                    f"skipped {total_skipped}: "
                    f"{skipped_short} short, "
                    f"{skipped_quality} quality, "
                    f"{reasons_str})"
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    print(f"\nDone! Generated {generated} augmentations.")
    print(f"  Attempted: {attempted}")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (quality/error): {skipped_quality}")
    for reason, count in sorted(skip_reasons.items()):
        print(f"  Skipped ({reason}): {count}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
