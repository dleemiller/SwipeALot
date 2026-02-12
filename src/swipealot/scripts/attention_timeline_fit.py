#!/usr/bin/env python3
"""Fit per-character curves to attention timelines (selected layers only).

Usage:
  uv run attention-fit --checkpoint checkpoints/base_20251228_145048/checkpoint-69000 --word-index 166
  uv run attention-fit --checkpoint checkpoints/base_20251228_145048/checkpoint-69000 --word "swanland" --layers 2 3 4 5 6 11 --fit-family lorentzian
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

from swipealot.analysis.attention_capture import get_all_layer_attentions
from swipealot.analysis.attention_extractor import extract_path_to_char_attention
from swipealot.analysis.attention_visualizer import (
    _filter_path_points,
    _infer_time_axis,
    _pool_layer_attentions,
)


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
) -> tuple[float, float, float]:
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
    alpha_raw = None
    log_nu = None
    if family == "skew-normal":
        alpha_raw = torch.tensor(0.0, device=device, requires_grad=True)
        params.append(alpha_raw)
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
        elif family == "lorentzian":
            pred = amp * (1.0 / (1.0 + ((t_t - mu) / sigma) ** 2))
        elif family == "student":
            nu = torch.exp(log_nu).clamp_min(1e-4) + 1.0
            z2 = ((t_t - mu) / sigma) ** 2
            pred = amp * torch.pow(1.0 + (z2 / nu), -(nu + 1.0) * 0.5)
        elif family == "skew-normal":
            alpha = alpha_raw
            z = (t_t - mu) / sigma
            phi = torch.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
            cdf = 0.5 * (1.0 + torch.erf(alpha * z / np.sqrt(2.0)))
            pred = amp * 2.0 * phi * cdf
        else:
            raise ValueError(f"Unknown fit family: {family}")
        loss = torch.mean((pred - y_t) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    mu = float(torch.sigmoid(mu_raw).item())
    sigma = float(torch.exp(log_sigma_param).clamp_min(1e-4).item())
    amp = float(torch.exp(log_amp).item())
    alpha = float(alpha_raw.item()) if alpha_raw is not None else float("nan")
    nu = (
        float((torch.exp(log_nu).clamp_min(1e-4) + 1.0).item())
        if log_nu is not None
        else float("nan")
    )
    log_mu = float("nan")
    log_sigma = float("nan")
    return {
        "mu": mu,
        "sigma": sigma,
        "amp": amp,
        "alpha": alpha,
        "nu": nu,
        "log_mu": log_mu,
        "log_sigma": log_sigma,
    }


def _erf_np(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf)(x)


def _load_letter_positions(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Positions JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Positions JSON must map letter -> position")
    out: dict[str, tuple[float, float]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or len(key) != 1 or not key.isalpha():
            continue
        if isinstance(value, dict):
            x = float(value.get("x", 0.0))
            y = float(value.get("y", 0.0))
            out[key.lower()] = (x, y)
            continue
        if isinstance(value, list):
            points = []
            for item in value:
                if isinstance(item, dict):
                    x = float(item.get("x", 0.0))
                    y = float(item.get("y", 0.0))
                else:
                    arr = np.asarray(item, dtype=np.float64)
                    if arr.shape[0] < 2:
                        continue
                    x, y = float(arr[0]), float(arr[1])
                points.append([x, y])
            if points:
                med = np.median(np.asarray(points), axis=0)
                out[key.lower()] = (float(med[0]), float(med[1]))
    return out


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
    if family == "lorentzian":
        z = (t - mu) / sigma
        return amp * (1.0 / (1.0 + z**2))
    if family == "student":
        z2 = ((t - mu) / sigma) ** 2
        nu = max(params.get("nu", 2.0), eps)
        return amp * np.power(1.0 + (z2 / nu), -(nu + 1.0) * 0.5)
    if family == "skew-normal":
        z = (t - mu) / sigma
        alpha = params.get("alpha", 0.0)
        phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        cdf = 0.5 * (1.0 + _erf_np(alpha * z / np.sqrt(2.0)))
        return amp * 2.0 * phi * cdf
    raise ValueError(f"Unknown fit family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit per-character curves to attention timelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a HuggingFace checkpoint directory (e.g. .../checkpoint-10)",
    )

    word_group = parser.add_mutually_exclusive_group(required=True)
    word_group.add_argument("--word-index", type=int, help="Index of word in validation dataset")
    word_group.add_argument("--word", type=str, help="Specific word to find in validation set")

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--last-k-layers",
        type=int,
        default=3,
        help="Use the last K transformer layers (default: 3)",
    )
    layer_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Explicit transformer layers to visualize",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["max", "mean", "sum", "logsumexp"],
        default="mean",
        help="How to aggregate attention across heads (default: mean)",
    )
    parser.add_argument(
        "--layer-pooling",
        type=str,
        choices=["max", "mean", "sum", "logsumexp"],
        default="mean",
        help="How to pool across selected layers (default: mean)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sharpen attention over path points (<1.0 sharper, >1.0 flatter)",
    )
    parser.add_argument(
        "--fit-family",
        type=str,
        choices=["gaussian", "lorentzian", "student", "skew-normal"],
        default="gaussian",
        help="Curve family to fit per character (default: gaussian)",
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=200,
        help="Optimization steps per character (default: 200)",
    )
    parser.add_argument(
        "--fit-lr",
        type=float,
        default=0.1,
        help="Learning rate for curve fitting (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/attention_fits",
        help="Output directory for visualizations (default: visualizations/attention_fits)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV path for fitted parameters",
    )
    parser.add_argument(
        "--letter-positions",
        type=str,
        default=None,
        help="Optional JSON mapping letter -> median x/y for overlay on path plot",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)",
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
        default=100000,
        help="Number of samples to load from dataset when searching by word (default: 100000)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        print(f"Error: checkpoint must be a directory: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.eval()

    if args.word_index is not None:
        dataset = load_dataset(
            args.dataset_name,
            split=f"{args.dataset_split}[{args.word_index}:{args.word_index + 1}]",
        )
        if len(dataset) == 0:
            print(
                f"Error: Index {args.word_index} out of range for {args.dataset_split}",
                file=sys.stderr,
            )
            sys.exit(1)
        example = dataset[0]
        word = example["word"].lower()
    else:
        dataset = load_dataset(
            args.dataset_name, split=f"{args.dataset_split}[:{args.num_samples}]"
        )
        example = None
        for sample in dataset:
            if sample["word"].lower() == args.word.lower():
                example = sample
                break
        if example is None:
            print(
                f"Error: Word '{args.word}' not found in first {args.num_samples} samples",
                file=sys.stderr,
            )
            sys.exit(1)
        word = example["word"].lower()

    path_data = example["data"]
    inputs = processor(path_coords=path_data, text=word, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    path_len = inputs["path_coords"].shape[1]
    char_len = inputs["input_ids"].shape[1]
    n_chars = min(len(word), char_len)

    attn_mask = inputs["attention_mask"][0]
    path_mask_tensor = attn_mask[1 : 1 + path_len]
    path_mask = path_mask_tensor.detach().cpu().numpy()

    _, attentions = get_all_layer_attentions(model, inputs, print_fallback_note=True)

    all_layer_attentions: dict[int, np.ndarray] = {}
    for layer_idx, attn in enumerate(attentions):
        char_to_path = extract_path_to_char_attention(
            attn, path_len=path_len, char_len=char_len, aggregation=args.aggregation
        )
        char_to_path = _apply_temperature(
            char_to_path, path_mask_tensor, temperature=args.temperature
        )[0]
        all_layer_attentions[layer_idx] = char_to_path[:n_chars].detach().cpu().numpy()

    if args.layers is not None:
        layer_indices = [idx for idx in args.layers if 0 <= idx < len(all_layer_attentions)]
        if not layer_indices:
            print("Error: --layers did not include any valid indices", file=sys.stderr)
            sys.exit(1)
        layer_indices = sorted(set(layer_indices))
    else:
        last_k = min(max(int(args.last_k_layers), 1), len(all_layer_attentions))
        layer_indices = list(range(len(all_layer_attentions) - last_k, len(all_layer_attentions)))

    layer_attentions = {k: all_layer_attentions[k] for k in layer_indices}

    attention_stack = np.stack([attn for attn in layer_attentions.values()], axis=0)
    pooled_attention = _pool_layer_attentions(attention_stack, args.layer_pooling)

    path_coords = inputs["path_coords"][0].detach().cpu().numpy()
    path_coords_filtered, valid_indices = _filter_path_points(path_coords, path_mask)
    if path_mask is not None:
        pooled_attention = pooled_attention[:, valid_indices]

    times, time_label = _infer_time_axis(path_coords_filtered)
    if len(times) < 2:
        print("Error: Not enough path points to fit timeline", file=sys.stderr)
        sys.exit(1)
    t_min = float(times[0])
    t_span = float(times[-1] - times[0])
    t_norm = (times - t_min) / max(t_span, 1e-6)

    csv_rows = []
    fitted = []
    for idx in range(n_chars):
        y = pooled_attention[idx, :]
        mu_seed = 0.5 if n_chars == 1 else idx / float(n_chars - 1)
        sigma_seed = 0.5 / max(n_chars, 1)
        amp_seed = float(y.max()) if y.size else 0.0
        fit = _fit_curve(
            t_norm,
            y,
            mu_seed,
            sigma_seed,
            amp_seed,
            args.fit_family,
            steps=args.fit_steps,
            lr=args.fit_lr,
            device=device,
        )
        mu_time = t_min + fit["mu"] * t_span
        sigma_time = fit["sigma"] * t_span
        fitted.append(fit)
        csv_rows.append(
            {
                "char": word[idx],
                "pos": idx,
                "mu_norm": fit["mu"],
                "sigma_norm": fit["sigma"],
                "amp": fit["amp"],
                "alpha": fit["alpha"],
                "nu": fit["nu"],
                "log_mu": fit["log_mu"],
                "log_sigma": fit["log_sigma"],
                "fit_family": args.fit_family,
                "mu_time": mu_time,
                "sigma_time": sigma_time,
                "mu_seed": mu_seed,
                "sigma_seed": sigma_seed,
            }
        )

    if args.output_csv is not None:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    import matplotlib.pyplot as plt

    fig, (ax_time, ax_xy) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_chars))

    peak_points = []
    for idx in range(n_chars):
        char = word[idx]
        y = pooled_attention[idx, :]
        fit_y = _eval_curve(t_norm, fitted[idx], args.fit_family)

        if not np.any(np.isfinite(fit_y)):
            mu_fallback = float(np.clip(fitted[idx]["mu"], 0.0, 1.0))
            peak_idx = int(np.argmin(np.abs(t_norm - mu_fallback)))
        else:
            peak_idx = int(np.nanargmax(fit_y))

        peak_t_norm = float(np.clip(t_norm[peak_idx], 0.0, 1.0))
        peak_time = t_min + peak_t_norm * t_span
        peak_val = float(fit_y[peak_idx]) if np.isfinite(fit_y[peak_idx]) else float("nan")
        peak_points.append((peak_idx, peak_time, peak_val))

        ax_time.plot(
            times,
            y,
            linewidth=2.5,
            label=f"'{char}' (pos {idx})",
            color=colors[idx],
            alpha=0.85,
            marker="o",
            markersize=3,
            markeredgewidth=0.4,
            markeredgecolor="white",
        )
        ax_time.plot(
            times,
            fit_y,
            linewidth=2.0,
            color=colors[idx],
            alpha=0.9,
            linestyle="--",
        )
        ax_time.scatter(
            [peak_time],
            [peak_val],
            color=colors[idx],
            s=40,
            edgecolors="black",
            linewidths=0.6,
            zorder=11,
        )

    all_stack = np.stack([pooled_attention[i, :] for i in range(n_chars)], axis=0)
    ax_time.plot(
        times,
        all_stack.max(axis=0),
        linewidth=3,
        label="Max across all tokens",
        color="black",
        alpha=0.85,
        linestyle=":",
        zorder=10,
    )
    ax_time.set_xlabel(time_label, fontsize=12, fontweight="bold")
    ax_time.set_ylabel("Attention Score", fontsize=12, fontweight="bold")
    ax_time.set_title(
        f'Attention Timeline ({args.fit_family.capitalize()} Fits): "{word}" '
        f"({args.layer_pooling.capitalize()} across {len(layer_attentions)} layers)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax_time.grid(True, alpha=0.3, linewidth=0.5)
    ax_time.set_ylim(bottom=0)
    ax_time.legend(loc="best", fontsize=9, framealpha=0.9)

    ax_xy.plot(
        path_coords_filtered[:, 0],
        path_coords_filtered[:, 1],
        color="gray",
        linewidth=2.0,
        alpha=0.6,
        zorder=1,
    )
    if args.letter_positions is not None:
        try:
            letter_positions = _load_letter_positions(Path(args.letter_positions))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            letter_positions = {}
        for letter, (x, y) in sorted(letter_positions.items()):
            ax_xy.text(
                x,
                y,
                letter,
                fontsize=9,
                ha="center",
                va="center",
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
                zorder=3,
            )
    for idx, (peak_idx, _, _) in enumerate(peak_points):
        x = path_coords_filtered[peak_idx, 0]
        y = path_coords_filtered[peak_idx, 1]
        ax_xy.scatter(
            [x],
            [y],
            s=60,
            color=colors[idx],
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
    ax_xy.set_xlabel("X", fontsize=12, fontweight="bold")
    ax_xy.set_ylabel("Y", fontsize=12, fontweight="bold")
    ax_xy.set_title("Swipe Path with Fit Peaks", fontsize=14, fontweight="bold", pad=20)
    ax_xy.set_xlim(0, 1)
    ax_xy.set_ylim(0, 1)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    out_path = output_dir / f"{word}_timeline_fit_{args.fit_family}_{args.layer_pooling}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    if args.output_csv is not None:
        print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
