#!/usr/bin/env python3
"""Blend original and augmented swipe paths from keyboard augmentation output.

Reads JSONL with 'data' (original) and 'augmented_data' (reconstructed) keys,
applies a blend factor and optional Savitzky-Golay boundary smoothing, and writes
a new JSONL with the blended path as 'data'.

Usage:
  uv run blend-augmented \
      --input augmented_keyboard.jsonl \
      --output blended_keyboard.jsonl \
      --blend 0.5
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.signal import savgol_filter


def blend_paths(
    original: list[dict],
    augmented: list[dict],
    blend: float,
) -> list[dict]:
    """Blend original and augmented paths point-by-point.

    Args:
        original: List of {x, y, t} dicts (original path).
        augmented: List of {x, y, t} dicts (fully reconstructed path).
        blend: Blend factor. 1.0 = fully augmented, 0.0 = fully original.

    Returns:
        List of {x, y, t} dicts with blended x, y and original t.
    """
    result = []
    for orig, aug in zip(original, augmented, strict=True):
        result.append(
            {
                "x": blend * aug["x"] + (1.0 - blend) * orig["x"],
                "y": blend * aug["y"] + (1.0 - blend) * orig["y"],
                "t": orig["t"],
            }
        )
    return result


def smooth_boundaries(
    blended: list[dict],
    original: list[dict],
    augmented: list[dict],
    smooth_window: int,
    smooth_polyorder: int,
) -> list[dict]:
    """Apply Savitzky-Golay smoothing near points where paths diverge.

    Finds boundary regions where original and augmented differ significantly,
    and smooths the blended path in those transition zones.
    """
    n = len(blended)
    if n < smooth_window:
        return blended

    x_arr = np.array([p["x"] for p in blended])
    y_arr = np.array([p["y"] for p in blended])

    orig_x = np.array([p["x"] for p in original])
    orig_y = np.array([p["y"] for p in original])
    aug_x = np.array([p["x"] for p in augmented])
    aug_y = np.array([p["y"] for p in augmented])

    # Find where augmented differs from original (mask region boundaries)
    diff = np.sqrt((aug_x - orig_x) ** 2 + (aug_y - orig_y) ** 2)
    threshold = 0.01
    in_mask = diff > threshold

    # Find boundary transitions
    transitions = np.where(np.diff(in_mask.astype(int)) != 0)[0]
    half_w = smooth_window // 2

    for t_idx in transitions:
        start = max(0, t_idx - half_w)
        end = min(n, t_idx + half_w + 1)
        if end - start >= smooth_window:
            x_arr[start:end] = savgol_filter(x_arr[start:end], smooth_window, smooth_polyorder)
            y_arr[start:end] = savgol_filter(y_arr[start:end], smooth_window, smooth_polyorder)

    return [
        {
            "x": float(np.clip(x_arr[i], 0, 1)),
            "y": float(np.clip(y_arr[i], 0, 1)),
            "t": blended[i]["t"],
        }
        for i in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend original and augmented swipe paths from keyboard augmentation output."
    )
    parser.add_argument("--input", required=True, help="Input JSONL with data and augmented_data")
    parser.add_argument("--output", required=True, help="Output JSONL with blended data")
    parser.add_argument(
        "--blend",
        type=float,
        default=0.5,
        help="Blend factor: 1.0 = fully augmented, 0.0 = fully original (default: 0.5)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Savitzky-Golay window for boundary smoothing (0 to disable, default: 0)",
    )
    parser.add_argument(
        "--keep-both",
        action="store_true",
        help="Keep both data and augmented_data in output (default: only blended data)",
    )
    args = parser.parse_args()

    smooth_window = args.smooth_window
    if smooth_window > 0 and smooth_window % 2 == 0:
        smooth_window += 1
    smooth_polyorder = min(3, smooth_window - 1) if smooth_window > 0 else 0

    n_processed = 0
    n_skipped = 0

    with (
        open(args.input, encoding="utf-8") as in_f,
        open(args.output, "w", encoding="utf-8") as out_f,
    ):
        for line in in_f:
            row = json.loads(line)

            if "augmented_data" not in row:
                # Pass through rows without augmented_data (e.g. from older format)
                out_f.write(line)
                n_skipped += 1
                continue

            original = row["data"]
            augmented = row["augmented_data"]

            if len(original) != len(augmented):
                print(
                    f"Warning: length mismatch at line {n_processed + n_skipped + 1}, skipping",
                    file=sys.stderr,
                )
                n_skipped += 1
                continue

            blended = blend_paths(original, augmented, args.blend)

            if smooth_window > 0:
                blended = smooth_boundaries(
                    blended, original, augmented, smooth_window, smooth_polyorder
                )

            out_row = {
                "word": row["word"],
                "augmented": row["augmented"],
                "data": blended,
                "dataset_index": row.get("dataset_index"),
                "flips": row.get("flips"),
                "mask_regions": row.get("mask_regions"),
            }

            if args.keep_both:
                out_row["original_data"] = original
                out_row["augmented_data"] = augmented

            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_processed += 1

    print(f"Done! Processed {n_processed} rows, skipped {n_skipped}.")
    print(f"  Blend factor: {args.blend}")
    if smooth_window > 0:
        print(f"  Smoothing: window={smooth_window}, polyorder={smooth_polyorder}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
