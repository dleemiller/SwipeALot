#!/usr/bin/env python3
"""Build median dataset letter positions from start/end position JSONs.

Usage:
  uv run letter-positions --output demos/encoder_keyboard_positions_dataset_median.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for start in candidates:
        for parent in [start] + list(start.parents):
            if (parent / "pyproject.toml").exists():
                return parent
    return Path.cwd()


def _load_points(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Positions JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Positions JSON must be an object mapping letter -> list of points")
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if not isinstance(key, str) or len(key) != 1 or not key.isalpha():
            continue
        if not isinstance(value, list):
            continue
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
            out[key.lower()] = np.asarray(points, dtype=np.float64)
    return out


def main() -> None:
    repo_root = _find_repo_root()
    parser = argparse.ArgumentParser(
        description="Build median letter positions from dataset-derived start/end JSONs."
    )
    parser.add_argument(
        "--start-positions",
        type=str,
        default=str(repo_root / "demos" / "encoder_keyboard_positions_dataset_start.json"),
        help="JSON mapping letter -> start positions (default: demos/encoder_keyboard_positions_dataset_start.json)",
    )
    parser.add_argument(
        "--end-positions",
        type=str,
        default=str(repo_root / "demos" / "encoder_keyboard_positions_dataset_end.json"),
        help="JSON mapping letter -> end positions (default: demos/encoder_keyboard_positions_dataset_end.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "demos" / "encoder_keyboard_positions_dataset_median.json"),
        help="Output JSON path for median positions (default: demos/encoder_keyboard_positions_dataset_median.json)",
    )
    args = parser.parse_args()

    start_positions = _load_points(Path(args.start_positions))
    end_positions = _load_points(Path(args.end_positions))

    letters = sorted(set(start_positions.keys()) | set(end_positions.keys()))
    out: dict[str, dict[str, float | int]] = {}
    for letter in letters:
        points = []
        if letter in start_positions:
            points.append(start_positions[letter])
        if letter in end_positions:
            points.append(end_positions[letter])
        if not points:
            continue
        stacked = np.vstack(points)
        med = np.median(stacked, axis=0)
        out[letter] = {"x": float(med[0]), "y": float(med[1]), "count": int(stacked.shape[0])}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)

    print(f"Saved median positions to: {output_path}")


if __name__ == "__main__":
    main()
