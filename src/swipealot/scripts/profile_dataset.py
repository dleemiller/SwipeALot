#!/usr/bin/env python3
"""Profile time delta distribution in swipe path dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def profile_time_deltas(dataset_name: str = "futo-org/swipe.futo.org", n_samples: int = 10000):
    """Analyze time delta distribution in dataset."""
    print(f"Loading dataset: {dataset_name}")
    print(f"Profiling first {n_samples} training samples...")

    dataset = load_dataset(dataset_name, split=f"train[:{n_samples}]")

    all_dts = []
    path_lengths = []
    durations = []

    for sample in dataset:
        data_points = sample["data"]

        if len(data_points) < 2:
            continue

        # Extract timestamps
        timestamps = [p["t"] for p in data_points]

        # Compute time deltas
        dts = np.diff(timestamps)
        all_dts.extend(dts)

        # Track path length and duration
        path_lengths.append(len(data_points))
        durations.append(timestamps[-1] - timestamps[0])

    dts_array = np.array(all_dts)
    path_lengths = np.array(path_lengths)
    durations = np.array(durations)

    # Statistics
    print("\n" + "=" * 60)
    print("TIME DELTA (dt) STATISTICS")
    print("=" * 60)
    print(f"Total samples analyzed: {len(dataset)}")
    print(f"Total time deltas: {len(dts_array):,}")
    print()
    print("dt statistics (milliseconds):")
    print(f"  Min:    {dts_array.min():.4f} ms")
    print(f"  Max:    {dts_array.max():.4f} ms")
    print(f"  Mean:   {dts_array.mean():.4f} ms")
    print(f"  Median: {np.median(dts_array):.4f} ms")
    print(f"  Std:    {dts_array.std():.4f} ms")
    print()
    print("Percentiles:")
    print(f"  P01:  {np.percentile(dts_array, 1):.4f} ms")
    print(f"  P05:  {np.percentile(dts_array, 5):.4f} ms")
    print(f"  P25:  {np.percentile(dts_array, 25):.4f} ms")
    print(f"  P50:  {np.percentile(dts_array, 50):.4f} ms")
    print(f"  P75:  {np.percentile(dts_array, 75):.4f} ms")
    print(f"  P95:  {np.percentile(dts_array, 95):.4f} ms")
    print(f"  P99:  {np.percentile(dts_array, 99):.4f} ms")
    print()

    # Check for outliers
    outliers_low = (dts_array < 0.1).sum()
    outliers_high = (dts_array > 1000).sum()
    print(f"Outliers:")
    print(f"  < 0.1 ms:    {outliers_low:,} ({100 * outliers_low / len(dts_array):.2f}%)")
    print(f"  > 1000 ms:   {outliers_high:,} ({100 * outliers_high / len(dts_array):.2f}%)")

    print("\n" + "=" * 60)
    print("PATH LENGTH STATISTICS")
    print("=" * 60)
    print(f"  Min:    {path_lengths.min()}")
    print(f"  Max:    {path_lengths.max()}")
    print(f"  Mean:   {path_lengths.mean():.1f}")
    print(f"  Median: {np.median(path_lengths):.0f}")

    print("\n" + "=" * 60)
    print("SWIPE DURATION STATISTICS")
    print("=" * 60)
    print(f"  Min:    {durations.min():.2f} ms")
    print(f"  Max:    {durations.max():.2f} ms")
    print(f"  Mean:   {durations.mean():.2f} ms")
    print(f"  Median: {np.median(durations):.2f} ms")

    # Create visualizations
    output_dir = Path("profiling_results")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: dt distribution (log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(np.log1p(dts_array), bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("log1p(dt) [log(1 + milliseconds)]")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Time Deltas (log scale)")
    ax1.grid(True, alpha=0.3)

    # Add percentile lines
    for p, label in [(50, "P50"), (95, "P95"), (99, "P99")]:
        val = np.log1p(np.percentile(dts_array, p))
        ax1.axvline(val, color="red", linestyle="--", alpha=0.5, label=label)
    ax1.legend()

    # Plot 2: dt distribution (linear scale, zoomed to reasonable range)
    dt_p99 = np.percentile(dts_array, 99)
    dts_clipped = dts_array[dts_array <= dt_p99]
    ax2.hist(dts_clipped, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("dt [milliseconds]")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Distribution of Time Deltas (0 to P99={dt_p99:.1f} ms)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "dt_distribution.png", dpi=150)
    print(f"\n✓ Saved plot: {output_dir / 'dt_distribution.png'}")
    plt.close()

    # Plot 3: Path length distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(path_lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Path Length (number of points)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Path Lengths")
    ax.axvline(128, color="red", linestyle="--", label="max_path_len=128", linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "path_length_distribution.png", dpi=150)
    print(f"✓ Saved plot: {output_dir / 'path_length_distribution.png'}")
    plt.close()

    # Plot 4: Duration distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(durations, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Swipe Duration [milliseconds]")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Swipe Durations")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "duration_distribution.png", dpi=150)
    print(f"✓ Saved plot: {output_dir / 'duration_distribution.png'}")
    plt.close()

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    dt_p01 = np.percentile(dts_array, 1)
    dt_p99 = np.percentile(dts_array, 99)

    print(f"\nBased on analysis of {n_samples} samples:")
    print()
    print("Recommended dt clamping range:")
    print(f"  dt_clamp_min: {max(0.1, dt_p01):.2f} ms  (P01={dt_p01:.2f} ms)")
    print(f"  dt_clamp_max: {min(1000.0, dt_p99):.2f} ms  (P99={dt_p99:.2f} ms)")
    print()

    if outliers_low + outliers_high > 0.01 * len(dts_array):
        print("⚠ Warning: >1% of time deltas are outliers")
        print("  Consider adjusting clamping range or investigating data quality")
    else:
        print("✓ Outlier rate is acceptable (<1%)")

    paths_over_128 = (path_lengths > 128).sum()
    paths_under_128 = (path_lengths < 128).sum()
    print()
    print(f"Path length distribution:")
    print(f"  > 128 points: {paths_over_128:,} ({100 * paths_over_128 / len(path_lengths):.1f}%) - will downsample")
    print(f"  < 128 points: {paths_under_128:,} ({100 * paths_under_128/ len(path_lengths):.1f}%) - will upsample")
    print(f"  = 128 points: {(path_lengths == 128).sum():,} ({100 * (path_lengths == 128).sum() / len(path_lengths):.1f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Profile swipe path time deltas")
    parser.add_argument(
        "--dataset",
        type=str,
        default="futo-org/swipe.futo.org",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of samples to profile"
    )
    args = parser.parse_args()

    profile_time_deltas(args.dataset, args.n_samples)


if __name__ == "__main__":
    main()
