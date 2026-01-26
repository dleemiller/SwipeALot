#!/usr/bin/env python3
"""CLI tool for generating attention visualizations on swipe paths.

Usage:
    uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word-index 10
    uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word "hello"
    uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word-index 10 --last-k-layers 3
    uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word-index 10 --layers 2 3 4 5 6 11
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

from swipealot.analysis import (
    create_attention_timeline_plot,
    create_layer_comparison_grid,
    create_layer_pooled_visualization,
    create_single_layer_timeline_plot,
    create_summary_visualization,
    extract_path_to_char_attention,
)
from swipealot.analysis.attention_capture import get_all_layer_attentions


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention visualizations for swipe keyboard model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize word at index 10 from validation set
  uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word-index 10

  # Visualize specific word
  uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word "hello"

  # Specify last-k layers, temperature, and output directory
  uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word-index 5 --last-k-layers 3 --temperature 0.6 --output visualizations/attention/custom

  # Specify explicit layers for comparison
  uv run attention-map --checkpoint checkpoints/base_20251217_113408/checkpoint-10 --word-index 5 --layers 2 3 4 5 6 11 --temperature 0.6 --output visualizations/attention/custom
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a HuggingFace checkpoint directory (e.g. .../checkpoint-10)",
    )

    # Word selection: either by index or by word string
    word_group = parser.add_mutually_exclusive_group(required=True)
    word_group.add_argument(
        "--word-index",
        type=int,
        help="Index of word in validation dataset to visualize",
    )
    word_group.add_argument(
        "--word",
        type=str,
        help="Specific word to find and visualize in validation set",
    )

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--last-k-layers",
        type=int,
        default=3,
        help="Use the last K transformer layers for layer comparison (default: 3)",
    )
    layer_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Explicit transformer layers to visualize",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sharpen attention over path points (<1.0 sharper, >1.0 flatter)",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["max", "mean", "sum", "logsumexp"],
        default="max",
        help="How to aggregate attention across heads (default: max)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/attention",
        help="Output directory for visualizations (default: visualizations/attention)",
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

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    if not checkpoint_path.is_dir():
        print(
            f"Error: checkpoint must be a directory (got file): {checkpoint_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Attention Map Visualization")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load HF checkpoint (model + processor)
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.eval()

    config = model.config
    print(f"   Model: {config.n_layers} layers, {config.n_heads} heads")
    print(f"   Max path len: {config.max_path_len}")
    print(f"   Max char len: {config.max_char_len}")
    print(f"   Vocab size: {config.vocab_size}")

    # 4. Load sample from dataset
    print(f"\n4. Loading sample from {args.dataset_split} set...")

    if args.word_index is not None:
        # Load specific index
        dataset = load_dataset(
            args.dataset_name,
            split=f"{args.dataset_split}[{args.word_index}:{args.word_index + 1}]",
        )
        if len(dataset) == 0:
            print(
                f"Error: Index {args.word_index} out of range for {args.dataset_split} set",
                file=sys.stderr,
            )
            sys.exit(1)
        example = dataset[0]
        word = example["word"].lower()
        print(f"   Word at index {args.word_index}: '{word}'")

    else:
        # Search for specific word
        print(f"   Searching for word '{args.word}' in first {args.num_samples} samples...")
        dataset = load_dataset(
            args.dataset_name, split=f"{args.dataset_split}[:{args.num_samples}]"
        )

        example = None
        for idx, sample in enumerate(dataset):
            if sample["word"].lower() == args.word.lower():
                example = sample
                print(f"   Found '{args.word}' at index {idx}")
                break

        if example is None:
            print(
                f"Error: Word '{args.word}' not found in first {args.num_samples} samples",
                file=sys.stderr,
            )
            sys.exit(1)

        word = example["word"].lower()

    # 5. Process sample
    print(f"\n5. Processing word: '{word}'")
    path_data = example["data"]
    print(f"   Path length: {len(path_data)} points")

    inputs = processor(path_coords=path_data, text=word, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    path_len = inputs["path_coords"].shape[1]
    char_len = inputs["input_ids"].shape[1]
    print(f"   Model path_len={path_len}, char_len={char_len}")

    # Extract masks for plotting
    attn_mask = inputs["attention_mask"][0]
    path_mask_tensor = attn_mask[1 : 1 + path_len]
    path_mask = path_mask_tensor.detach().cpu().numpy()

    # 6. Extract attention from all layers (native HF output_attentions if available, else hook capture)
    print(f"\n6. Extracting attention from all {config.n_layers} layers...")
    outputs, attentions = get_all_layer_attentions(model, inputs, print_fallback_note=True)

    # Extract char→path attention with specified head aggregation for all layers
    all_layer_attentions = {}
    for layer_idx, attn in enumerate(attentions):
        char_to_path = extract_path_to_char_attention(
            attn, path_len=path_len, char_len=char_len, aggregation=args.aggregation
        )
        char_to_path = _apply_temperature(
            char_to_path, path_mask_tensor, temperature=args.temperature
        )[0]  # remove batch dim
        # Restrict to actual characters in the provided word (exclude EOS + padding)
        n_chars = min(len(word), char_len)
        all_layer_attentions[layer_idx] = char_to_path[:n_chars].detach().cpu().numpy()

    print(f"   Extracted char→path attention for {len(all_layer_attentions)} layers")

    # Also keep attention for selected layers for comparison visualization
    if args.layers is not None:
        layer_indices = [idx for idx in args.layers if 0 <= idx < len(all_layer_attentions)]
        if not layer_indices:
            print(
                "Error: --layers did not include any valid indices for this model",
                file=sys.stderr,
            )
            sys.exit(1)
        layer_indices = sorted(set(layer_indices))
        last_k = None
    else:
        last_k = min(max(int(args.last_k_layers), 1), len(all_layer_attentions))
        layer_indices = list(range(len(all_layer_attentions) - last_k, len(all_layer_attentions)))
    layer_attentions = {k: all_layer_attentions[k] for k in layer_indices}
    print(f"   Using layers {list(layer_attentions.keys())} for layer comparison grid")

    # 7. Create visualizations
    print("\n7. Creating visualizations...")

    # Layer comparison grid (first 3 characters or all if word is short)
    n_chars = min(3, len(word))
    print(f"   - Layer comparison grid ({n_chars} characters)...")
    grid_path = output_dir / f"{word}_layer_comparison.png"

    import numpy as np

    fig1 = create_layer_comparison_grid(
        layer_attentions=layer_attentions,
        path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
        word=word,
        char_indices=list(range(n_chars)),
        save_path=str(grid_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {grid_path}")

    import matplotlib.pyplot as plt

    plt.close(fig1)

    # Summary visualization
    print("   - Summary visualization...")
    summary_path = output_dir / f"{word}_summary.png"
    fig2 = create_summary_visualization(
        layer_attentions=layer_attentions,
        path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
        word=word,
        save_path=str(summary_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {summary_path}")
    plt.close(fig2)

    # Layer-pooled visualization (across all layers)
    print(f"   - Layer-pooled visualization ({args.aggregation} across all layers)...")
    pooled_path = output_dir / f"{word}_layer_pooled_{args.aggregation}.png"
    fig3 = create_layer_pooled_visualization(
        layer_attentions=all_layer_attentions,
        path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
        word=word,
        pooling_method=args.aggregation,
        save_path=str(pooled_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {pooled_path}")
    plt.close(fig3)

    # Layer-pooled visualization (selected layers only)
    print(f"   - Selected-layer pooled visualization ({args.aggregation})...")
    selected_pooled_path = output_dir / f"{word}_layer_pooled_selected_{args.aggregation}.png"
    fig3b = create_layer_pooled_visualization(
        layer_attentions=layer_attentions,
        path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
        word=word,
        pooling_method=args.aggregation,
        save_path=str(selected_pooled_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {selected_pooled_path}")
    plt.close(fig3b)

    # Timeline plot (attention vs time for each character)
    print("   - Timeline plot (attention vs time for each character)...")
    timeline_path = output_dir / f"{word}_timeline_{args.aggregation}.png"
    fig4 = create_attention_timeline_plot(
        layer_attentions=all_layer_attentions,
        path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
        word=word,
        pooling_method=args.aggregation,
        save_path=str(timeline_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {timeline_path}")
    plt.close(fig4)

    # Timeline plot (selected layers only)
    print("   - Selected-layer timeline plot...")
    selected_timeline_path = output_dir / f"{word}_timeline_selected_{args.aggregation}.png"
    fig4b = create_attention_timeline_plot(
        layer_attentions=layer_attentions,
        path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
        word=word,
        pooling_method=args.aggregation,
        save_path=str(selected_timeline_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {selected_timeline_path}")
    plt.close(fig4b)

    # Per-layer timeline plots
    print(f"   - Per-layer timeline plots (all {len(all_layer_attentions)} layers)...")
    per_layer_paths = []
    for layer_idx, layer_attn in all_layer_attentions.items():
        per_layer_path = (
            output_dir / f"{word}_timeline_layer_{layer_idx:02d}_{args.aggregation}.png"
        )
        fig_layer = create_single_layer_timeline_plot(
            layer_attention=layer_attn,
            layer_idx=layer_idx,
            path_coords=inputs["path_coords"][0].detach().cpu().numpy(),
            word=word,
            save_path=str(per_layer_path),
            path_mask=np.array(path_mask),
        )
        per_layer_paths.append(per_layer_path.name)
        plt.close(fig_layer)
    print(f"     Saved {len(per_layer_paths)} layer-specific timeline plots")

    print("\n" + "=" * 70)
    print("✓ Visualization complete!")
    print(f"  Word: '{word}'")
    print(f"  Aggregation: {args.aggregation}")
    if last_k is None:
        print(f"  Layers: {layer_indices}")
    else:
        print(f"  Last-k layers: {last_k}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output directory: {output_dir}")
    print("  Files created:")
    print(f"    - {grid_path.name}")
    print(f"    - {summary_path.name}")
    print(f"    - {pooled_path.name}")
    print(f"    - {selected_pooled_path.name}")
    print(f"    - {timeline_path.name}")
    print(f"    - {selected_timeline_path.name}")
    print(f"    - {len(per_layer_paths)} per-layer timeline plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
