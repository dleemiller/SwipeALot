#!/usr/bin/env python3
"""CLI tool for generating attention visualizations on swipe paths.

Usage:
    uv run attention-map --checkpoint checkpoints/base_20251213_164813/best.pt --word-index 10
    uv run attention-map --checkpoint checkpoints/base_20251213_164813/best.pt --word "hello"
    uv run attention-map --checkpoint checkpoints/base_20251213_164813/best.pt --word-index 10 --layers 0 6 11
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset

from swipealot.analysis import (
    AttentionHookManager,
    create_attention_timeline_plot,
    create_layer_comparison_grid,
    create_layer_pooled_visualization,
    create_single_layer_timeline_plot,
    create_summary_visualization,
    extract_path_to_char_attention,
    extract_special_token_to_path_attention,
)
from swipealot.data import SwipeDataset
from swipealot.data.preprocessing import (
    normalize_and_compute_features,
    sample_path_points_with_features,
)
from swipealot.models import SwipeTransformerModel


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention visualizations for swipe keyboard model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize word at index 10 from validation set
  uv run attention-map --checkpoint checkpoints/base_20251213_164813/best.pt --word-index 10

  # Visualize specific word
  uv run attention-map --checkpoint checkpoints/base_20251213_164813/best.pt --word "hello"

  # Specify custom layers and output directory
  uv run attention-map --checkpoint checkpoints/base_20251213_164813/best.pt --word-index 5 --layers 0 6 11 --output visualizations/attention/custom
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
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

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 6, 11],
        help="Transformer layers to visualize (default: 0 6 11)",
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
        "--num-samples",
        type=int,
        default=100000,
        help="Number of samples to load from dataset when searching by word (default: 1000)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Attention Map Visualization")
    print("=" * 70)

    # 1. Load checkpoint and config
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Model: {config.model.n_layers} layers, {config.model.n_heads} heads")

    # 2. Build tokenizer
    print("\n2. Building tokenizer...")
    sample_dataset = SwipeDataset(
        split="train",
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        dataset_name=config.data.dataset_name,
        max_samples=10000,
    )
    tokenizer = sample_dataset.tokenizer
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # 3. Load model
    print("\n3. Loading model...")
    model = SwipeTransformerModel(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("   Model loaded successfully")

    # 4. Load sample from dataset
    print(f"\n4. Loading sample from {args.dataset_split} set...")

    if args.word_index is not None:
        # Load specific index
        dataset = load_dataset(
            config.data.dataset_name,
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
            config.data.dataset_name, split=f"{args.dataset_split}[:{args.num_samples}]"
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

    processed_path = normalize_and_compute_features(path_data)
    path_coords, path_mask = sample_path_points_with_features(
        processed_path, config.data.max_path_len
    )

    # Tokenize word
    char_tokens = tokenizer.encode(word) + [tokenizer.eos_token_id]
    if len(char_tokens) < config.data.max_char_len:
        char_tokens = char_tokens + [tokenizer.pad_token_id] * (
            config.data.max_char_len - len(char_tokens)
        )
    else:
        char_tokens = char_tokens[: config.data.max_char_len - 1] + [tokenizer.eos_token_id]

    char_mask = [1 if token != tokenizer.pad_token_id else 0 for token in char_tokens]

    # Create attention mask
    cls_mask = torch.ones(1, 1, dtype=torch.long)
    sep_mask = torch.ones(1, 1, dtype=torch.long)
    attention_mask = torch.cat(
        [
            cls_mask,
            torch.tensor(path_mask, dtype=torch.long).unsqueeze(0),
            sep_mask,
            torch.tensor(char_mask, dtype=torch.long).unsqueeze(0),
        ],
        dim=1,
    )

    # Convert to tensors
    path_coords_tensor = torch.tensor(path_coords, dtype=torch.float32).unsqueeze(0)
    char_tokens_tensor = torch.tensor(char_tokens, dtype=torch.long).unsqueeze(0)

    # 6. Extract attention from ALL layers for layer-pooled visualization
    print("\n6. Extracting attention from all 12 layers...")
    all_layers = list(range(config.model.n_layers))
    hook_manager_all = AttentionHookManager(model, target_layers=all_layers)

    attention_weights_all = hook_manager_all.extract_attention(
        path_coords=path_coords_tensor,
        char_tokens=char_tokens_tensor,
        attention_mask=attention_mask,
    )

    # Extract char→path attention with specified aggregation for all layers
    all_layer_attentions = {}
    for layer_idx, attn in attention_weights_all.items():
        char_to_path = extract_path_to_char_attention(attn, aggregation=args.aggregation)
        all_layer_attentions[layer_idx] = char_to_path[0].cpu().numpy()  # Remove batch dim

    print(f"   Extracted attention from all {len(all_layer_attentions)} layers")

    # Extract special token→path attention for all layers
    special_token_attentions = {"cls": {}, "sep": {}, "eos": {}}
    for layer_idx, attn in attention_weights_all.items():
        special_tokens = extract_special_token_to_path_attention(
            attn, word_length=len(word), aggregation=args.aggregation
        )
        for token_name, token_attn in special_tokens.items():
            special_token_attentions[token_name][layer_idx] = (
                token_attn[0].cpu().numpy()
            )  # Remove batch dim

    print("   Extracted special token attention (CLS, SEP, EOS) from all layers")

    # Also keep attention for specified layers for comparison visualization
    layer_attentions = {k: v for k, v in all_layer_attentions.items() if k in args.layers}
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
        path_coords=path_coords,
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
        path_coords=path_coords,
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
        path_coords=path_coords,
        word=word,
        pooling_method=args.aggregation,
        save_path=str(pooled_path),
        path_mask=np.array(path_mask),
    )
    print(f"     Saved to: {pooled_path}")
    plt.close(fig3)

    # Timeline plot (attention vs time for each character)
    print("   - Timeline plot (attention vs time for each character)...")
    timeline_path = output_dir / f"{word}_timeline_{args.aggregation}.png"
    fig4 = create_attention_timeline_plot(
        layer_attentions=all_layer_attentions,
        path_coords=path_coords,
        word=word,
        pooling_method=args.aggregation,
        save_path=str(timeline_path),
        path_mask=np.array(path_mask),
        special_token_attentions=special_token_attentions,
    )
    print(f"     Saved to: {timeline_path}")
    plt.close(fig4)

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
            path_coords=path_coords,
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
    print(f"  Output directory: {output_dir}")
    print("  Files created:")
    print(f"    - {grid_path.name}")
    print(f"    - {summary_path.name}")
    print(f"    - {pooled_path.name}")
    print(f"    - {timeline_path.name}")
    print(f"    - {len(per_layer_paths)} per-layer timeline plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
