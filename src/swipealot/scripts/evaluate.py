"""Evaluation script for swipe keyboard model."""

import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from swipealot.data import ValidationCollator, SwipeDataset
from swipealot.models import SwipeTransformerModel
from swipealot.training import CharacterAccuracy, WordAccuracy
from swipealot.utils import batch_to_device, extract_character_logits


def evaluate_model(model, test_loader, tokenizer, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        tokenizer: Character tokenizer
        device: Device to run on

    Returns:
        Dictionary of metrics, predictions, targets, and sample data
    """
    model.eval()

    char_accuracy = CharacterAccuracy(vocab_size=tokenizer.vocab_size, device=str(device))
    word_accuracy = WordAccuracy(tokenizer)

    all_predictions = []
    all_targets = []
    all_samples = []  # Store (path_coords, target, prediction) tuples

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            batch = batch_to_device(batch, device)

            # Forward pass
            outputs = model(
                path_coords=batch["path_coords"],
                char_tokens=batch["char_tokens"],
                attention_mask=batch["attention_mask"],
            )

            # Extract character predictions
            char_logits = outputs["char_logits"]
            path_len = batch["path_coords"].shape[1]
            char_logits_subset = extract_character_logits(
                char_logits, path_len, batch["char_tokens"].shape[1]
            )

            # Get predictions
            pred_tokens = char_logits_subset.argmax(dim=-1)

            # Decode predictions
            for i, pred in enumerate(pred_tokens):
                pred_word = tokenizer.decode(pred.cpu().tolist())
                target_word = batch["words"][i]
                all_predictions.append(pred_word.strip())
                all_targets.append(target_word.strip())

                # Store sample data for visualization
                path_coords = batch["path_coords"][i].cpu().numpy()
                all_samples.append((path_coords, target_word.strip(), pred_word.strip()))

            # Update metrics (only on non-padding positions)
            # Use the masking-aware labels so padding is ignored
            char_labels = batch["char_labels"]
            char_accuracy.update(char_logits_subset, char_labels)
            word_accuracy.update(char_logits_subset, batch["words"])

    # Compute final metrics
    char_acc = char_accuracy.compute()
    word_acc = word_accuracy.compute()

    results = {
        "char_accuracy": char_acc,
        "word_accuracy": word_acc,
        "num_samples": len(all_predictions),
    }

    return results, all_predictions, all_targets, all_samples


def visualize_samples(samples, num_samples=20, output_file="eval_samples.png"):
    """
    Visualize swipe paths with their predictions.

    Args:
        samples: List of (path_coords, target, prediction) tuples
        num_samples: Number of samples to visualize
        output_file: Output file path
    """
    num_samples = min(num_samples, len(samples))
    cols = 5
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Swipe Paths and Predictions", fontsize=16, fontweight="bold")

    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        path_coords, target, prediction = samples[idx]

        # Extract x, y coordinates (ignoring padding)
        # Path coords shape: (max_path_len, 3) where last dim is (x, y, t)
        x_coords = path_coords[:, 0]
        y_coords = path_coords[:, 1]

        # Remove padding (coordinates that are all zeros)
        mask = (x_coords != 0) | (y_coords != 0)
        if mask.sum() > 0:
            x_coords = x_coords[mask]
            y_coords = y_coords[mask]

        # Plot the path
        if len(x_coords) > 0:
            ax.plot(x_coords, y_coords, "o-", linewidth=2, markersize=3, color="blue", alpha=0.7)
            ax.plot(x_coords[0], y_coords[0], "go", markersize=8, label="Start", zorder=10)
            ax.plot(x_coords[-1], y_coords[-1], "ro", markersize=8, label="End", zorder=10)

        # Formatting
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis to match screen coordinates

        # Title with target and prediction
        correct = "✓" if prediction == target else "✗"
        color = "green" if prediction == target else "red"
        ax.set_title(
            f'{correct} Target: "{target}"\nPred: "{prediction}"', fontsize=9, color=color, pad=10
        )

        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)

    # Hide unused subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate swipe keyboard model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (train/validation/test)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument("--show_examples", action="store_true", help="Show example predictions")
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualization of swipe paths and predictions"
    )
    parser.add_argument(
        "--viz_samples", type=int, default=20, help="Number of samples to visualize"
    )
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Handle both old (TrainingConfig only) and new (full Config) checkpoint formats
    from swipealot.config import Config, DataConfig, ModelConfig

    if not hasattr(config, "data"):
        # Old checkpoint format - only has TrainingConfig
        print("Warning: Old checkpoint format detected (missing data/model config)")
        print("Using default values for data and model config...")

        # Extract what we can from the model state dict
        model_state = checkpoint["model_state_dict"]

        # Get vocab size from embedding layer
        vocab_size = model_state["embeddings.char_embedding.embedding.weight"].shape[0]

        # Infer model architecture from state dict
        # d_model from path embedding projection output size
        d_model = model_state["embeddings.path_embedding.projection.weight"].shape[0]

        # n_layers by counting encoder layers
        n_layers = max(
            int(key.split(".")[2])
            for key in model_state.keys()
            if key.startswith("encoder.layers.")
        ) + 1

        # d_ff from first linear layer
        d_ff = model_state["encoder.layers.0.linear1.weight"].shape[0]

        # n_heads: can't infer directly from state dict, use heuristic based on d_model
        # Common configs: d_model=256->4 heads, d_model=512->8 heads, d_model=768->12 heads
        if d_model >= 768:
            n_heads = 12
        elif d_model >= 512:
            n_heads = 8
        elif d_model >= 256:
            n_heads = 4
        else:
            n_heads = 4

        # max sequence length from positional embedding
        max_seq_len = model_state["embeddings.positional_embedding.embedding.weight"].shape[0]
        # max_seq_len = max_path_len + max_char_len + 2 (CLS, SEP)
        # Common configs: 64+38+2=104, 128+48+2=178
        if max_seq_len >= 178:
            max_path_len = 128
            max_char_len = 48
        else:
            max_path_len = 64
            max_char_len = 38

        print(
            f"Inferred model architecture: d_model={d_model}, n_layers={n_layers}, "
            f"n_heads={n_heads}, d_ff={d_ff}, max_path_len={max_path_len}, "
            f"max_char_len={max_char_len}"
        )

        # Create configs with inferred architecture
        model_config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_path_len=max_path_len,
            max_char_len=max_char_len,
        )
        data_config = DataConfig(max_path_len=max_path_len, max_char_len=max_char_len)

        # Create full config with defaults
        full_config = Config(model=model_config, data=data_config, training=config)
        config = full_config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build tokenizer
    print("\nBuilding tokenizer...")
    sample_dataset = SwipeDataset(
        split="train",
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        dataset_name=config.data.dataset_name,
        max_samples=10000,
    )
    tokenizer = sample_dataset.tokenizer

    # Create test dataset
    print(f"\nLoading {args.split} dataset...")
    test_dataset = SwipeDataset(
        split=args.split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=args.num_samples,
    )

    # For evaluation, use ValidationCollator which provides deterministic results
    # (no random masking, evaluates full reconstruction from unmasked input)
    collator = ValidationCollator(tokenizer=tokenizer)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collator
    )

    # Create model
    print("\nCreating model...")
    model = SwipeTransformerModel(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Evaluate
    print("\nEvaluating...")
    results, predictions, targets, samples = evaluate_model(model, test_loader, tokenizer, device)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Character Accuracy: {results['char_accuracy']:.4f}")
    print(f"Word Accuracy: {results['word_accuracy']:.4f}")
    print("=" * 60)

    # Show examples (always show them)
    print("\nSample Predictions (case-insensitive comparison):")
    print("-" * 60)
    num_to_show = min(args.viz_samples if args.visualize else 20, len(predictions))
    for i in range(num_to_show):
        # Case-insensitive comparison since model is uncased
        correct = "✓" if predictions[i].lower() == targets[i].lower() else "✗"
        print(f"{correct} Target: '{targets[i]}' | Predicted: '{predictions[i]}'")

    # Visualize if requested
    if args.visualize:
        print("\nCreating visualization...")
        visualize_samples(samples, num_samples=args.viz_samples)


if __name__ == "__main__":
    main()
