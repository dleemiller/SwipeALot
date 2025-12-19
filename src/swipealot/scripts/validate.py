"""Validate a HuggingFace SwipeALot checkpoint via the customer-facing APIs.

This script loads checkpoints using `AutoModel` / `AutoProcessor` with `trust_remote_code=True`
so the results reflect what will run when the checkpoint is exported / uploaded.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from swipealot.text_utils import swipable_length


def load_model_and_processor(model_path: Path):
    """Load HuggingFace model and processor."""
    from transformers import AutoModel, AutoProcessor

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"  ✓ Model loaded on {device}")
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, processor, device


def test_length_prediction(model, processor, device, dataset_name: str, n_samples: int = 1000):
    """
    Test the model's ability to predict word length from path.

    The current architecture predicts length as a scalar regression output (`length_logits: [batch]`).
    """
    print(f"\n{'=' * 60}")
    print("Testing Length Prediction")
    print(f"{'=' * 60}\n")

    # Load test data
    print(f"Loading dataset: {dataset_name}")
    test_data = load_dataset(dataset_name, split=f"test[:{n_samples}]")
    print(f"  ✓ Loaded {len(test_data)} samples")

    # Prepare samples
    predicted_lengths = []
    true_lengths = []

    print("\nPredicting word lengths...")
    with torch.no_grad():
        for item in tqdm(test_data, desc="Testing"):
            word = item["word"]
            # Count only alphanumeric characters (matching training)
            true_length = swipable_length(word)

            # Process input (path only, no text)
            inputs = processor(path_coords=item["data"], text=None, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model output
            outputs = model(**inputs)

            # Check if model has length prediction
            if not hasattr(outputs, "length_logits"):
                print("  ✗ Model does not have length prediction capability")
                return None

            pred_len = float(outputs.length_logits.reshape(-1)[0].item())
            predicted_lengths.append(pred_len)
            true_lengths.append(float(true_length))

    # Compute metrics
    predicted_lengths = np.array(predicted_lengths, dtype=np.float64)
    true_lengths = np.array(true_lengths, dtype=np.float64)

    err = predicted_lengths - true_lengths
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    pred_round = np.round(predicted_lengths).clip(min=0.0)
    exact_accuracy = float(accuracy_score(true_lengths.astype(int), pred_round.astype(int)))
    within_1 = float(np.mean(np.abs(pred_round - true_lengths) <= 1))
    within_2 = float(np.mean(np.abs(pred_round - true_lengths) <= 2))

    print("\n" + "=" * 60)
    print("Length Prediction Results")
    print("=" * 60)
    print(f"Samples tested: {len(test_data)}")
    print(f"\nExact Accuracy: {exact_accuracy:.4f}")
    print(f"Within ±1: {within_1:.4f}")
    print(f"Within ±2: {within_2:.4f}")
    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot confusion matrix (rounded predictions; for lengths up to 15)
    max_len = int(min(15, max(max(true_lengths), max(pred_round))))
    mask = (true_lengths <= max_len) & (pred_round <= max_len)
    cm = confusion_matrix(
        true_lengths[mask].astype(int),
        pred_round[mask].astype(int),
        labels=list(range(max_len + 1)),
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(max_len + 1),
        yticklabels=range(max_len + 1),
    )
    plt.xlabel("Predicted Length")
    plt.ylabel("True Length")
    plt.title("Length Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig("length_prediction_confusion_matrix.png", dpi=150)
    print("\n  ✓ Saved confusion matrix to length_prediction_confusion_matrix.png")

    # Plot error distribution
    errors = predicted_lengths - true_lengths
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title("Length Prediction Error Distribution")
    plt.axvline(0, color="red", linestyle="--", label="Perfect prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("length_prediction_errors.png", dpi=150)
    print("  ✓ Saved error distribution to length_prediction_errors.png")

    return {
        "exact_accuracy": exact_accuracy,
        "within_1": within_1,
        "within_2": within_2,
        "mae": mae,
        "rmse": rmse,
    }


def _is_alphanumeric_match(true_char: str, pred_char: str) -> bool:
    """
    Check if two characters match for evaluation purposes.
    Case-insensitive for letters, exact match for numbers.
    Returns False for punctuation and special tokens.
    """
    # Only evaluate a-z and 0-9
    if not (true_char.isalnum() and pred_char.isalnum()):
        return False
    # Case-insensitive comparison
    return true_char.lower() == pred_char.lower()


def _should_evaluate_char(char: str) -> bool:
    """Check if character should be included in accuracy evaluation."""
    # Only evaluate alphanumeric characters
    return char.isalnum()


def test_masked_prediction(model, processor, device, dataset_name: str, n_samples: int = 1000):
    """
    Test base model's ability to predict masked characters.

    Only evaluates alphanumeric characters (a-z, 0-9) with case-insensitive matching.
    Ignores punctuation, special tokens, and EOS.
    """
    print(f"\n{'=' * 60}")
    print("Testing Masked Character Prediction")
    print(f"{'=' * 60}\n")

    # Load test data
    print(f"Loading dataset: {dataset_name}")
    test_data = load_dataset(dataset_name, split=f"test[:{n_samples}]")
    print(f"  ✓ Loaded {len(test_data)} samples")

    # Get tokenizer
    tokenizer = processor.tokenizer
    mask_token_id = tokenizer.mask_token_id

    # Prepare samples with masking
    all_predictions = []
    all_prediction_probs = []  # For top-k accuracy
    all_labels = []
    all_accuracies = []

    print("\nTesting masked character prediction...")
    with torch.no_grad():
        for item in tqdm(test_data, desc="Testing"):
            word = item["word"]

            # Skip very short words
            if len(word) < 2:
                continue

            # Process the full word first to get the proper sequence
            inputs_original = processor(path_coords=item["data"], text=word, return_tensors="pt")

            # Get the character tokens
            char_ids = inputs_original["input_ids"][0].tolist()

            # Randomly mask 30% of characters
            mask_prob = 0.3
            masked_positions = []
            original_ids = []
            masked_char_ids = char_ids.copy()

            for i, char_id in enumerate(char_ids):
                # Skip padding tokens
                if char_id == 0:  # PAD token
                    break
                if np.random.random() < mask_prob:
                    masked_positions.append(i)
                    original_ids.append(char_id)
                    masked_char_ids[i] = mask_token_id

            # Skip if no characters were masked
            if len(masked_positions) == 0:
                continue

            # Create inputs with masked tokens
            inputs = {
                "path_coords": inputs_original["path_coords"].to(device),
                "input_ids": torch.tensor([masked_char_ids], dtype=torch.long).to(device),
                "attention_mask": inputs_original["attention_mask"].to(device),
            }

            # Get predictions
            outputs = model(**inputs)

            if not hasattr(outputs, "char_logits"):
                print("  ✗ Model does not have character prediction capability")
                return None

            char_logits = outputs.char_logits
            # New behavior: [batch, char_len, vocab_size]. Legacy: [batch, seq_len, vocab_size].
            is_char_segment_only = char_logits.shape[1] == inputs_original["input_ids"].shape[1]

            # Extract predictions for masked positions
            # Only evaluate alphanumeric characters
            word_evaluable_predictions = []
            for pos, true_id in zip(masked_positions, original_ids, strict=True):
                # Get true and predicted characters
                true_char = tokenizer._tokenizer.id_to_char.get(true_id, "?")

                # Skip non-alphanumeric characters (punctuation, EOS, etc.)
                if not _should_evaluate_char(true_char):
                    continue

                logits = (
                    char_logits[0, pos]
                    if is_char_segment_only
                    else char_logits[0, (1 + inputs["path_coords"].shape[1] + 1 + pos)]
                )
                pred_id = logits.argmax().item()
                pred_char = tokenizer._tokenizer.id_to_char.get(pred_id, "?")

                # Store for character-level accuracy
                all_predictions.append(pred_id)
                all_prediction_probs.append(logits.cpu().numpy())
                all_labels.append(true_id)

                # Store for word-level accuracy (case-insensitive)
                word_evaluable_predictions.append(_is_alphanumeric_match(true_char, pred_char))

            # Compute word-level accuracy (all alphanumeric masked chars correct)
            if len(word_evaluable_predictions) > 0:
                word_correct = all(word_evaluable_predictions)
                all_accuracies.append(word_correct)

    # Compute metrics with case-insensitive matching
    # Convert IDs to characters and compare case-insensitively
    char_correct = 0
    for true_id, pred_id in zip(all_labels, all_predictions, strict=True):
        true_char = tokenizer._tokenizer.id_to_char.get(true_id, "?")
        pred_char = tokenizer._tokenizer.id_to_char.get(pred_id, "?")
        if _is_alphanumeric_match(true_char, pred_char):
            char_correct += 1

    char_accuracy = char_correct / len(all_labels) if len(all_labels) > 0 else 0
    word_accuracy = np.mean(all_accuracies) if len(all_accuracies) > 0 else 0

    # Top-k accuracy (using probabilities, not predictions)
    # For top-k, we need to check if any of the top-k predictions match (case-insensitive)
    all_prediction_probs = np.array(all_prediction_probs)  # [n_samples, vocab_size]

    top3_correct = 0
    top5_correct = 0
    for _i, (true_id, probs) in enumerate(zip(all_labels, all_prediction_probs, strict=True)):
        true_char = tokenizer._tokenizer.id_to_char.get(true_id, "?")
        top5_ids = np.argsort(probs)[-5:][::-1]  # Top 5 predictions
        top3_ids = top5_ids[:3]  # Top 3 predictions

        # Check if any top-k prediction matches (case-insensitive)
        top3_chars = [tokenizer._tokenizer.id_to_char.get(int(pid), "?") for pid in top3_ids]
        top5_chars = [tokenizer._tokenizer.id_to_char.get(int(pid), "?") for pid in top5_ids]

        if any(_is_alphanumeric_match(true_char, pred_char) for pred_char in top3_chars):
            top3_correct += 1
        if any(_is_alphanumeric_match(true_char, pred_char) for pred_char in top5_chars):
            top5_correct += 1

    top3_accuracy = top3_correct / len(all_labels) if len(all_labels) > 0 else 0
    top5_accuracy = top5_correct / len(all_labels) if len(all_labels) > 0 else 0

    print("\n" + "=" * 60)
    print("Masked Character Prediction Results")
    print("=" * 60)
    print(f"Samples tested: {len(all_accuracies)}")
    print(f"Characters evaluated: {len(all_labels)} (alphanumeric only, case-insensitive)")
    print(f"\nCharacter Accuracy: {char_accuracy:.4f}")
    print(f"Character Top-3 Accuracy: {top3_accuracy:.4f}")
    print(f"Character Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"\nWord Accuracy (all chars correct): {word_accuracy:.4f}")

    return {
        "char_accuracy": char_accuracy,
        "char_top3_accuracy": top3_accuracy,
        "char_top5_accuracy": top5_accuracy,
        "word_accuracy": word_accuracy,
        "n_chars_tested": len(all_labels),
        "n_words_tested": len(all_accuracies),
    }


def test_full_reconstruction(model, processor, device, dataset_name: str, n_samples: int = 1000):
    """
    Test full character reconstruction matching training validation setup.

    Masks ALL characters (100%) and evaluates model's ability to reconstruct
    the entire word from just the path coordinates and positional information.
    This matches the ValidationCollator used during training.

    Only evaluates alphanumeric characters (a-z, 0-9) with case-insensitive matching.
    """
    print(f"\n{'=' * 60}")
    print("Testing Full Reconstruction (100% masking)")
    print(f"{'=' * 60}\n")

    # Load test data
    print(f"Loading dataset: {dataset_name}")
    test_data = load_dataset(dataset_name, split=f"test[:{n_samples}]")
    print(f"  ✓ Loaded {len(test_data)} samples")

    # Get tokenizer
    tokenizer = processor.tokenizer
    mask_token_id = tokenizer.mask_token_id

    # Tracking metrics
    all_predictions = []
    all_labels = []
    all_word_correct = []

    print("\nTesting full reconstruction (all characters masked)...")
    with torch.no_grad():
        for item in tqdm(test_data, desc="Testing"):
            word = item["word"]

            # Skip very short words
            if len(word) < 2:
                continue

            # Process to get proper sequence
            inputs_original = processor(path_coords=item["data"], text=word, return_tensors="pt")
            char_ids = inputs_original["input_ids"][0].tolist()

            # Mask ALL character tokens (matching ValidationCollator)
            masked_char_ids = []
            evaluable_positions = []
            true_ids = []

            for i, char_id in enumerate(char_ids):
                # Skip padding tokens
                if char_id == 0:  # PAD token
                    masked_char_ids.append(char_id)
                    break

                # Get character
                true_char = tokenizer._tokenizer.id_to_char.get(char_id, "?")

                # Mask all non-padding tokens
                masked_char_ids.append(mask_token_id)

                # Only evaluate alphanumeric
                if _should_evaluate_char(true_char):
                    evaluable_positions.append(i)
                    true_ids.append(char_id)

            # Skip if no evaluable characters
            if len(evaluable_positions) == 0:
                continue

            # Pad masked_char_ids to match original length
            while len(masked_char_ids) < len(char_ids):
                masked_char_ids.append(0)

            # Create inputs with all characters masked
            inputs = {
                "path_coords": inputs_original["path_coords"].to(device),
                "input_ids": torch.tensor([masked_char_ids], dtype=torch.long).to(device),
                "attention_mask": inputs_original["attention_mask"].to(device),
            }

            # Get predictions
            outputs = model(**inputs)

            if not hasattr(outputs, "char_logits"):
                print("  ✗ Model does not have character prediction capability")
                return None

            char_logits = outputs.char_logits
            is_char_segment_only = char_logits.shape[1] == inputs_original["input_ids"].shape[1]

            # Evaluate all alphanumeric positions
            word_predictions = []
            for pos, true_id in zip(evaluable_positions, true_ids, strict=True):
                logits = (
                    char_logits[0, pos]
                    if is_char_segment_only
                    else char_logits[0, (1 + inputs["path_coords"].shape[1] + 1 + pos)]
                )
                pred_id = logits.argmax().item()

                true_char = tokenizer._tokenizer.id_to_char.get(true_id, "?")
                pred_char = tokenizer._tokenizer.id_to_char.get(pred_id, "?")

                # Store for metrics
                all_predictions.append(pred_id)
                all_labels.append(true_id)

                # Check if correct (case-insensitive)
                is_correct = _is_alphanumeric_match(true_char, pred_char)
                word_predictions.append(is_correct)

            # Word is correct if all alphanumeric characters are correct
            if len(word_predictions) > 0:
                all_word_correct.append(all(word_predictions))

    # Compute metrics with case-insensitive matching
    char_correct = 0
    for true_id, pred_id in zip(all_labels, all_predictions, strict=True):
        true_char = tokenizer._tokenizer.id_to_char.get(true_id, "?")
        pred_char = tokenizer._tokenizer.id_to_char.get(pred_id, "?")
        if _is_alphanumeric_match(true_char, pred_char):
            char_correct += 1

    char_accuracy = char_correct / len(all_labels) if len(all_labels) > 0 else 0
    word_accuracy = np.mean(all_word_correct) if len(all_word_correct) > 0 else 0

    print("\n" + "=" * 60)
    print("Full Reconstruction Results (ValidationCollator setup)")
    print("=" * 60)
    print(f"Samples tested: {len(all_word_correct)}")
    print(f"Characters evaluated: {len(all_labels)} (alphanumeric only, case-insensitive)")
    print(f"\nCharacter Accuracy: {char_accuracy:.4f}")
    print(f"Word Accuracy (all chars correct): {word_accuracy:.4f}")
    print("\nThis matches the training validation setup:")
    print("  - 100% of characters masked")
    print("  - Model predicts from path + position only")
    print("  - Training validation achieved ~95% char accuracy")

    return {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "n_chars_tested": len(all_labels),
        "n_words_tested": len(all_word_correct),
    }


def test_path_reconstruction(model, processor, device, dataset_name: str, n_samples: int = 1000):
    """
    Test path reconstruction from masked path coordinates.

    Masks 30% of path points and measures MSE between predicted and true coordinates.

    Args:
        model: The model to test
        processor: The processor
        device: Device to run on
        dataset_name: Name of the HuggingFace dataset
        n_samples: Number of samples to test

    Returns:
        Dictionary with MSE metrics
    """
    print(f"\n{'=' * 60}")
    print("Testing Path Reconstruction")
    print(f"{'=' * 60}\n")

    # Check if model has path prediction capability
    if not hasattr(model, "path_head") or model.path_head is None:
        print("  ✗ Model does not have path prediction capability")
        return None

    # Load test data
    print(f"Loading dataset: {dataset_name}")
    test_data = load_dataset(dataset_name, split=f"test[:{n_samples}]")
    print(f"  ✓ Loaded {len(test_data)} samples")

    # Get max_path_len from processor
    max_path_len = processor.max_path_len

    # Tracking metrics
    all_mse = []
    all_masked_mse = []

    print("\nTesting path reconstruction (30% masking)...")
    mask_prob = 0.3

    with torch.no_grad():
        for item in tqdm(test_data, desc="Testing"):
            word = item["word"]
            base_inputs = processor(path_coords=item["data"], text=word, return_tensors="pt")
            path = base_inputs["path_coords"]  # [1, max_path_len, path_input_dim]
            full_attention_mask = base_inputs["attention_mask"][0].cpu().numpy()
            path_mask = full_attention_mask[1 : 1 + max_path_len].astype(int).tolist()

            # Create masked version of path
            masked_path = path.clone()
            mask_indices = []

            # Randomly mask path points
            for i in range(max_path_len):
                if path_mask[i] == 1 and np.random.random() < mask_prob:
                    masked_path[0, i] = 0.0  # Zero out masked coordinates
                    mask_indices.append(i)

            # Skip if no points were masked
            if len(mask_indices) == 0:
                continue

            # Process inputs with masked path
            inputs = base_inputs
            inputs["path_coords"] = masked_path
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get predictions
            outputs = model(**inputs)

            if not hasattr(outputs, "path_logits") or outputs.path_logits is None:
                print("  ✗ Model output does not have path_logits")
                return None

            path_logits = outputs.path_logits
            # New behavior: [batch, path_len, D]. Legacy: [batch, seq_len, D].
            if path_logits.shape[1] == max_path_len:
                path_predictions = path_logits[0]
            else:
                path_start = 1
                path_predictions = path_logits[0, path_start : path_start + max_path_len, :]

            # Calculate MSE for masked points only
            true_coords = path[0].cpu().numpy()
            pred_coords = path_predictions.cpu().numpy()

            # MSE over all valid points
            valid_indices = [i for i in range(max_path_len) if path_mask[i] == 1]
            if len(valid_indices) > 0:
                mse = np.mean((true_coords[valid_indices] - pred_coords[valid_indices]) ** 2)
                all_mse.append(mse)

            # MSE over masked points only
            if len(mask_indices) > 0:
                masked_mse = np.mean((true_coords[mask_indices] - pred_coords[mask_indices]) ** 2)
                all_masked_mse.append(masked_mse)

    # Compute metrics
    avg_mse = np.mean(all_mse) if len(all_mse) > 0 else 0
    avg_masked_mse = np.mean(all_masked_mse) if len(all_masked_mse) > 0 else 0
    rmse = np.sqrt(avg_mse)
    masked_rmse = np.sqrt(avg_masked_mse)

    print("\n" + "=" * 60)
    print("Path Reconstruction Results")
    print("=" * 60)
    print(f"Samples tested: {len(all_mse)}")
    print(f"Average points masked per sample: {len(all_masked_mse) / len(all_mse):.1f}")
    print(f"\nAll points MSE: {avg_mse:.6f}")
    print(f"All points RMSE: {rmse:.6f}")
    print(f"\nMasked points MSE: {avg_masked_mse:.6f}")
    print(f"Masked points RMSE: {masked_rmse:.6f}")

    return {
        "mse": avg_mse,
        "rmse": rmse,
        "masked_mse": avg_masked_mse,
        "masked_rmse": masked_rmse,
        "n_samples": len(all_mse),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate SwipeALot HuggingFace checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all applicable checks
  uv run validate --model checkpoints/.../final --n-samples 1000

  # Run just length evaluation
  uv run validate --model checkpoints/.../final --test length --n-samples 500
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "length", "masked", "reconstruction", "path"],
        default="all",
        help="Which test to run (default: all applicable tests)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of test samples (default: 1000)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="futo-org/swipe.futo.org",
        help="HuggingFace dataset to use for testing",
    )

    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)

    model, processor, device = load_model_and_processor(model_path)

    # Run tests
    results = {}

    if args.test in ["all", "length"] and bool(getattr(model.config, "predict_length", False)):
        length_results = test_length_prediction(
            model, processor, device, args.dataset, args.n_samples
        )
        if length_results:
            results["length"] = length_results

    if args.test in ["all", "masked"] and bool(getattr(model.config, "predict_char", False)):
        masked_results = test_masked_prediction(
            model, processor, device, args.dataset, args.n_samples
        )
        if masked_results:
            results["masked"] = masked_results

    if args.test in ["all", "reconstruction"] and bool(
        getattr(model.config, "predict_char", False)
    ):
        reconstruction_results = test_full_reconstruction(
            model, processor, device, args.dataset, args.n_samples
        )
        if reconstruction_results:
            results["reconstruction"] = reconstruction_results

    if args.test in ["all", "path"] and bool(getattr(model.config, "predict_path", False)):
        path_results = test_path_reconstruction(
            model, processor, device, args.dataset, args.n_samples
        )
        if path_results:
            results["path"] = path_results

    # Save results summary
    if results:
        results_df = pd.DataFrame([results]).T
        results_df.to_csv("test_results.csv")
        print("\n  ✓ Saved results summary to test_results.csv")

        print("\n" + "=" * 60)
        print("All Tests Complete")
        print("=" * 60)


if __name__ == "__main__":
    main()
