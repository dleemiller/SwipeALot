"""
Convert SwipeAlot training checkpoints to HuggingFace format.

Usage:
    uv run convert-hf --checkpoint checkpoints/best.pt --output ./hf_model
    uv run convert-hf --checkpoint checkpoints/best.pt --output ./hf_model --model-type embedding
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

from swipealot.config import Config
from swipealot.data.tokenizer import CharacterTokenizer
from swipealot.huggingface import (
    SwipeModel,
    SwipeProcessor,
    SwipeTokenizer,
    SwipeTransformerConfig,
    SwipeTransformerModel,
)


def _make_model_standalone(output_path: Path, model_type: str = "embedding"):
    """
    Make the HuggingFace model completely standalone by copying dependencies
    and fixing imports. After this, users can load the model without the repository.

    Args:
        output_path: Directory where model is saved
        model_type: Type of model ('base' or 'embedding') to set correct auto_map
    """
    # Get source directories
    root = Path(__file__).resolve().parents[3]
    hf_code_dir = root / "src" / "swipealot" / "huggingface"
    models_dir = root / "src" / "swipealot" / "models"
    data_dir = root / "src" / "swipealot" / "data"

    print("\nMaking model standalone...")

    # Copy and fix modeling_swipe.py
    modeling_src = hf_code_dir / "modeling_swipe.py"
    modeling_dst = output_path / "modeling_swipe.py"
    with open(modeling_src) as f:
        content = f.read()
    # Fix imports for HF's trust_remote_code system (expects relative imports)
    content = content.replace("from ..models.embeddings import", "from .embeddings import")
    content = content.replace("from ..models.heads import", "from .heads import")
    with open(modeling_dst, "w") as f:
        f.write(content)
    print("  ✓ Fixed modeling_swipe.py imports")

    # Copy configuration_swipe.py as-is
    shutil.copy2(hf_code_dir / "configuration_swipe.py", output_path / "configuration_swipe.py")
    print("  ✓ Copied configuration_swipe.py")

    # Copy and fix tokenization_swipe.py
    tokenization_src = hf_code_dir / "tokenization_swipe.py"
    tokenization_dst = output_path / "tokenization_swipe.py"
    with open(tokenization_src) as f:
        content = f.read()
    # Fix imports for HF's dynamic module loader
    content = content.replace(
        "from ..data.tokenizer import CharacterTokenizer",
        "from .tokenizer import CharacterTokenizer",
    )
    with open(tokenization_dst, "w") as f:
        f.write(content)
    print("  ✓ Fixed tokenization_swipe.py imports")

    # Copy processing_swipe.py as-is
    shutil.copy2(hf_code_dir / "processing_swipe.py", output_path / "processing_swipe.py")
    print("  ✓ Copied processing_swipe.py")

    # Copy dependencies
    shutil.copy2(models_dir / "embeddings.py", output_path / "embeddings.py")
    shutil.copy2(models_dir / "heads.py", output_path / "heads.py")
    shutil.copy2(data_dir / "tokenizer.py", output_path / "tokenizer.py")
    print("  ✓ Copied dependencies (embeddings.py, heads.py, tokenizer.py)")

    # Update config.json to add auto_map with correct model class
    config_path = output_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Set AutoModel class based on model type
    auto_model_class = (
        "modeling_swipe.SwipeTransformerModel"
        if model_type == "base"
        else "modeling_swipe.SwipeModel"
    )

    config_data["auto_map"] = {
        "AutoConfig": "configuration_swipe.SwipeTransformerConfig",
        "AutoModel": auto_model_class,
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  ✓ Added auto_map to config.json (AutoModel: {auto_model_class})")

    # Update tokenizer_config.json to add auto_map
    tokenizer_config_path = output_path / "tokenizer_config.json"
    with open(tokenizer_config_path) as f:
        tokenizer_config_data = json.load(f)

    tokenizer_config_data["auto_map"] = {
        "AutoTokenizer": ["tokenization_swipe.SwipeTokenizer", None]
    }

    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config_data, f, indent=2)
    print("  ✓ Added auto_map to tokenizer_config.json")

    # Update processor_config.json to add auto_map
    processor_config_path = output_path / "processor_config.json"
    if processor_config_path.exists():
        with open(processor_config_path) as f:
            processor_config_data = json.load(f)

        processor_config_data["auto_map"] = {"AutoProcessor": "processing_swipe.SwipeProcessor"}

        with open(processor_config_path, "w") as f:
            json.dump(processor_config_data, f, indent=2)
        print("  ✓ Added auto_map to processor_config.json")


def validate_model(output_path: Path, model_type: str, config: Config) -> bool:
    """
    Run inference test to verify converted model works.

    Args:
        output_path: Directory where model was saved
        model_type: Type of model ('base' or 'embedding')
        config: Training config

    Returns:
        True if validation successful
    """
    print("\nValidating converted model...")

    try:
        # Reload from saved directory
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(str(output_path), trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(str(output_path), trust_remote_code=True)

        # Create dummy inputs
        dummy_path = torch.randn(1, config.model.max_path_len, 3)
        dummy_text = "hello"

        # Process
        inputs = processor(path_coords=dummy_path, text=dummy_text, return_tensors="pt")

        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # Validate outputs
        if model_type == "embedding":
            if "pooler_output" not in outputs:
                raise ValueError("Missing pooler_output in embedding model outputs")
            if outputs.pooler_output.shape != (1, config.model.d_model):
                raise ValueError(
                    f"Unexpected pooler_output shape: {outputs.pooler_output.shape}, "
                    f"expected (1, {config.model.d_model})"
                )
            if torch.all(outputs.pooler_output == 0):
                raise ValueError("Embeddings are all zeros")
            print(f"  ✓ Pooler output shape: {outputs.pooler_output.shape}")

        else:  # base
            if "last_hidden_state" not in outputs:
                raise ValueError("Missing last_hidden_state in base model outputs")
            seq_len = 1 + config.model.max_path_len + 1 + config.model.max_char_len
            expected_shape = (1, seq_len, config.model.d_model)
            if outputs.last_hidden_state.shape != expected_shape:
                raise ValueError(
                    f"Unexpected last_hidden_state shape: {outputs.last_hidden_state.shape}, "
                    f"expected {expected_shape}"
                )
            print(f"  ✓ Hidden states shape: {outputs.last_hidden_state.shape}")

        print("  ✓ Validation successful!")
        return True

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        print("  Model was saved but may not work correctly")
        print("  Please check the conversion manually")
        return False


def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    model_type: str = "auto",
    config_path: Path | None = None,
    validate: bool = True,
) -> bool:
    """
    Convert training checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to training checkpoint (.pt file)
        output_path: Output directory for HuggingFace model
        model_type: 'auto', 'base', or 'embedding'
        config_path: Optional path to YAML config file
        validate: Whether to run validation after conversion

    Returns:
        True if conversion successful
    """
    print(f"\n{'=' * 60}")
    print("Converting SwipeAlot Checkpoint to HuggingFace Format")
    print(f"{'=' * 60}\n")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error: Failed to load checkpoint: {e}")
        return False

    # Validate checkpoint structure
    if "model_state_dict" not in checkpoint:
        print("Error: Checkpoint missing 'model_state_dict'")
        print(f"Available keys: {list(checkpoint.keys())}")
        return False

    # Extract components
    model_state_dict = checkpoint["model_state_dict"]
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"  Found {len(model_state_dict)} state dict keys")
    print(f"  Epoch: {epoch}, Global step: {global_step}")
    if metrics:
        print(f"  Metrics: {metrics}")

    # Get config
    if "config" not in checkpoint:
        if config_path is None:
            print("Error: Checkpoint missing 'config' and no --config provided")
            return False
        print(f"Loading config from {config_path}")
        config = Config.from_yaml(str(config_path))
    else:
        config = checkpoint["config"]
        print("  ✓ Loaded config from checkpoint")

    # Create tokenizer (deterministic vocab, no dataset needed)
    print("\nCreating tokenizer...")
    char_tokenizer = CharacterTokenizer()
    config.model.vocab_size = char_tokenizer.vocab_size
    print(f"  Vocab size: {char_tokenizer.vocab_size}")

    # Determine model type
    if model_type == "auto":
        # Auto-detect based on config
        if config.model.predict_path or config.model.predict_length:
            model_type = "base"
            print(f"  Auto-detected model type: base (predict_path={config.model.predict_path})")
        else:
            model_type = "base"
            print("  Auto-detected model type: base (default)")

    print(f"\nConverting to model type: {model_type}")

    # Create HuggingFace config
    print("Creating HuggingFace config...")
    hf_config = SwipeTransformerConfig(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        vocab_size=config.model.vocab_size,
        max_path_len=config.model.max_path_len,
        max_char_len=config.model.max_char_len,
        predict_path=(config.model.predict_path if model_type == "base" else False),
        pad_token_id=char_tokenizer.pad_token_id,
        cls_token_id=char_tokenizer.cls_token_id,
        sep_token_id=char_tokenizer.sep_token_id,
        mask_token_id=char_tokenizer.mask_token_id,
        unk_token_id=char_tokenizer.unk_token_id,
        eos_token_id=char_tokenizer.eos_token_id,
    )
    print(
        f"  Config: d_model={hf_config.d_model}, n_layers={hf_config.n_layers}, "
        f"n_heads={hf_config.n_heads}"
    )

    # Filter state dict for embedding model
    if model_type == "embedding":
        print("\nFiltering state dict for embedding model...")
        filtered_state_dict = {}
        skipped = []
        for key, value in model_state_dict.items():
            if any(head in key for head in ["classifier", "char_head", "path_head", "length_head"]):
                skipped.append(key)
            else:
                filtered_state_dict[key] = value
        state_dict = filtered_state_dict
        print(f"  Keeping {len(state_dict)} encoder weights")
        print(f"  Skipping {len(skipped)} head weights")
    else:
        state_dict = model_state_dict

    # Create HuggingFace model
    print(f"\nCreating HuggingFace model ({model_type})...")
    if model_type == "base":
        hf_model = SwipeTransformerModel(hf_config)
    else:  # embedding
        hf_model = SwipeModel(hf_config)

    # Load weights
    print("Loading weights into HuggingFace model...")
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)

    # Handle warnings
    if model_type == "embedding" and unexpected_keys:
        # Filter expected unexpected keys (heads)
        head_keys = [
            k
            for k in unexpected_keys
            if any(h in k for h in ["char_head", "path_head", "classifier", "length_head"])
        ]
        other_unexpected = [k for k in unexpected_keys if k not in head_keys]
        if head_keys:
            print(f"  ✓ Skipped {len(head_keys)} head weights (expected for embedding model)")
        if other_unexpected:
            print(f"  ⚠ Warning: Unexpected keys: {other_unexpected}")
    elif unexpected_keys:
        print(f"  ⚠ Warning: Unexpected keys: {unexpected_keys}")

    if missing_keys:
        # Check if missing keys are just heads (acceptable if predict_path/predict_length=False)
        critical_missing = [k for k in missing_keys if "head" not in k]
        if critical_missing:
            print(f"  ⚠ Warning: Critical missing keys: {critical_missing}")
        else:
            print(f"  ✓ Missing {len(missing_keys)} head keys (acceptable for this config)")

    if not missing_keys and not unexpected_keys:
        print("  ✓ All weights loaded successfully")

    # Create HuggingFace tokenizer
    print("\nCreating HuggingFace tokenizer...")
    hf_tokenizer = SwipeTokenizer()
    hf_tokenizer._tokenizer = char_tokenizer

    # Create processor
    print("Creating processor...")
    hf_processor = SwipeProcessor(
        tokenizer=hf_tokenizer,
        max_path_len=config.model.max_path_len,
        max_char_len=config.model.max_char_len,
    )

    # Save to output directory
    print(f"\nSaving to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model, tokenizer, processor with safetensors
    print("  Saving model (safetensors format)...")
    hf_model.save_pretrained(output_path, safe_serialization=True)
    print("  Saving tokenizer...")
    hf_tokenizer.save_pretrained(output_path)
    print("  Saving processor...")
    hf_processor.save_pretrained(output_path)

    # Make model completely standalone (no repository dependencies)
    _make_model_standalone(output_path, model_type=model_type)

    # Save conversion metadata
    print("\nSaving conversion metadata...")
    metadata = {
        "original_checkpoint": str(checkpoint_path),
        "original_config": str(config_path) if config_path else "embedded_in_checkpoint",
        "converted_at": str(pd.Timestamp.now()),
        "model_type": model_type,
        "vocab_size": char_tokenizer.vocab_size,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
    }

    with open(output_path / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # List created files
    print("\n✓ Conversion complete!")
    print(f"  Model saved to: {output_path}")
    print("  Files created:")
    for file in sorted(output_path.iterdir()):
        if file.is_file():
            print(f"    - {file.name}")

    # Validation
    if validate:
        success = validate_model(output_path, model_type, config)
        if not success:
            return False

    # Print usage instructions
    print("\n" + "=" * 60)
    print("Success! Your model is ready to use.")
    print("=" * 60)
    print("\nUsage:")
    print("  from transformers import AutoModel, AutoProcessor")
    print(f"  model = AutoModel.from_pretrained('{output_path}', trust_remote_code=True)")
    print(f"  processor = AutoProcessor.from_pretrained('{output_path}', trust_remote_code=True)")
    if model_type == "embedding":
        print("\n  # Get embeddings")
        print("  outputs = model(**inputs)")
        print("  embeddings = outputs.pooler_output  # [batch, d_model]")
    else:
        print("\n  # Get predictions")
        print("  outputs = model(**inputs)")
        print("  char_logits = outputs.char_logits  # Character predictions")

    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert SwipeAlot training checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert base model with auto-detection
  uv run convert-hf --checkpoint checkpoints/base_20231201/best.pt --output ./hf_base

  # Convert to embedding model (strip prediction heads)
  uv run convert-hf --checkpoint checkpoints/base_20231201/best.pt \\
      --output ./hf_embedding --model-type embedding

  # Convert with external config file
  uv run convert-hf --checkpoint checkpoints/best.pt --output ./hf_model \\
      --config configs/base.yaml

  # Skip validation
  uv run convert-hf --checkpoint checkpoints/best.pt --output ./hf_model --no-validate
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to training checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "base", "embedding"],
        default="auto",
        help="Model type to convert (default: auto-detect from checkpoint config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: load from checkpoint)",
    )

    # Validation arguments
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run inference test after conversion (default)",
    )
    validation_group.add_argument(
        "--no-validate", action="store_true", help="Skip validation testing"
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # Validate config if provided
    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Determine validation flag
    validate = not args.no_validate

    # Convert
    output_path = Path(args.output)
    success = convert_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        model_type=args.model_type,
        config_path=config_path,
        validate=validate,
    )

    if not success:
        print("\n✗ Conversion failed!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Next steps:")
    print(f"{'=' * 60}")
    print("  1. Test the model locally with the usage example above")
    print("  2. Upload to HuggingFace Hub:")
    print(f"     huggingface-cli upload <your-username>/<model-name> {output_path}")


if __name__ == "__main__":
    main()
