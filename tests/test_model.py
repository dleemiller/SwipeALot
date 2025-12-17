"""Quick test to verify model can be instantiated and run."""

import torch

from swipealot.huggingface import SwipeTransformerConfig, SwipeTransformerModel
from swipealot.training import SwipeLoss


def test_model():
    """Test basic model functionality."""
    print("Testing model instantiation and forward pass...")

    # Create simple config
    config = SwipeTransformerConfig(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        max_path_len=32,
        max_char_len=20,
        predict_path=True,
    )

    # Create model
    model = SwipeTransformerModel(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy inputs
    batch_size = 2
    path_coords = torch.randn(batch_size, config.max_path_len, config.path_input_dim)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_char_len))
    attention_mask = torch.ones(
        batch_size, 1 + config.max_path_len + 1 + config.max_char_len
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(path_coords=path_coords, input_ids=input_ids, attention_mask=attention_mask)

    print("✓ Forward pass successful")
    print(f"  - char_logits shape: {outputs['char_logits'].shape}")
    if "path_coords_pred" in outputs:
        print(f"  - path_coords_pred shape: {outputs['path_coords_pred'].shape}")

    # Test loss
    loss_fn = SwipeLoss()
    # Create char_labels: some positions are -100 (ignore), others are valid token IDs
    char_labels = torch.randint(0, config.vocab_size, (batch_size, config.max_char_len))
    # Randomly set some positions to -100 (ignored in loss)
    mask = torch.rand(batch_size, config.max_char_len) > 0.85
    char_labels[mask] = -100

    batch = {
        "path_coords": path_coords,
        "input_ids": input_ids,
        "char_labels": char_labels,
        "path_labels": torch.randn(batch_size, config.max_path_len, config.path_input_dim).clamp(
            0, 1
        ),  # [0, 1] range
        "path_mask_indices": torch.randint(0, 2, (batch_size, config.max_path_len)),
    }
    losses = loss_fn(outputs, batch)
    print("✓ Loss computation successful")
    print(f"  - total_loss: {losses['total_loss'].item():.4f}")
    print(f"  - char_loss: {losses['char_loss'].item():.4f}")
    if "path_loss" in losses:
        print(f"  - path_loss: {losses['path_loss'].item():.4f}")

    assert losses["total_loss"].isfinite()


if __name__ == "__main__":
    test_model()
