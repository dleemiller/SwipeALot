"""Quick test training run to verify everything works."""

import os

import torch
from torch.utils.data import DataLoader

from swipealot.config import Config
from swipealot.data import (
    MaskedCollator,
    SwipeDataset,
    compute_char_frequency_weights,
)
from swipealot.models import SwipeTransformerModel
from swipealot.training import SwipeLoss, SwipeTrainer


def test_training(num_steps=20, batch_size=32):
    """Run a quick training test."""

    print("=" * 60)
    print("QUICK TRAINING TEST")
    print("=" * 60)

    # Load config
    config = Config.from_yaml("configs/tiny.yaml")
    config.data.batch_size = batch_size
    config.training.log_interval = 5

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Build tokenizer from small sample
    print("\nBuilding tokenizer...")
    sample_dataset = SwipeDataset(
        split="train",
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        dataset_name=config.data.dataset_name,
        max_samples=5000,
    )
    tokenizer = sample_dataset.tokenizer
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Optional inverse-log frequency weights (disabled by default)
    char_freq_weights = None
    if config.training.use_char_freq_weights:
        print("\nComputing character frequency weights...")
        char_freq_weights = compute_char_frequency_weights(
            tokenizer, sample_dataset.dataset, max_samples=config.training.char_freq_max_samples
        )
        char_freq_weights = char_freq_weights.to(device)
        print(f"Computed weights for {len(char_freq_weights)} tokens")

    # Create small datasets for testing (enough for 20 steps with batch_size=32)
    num_samples = num_steps * batch_size + 50  # ~690 samples

    print(f"\nLoading {num_samples} training samples...")
    train_dataset = SwipeDataset(
        split="train",
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=num_samples,
    )

    print("Loading 100 validation samples...")
    val_dataset = SwipeDataset(
        split="validation",
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=100,
    )

    # Create data loaders
    print("\nCreating data loaders...")
    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=config.data.char_mask_prob,
        path_mask_prob=config.data.path_mask_prob,
        mask_path=config.data.mask_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = SwipeTransformerModel(config.model)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create loss function
    loss_fn = SwipeLoss(
        char_weight=config.training.char_loss_weight,
        path_weight=config.training.path_loss_weight,
        focal_gamma=config.training.focal_gamma if config.training.use_focal_loss else 0.0,
        char_class_weights=char_freq_weights,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SwipeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config.training,
        tokenizer=tokenizer,
    )

    # Run limited training
    print("\n" + "=" * 60)
    print(f"RUNNING {num_steps} TRAINING STEPS")
    print("=" * 60)

    model.train()
    from tqdm import tqdm

    step = 0
    for _batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        if step >= num_steps:
            break

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        if config.training.use_amp and device.type == "cuda":
            amp_dtype = torch.bfloat16 if config.training.amp_dtype == "bfloat16" else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                outputs = model(
                    path_coords=batch["path_coords"],
                    char_tokens=batch["char_tokens"],
                    attention_mask=batch["attention_mask"],
                )
                losses = loss_fn(outputs, batch)
                loss = losses["total_loss"]

            # Backward pass
            optimizer.zero_grad()
            if trainer.scaler is not None:
                trainer.scaler.scale(loss).backward()
                trainer.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            outputs = model(
                path_coords=batch["path_coords"],
                char_tokens=batch["char_tokens"],
                attention_mask=batch["attention_mask"],
            )
            losses = loss_fn(outputs, batch)
            loss = losses["total_loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Log
        if step % config.training.log_interval == 0:
            print(f"\nStep {step}/{num_steps}:")
            print(f"  Total loss: {loss.item():.4f}")
            print(f"  Char loss:  {losses['char_loss'].item():.4f}")
            if "path_loss" in losses:
                print(f"  Path loss:  {losses['path_loss'].item():.4f}")

        step += 1

    # Run validation
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION")
    print("=" * 60)

    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            outputs = model(
                path_coords=batch["path_coords"],
                char_tokens=batch["char_tokens"],
                attention_mask=batch["attention_mask"],
            )

            losses = loss_fn(outputs, batch)
            val_losses.append(losses["total_loss"].item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"\nValidation loss: {avg_val_loss:.4f}")

    # Save checkpoint
    print("\n" + "=" * 60)
    print("SAVING CHECKPOINT")
    print("=" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/test_run.pt"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "val_loss": avg_val_loss,
    }

    if trainer.scaler and device.type == "cuda":
        checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint to: {checkpoint_path}")

    # Verify we can load it
    print("\nVerifying checkpoint...")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print("✓ Checkpoint loaded successfully")
    print(f"  - Step: {loaded['step']}")
    print(f"  - Vocab size: {loaded['tokenizer_vocab_size']}")
    print(f"  - Val loss: {loaded['val_loss']:.4f}")

    # Check TensorBoard logs
    print("\n" + "=" * 60)
    print("CHECKING TENSORBOARD LOGS")
    print("=" * 60)

    if os.path.exists("logs"):
        import glob

        log_files = glob.glob("logs/**/*", recursive=True)
        print(f"✓ Found {len(log_files)} log files in logs/")
        print("\nTo view logs, run:")
        print("  tensorboard --logdir logs")
    else:
        print("⚠ No logs directory found")

    print("\n" + "=" * 60)
    print("TEST TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n✓ Trained for {step} steps")
    print(f"✓ Validation loss: {avg_val_loss:.4f}")
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print("✓ All systems working!")


if __name__ == "__main__":
    test_training(num_steps=20, batch_size=32)
