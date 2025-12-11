"""Trainer for swipe keyboard model."""

import os
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from swipealot.utils import batch_to_device, extract_character_logits

from .loss import SwipeLoss
from .metrics import CharacterAccuracy, WordAccuracy


class SwipeTrainer:
    """Trainer for swipe keyboard model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        loss_fn: SwipeLoss,
        device: torch.device,
        config,
        tokenizer=None,
        scheduler=None,
    ):
        """
        Initialize trainer.

        Args:
            model: SwipeTransformerModel
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_fn: Loss function
            device: Device to train on
            config: Full Config object (or TrainingConfig for backward compatibility)
            tokenizer: Character tokenizer (optional, for word accuracy)
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.tokenizer = tokenizer

        # Handle both full Config and TrainingConfig (backward compatibility)
        if hasattr(config, "training"):
            # Full Config object
            train_config = config.training
        else:
            # Legacy TrainingConfig only
            train_config = config

        # Metrics
        vocab_size = tokenizer.vocab_size if tokenizer else 128
        self.char_accuracy = CharacterAccuracy(vocab_size=vocab_size, device=str(device))
        if tokenizer:
            self.word_accuracy = WordAccuracy(tokenizer)
        else:
            self.word_accuracy = None

        # TensorBoard logging
        os.makedirs(train_config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=train_config.log_dir)
        self.global_step = 0

        # Checkpointing with Lightning ModelCheckpoint
        os.makedirs(train_config.checkpoint_dir, exist_ok=True)

        # Create Lightning ModelCheckpoint callback for better checkpoint management
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=train_config.checkpoint_dir,
            filename="checkpoint_epoch_{epoch}",
            save_top_k=train_config.keep_n_checkpoints,
            monitor="char_accuracy",  # Monitor character accuracy
            mode="max",  # Save checkpoints with highest accuracy
            save_last=True,  # Always save the last checkpoint
            verbose=True,
        )

        # Determine dtype for mixed precision
        self.amp_dtype = None
        if train_config.use_amp:
            if train_config.amp_dtype == "bfloat16":
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16

        # Gradient scaler for mixed precision (only for float16, not bfloat16)
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if (train_config.use_amp and train_config.amp_dtype == "float16")
            else None
        )

        # Store train_config for easy access
        self.train_config = train_config

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            # Move to device
            batch = batch_to_device(batch, self.device)

            # Forward pass with mixed precision
            if self.train_config.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(
                        path_coords=batch["path_coords"],
                        char_tokens=batch["char_tokens"],
                        attention_mask=batch["attention_mask"],
                    )
                    losses = self.loss_fn(outputs, batch)
                    loss = losses["total_loss"]

                # Backward pass
                self.optimizer.zero_grad()
                if self.scaler:  # float16 uses gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:  # bfloat16 doesn't need scaling
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                outputs = self.model(
                    path_coords=batch["path_coords"],
                    char_tokens=batch["char_tokens"],
                    attention_mask=batch["attention_mask"],
                )
                losses = self.loss_fn(outputs, batch)
                loss = losses["total_loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Logging
            epoch_losses.append(loss.item())
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            if self.global_step % self.train_config.log_interval == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)
                for k, v in losses.items():
                    if k != "total_loss":
                        self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)

            # Step-based validation
            if self.global_step % self.train_config.val_interval == 0:
                print(f"\n[Step {self.global_step}] Running validation...")
                val_metrics = self.evaluate_step(self.global_step)

                # Save best checkpoint
                current_accuracy = val_metrics["char_accuracy"]
                if not hasattr(self, "best_accuracy"):
                    self.best_accuracy = 0.0
                if current_accuracy > self.best_accuracy:
                    self.best_accuracy = current_accuracy
                    self.save_checkpoint(epoch, val_metrics)

                # Return to training mode
                self.model.train()

            self.global_step += 1

        return sum(epoch_losses) / len(epoch_losses)

    def evaluate_step(self, step: int) -> dict[str, float]:
        """
        Evaluate on validation set (called during training steps).

        Args:
            step: Current training step

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.char_accuracy.reset()
        if self.word_accuracy:
            self.word_accuracy.reset()
        eval_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(
                    path_coords=batch["path_coords"],
                    char_tokens=batch["char_tokens"],
                    attention_mask=batch["attention_mask"],
                )

                losses = self.loss_fn(outputs, batch)
                eval_losses.append(losses["total_loss"].item())

                # Extract character predictions
                char_logits = outputs["char_logits"]
                path_len = batch["path_coords"].shape[1]
                char_logits_subset = extract_character_logits(
                    char_logits, path_len, batch["char_labels"].shape[1]
                )

                # Update metrics
                self.char_accuracy.update(char_logits_subset, batch["char_labels"])

                if self.word_accuracy:
                    self.word_accuracy.update(char_logits_subset, batch["words"])

        # Compute metrics
        avg_loss = sum(eval_losses) / len(eval_losses)
        char_acc = self.char_accuracy.compute()

        metrics = {"loss": avg_loss, "char_accuracy": char_acc}

        if self.word_accuracy:
            word_acc = self.word_accuracy.compute()
            metrics["word_accuracy"] = word_acc

        # Log to TensorBoard
        self.writer.add_scalar("eval/loss", avg_loss, step)
        self.writer.add_scalar("eval/char_accuracy", char_acc, step)
        if self.word_accuracy:
            self.writer.add_scalar("eval/word_accuracy", word_acc, step)

        # Print metrics
        print(f"Val - Loss: {avg_loss:.4f}, Char Accuracy: {char_acc:.4f}", end="")
        if self.word_accuracy:
            print(f", Word Accuracy: {word_acc:.4f}")
        else:
            print()

        return metrics

    def evaluate(self, epoch: int) -> dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.char_accuracy.reset()
        if self.word_accuracy:
            self.word_accuracy.reset()
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = batch_to_device(batch, self.device)

                outputs = self.model(
                    path_coords=batch["path_coords"],
                    char_tokens=batch["char_tokens"],
                    attention_mask=batch["attention_mask"],
                )

                losses = self.loss_fn(outputs, batch)
                eval_losses.append(losses["total_loss"].item())

                # Extract character predictions
                char_logits = outputs["char_logits"]
                path_len = batch["path_coords"].shape[1]
                char_logits_subset = extract_character_logits(
                    char_logits, path_len, batch["char_labels"].shape[1]
                )

                # Update metrics
                self.char_accuracy.update(char_logits_subset, batch["char_labels"])

                if self.word_accuracy:
                    self.word_accuracy.update(char_logits_subset, batch["words"])

        # Compute metrics
        avg_loss = sum(eval_losses) / len(eval_losses)
        char_acc = self.char_accuracy.compute()

        metrics = {"loss": avg_loss, "char_accuracy": char_acc}

        if self.word_accuracy:
            word_acc = self.word_accuracy.compute()
            metrics["word_accuracy"] = word_acc

        # Log to TensorBoard (use global_step for proper alignment with training metrics)
        self.writer.add_scalar("eval/loss", avg_loss, self.global_step)
        self.writer.add_scalar("eval/char_accuracy", char_acc, self.global_step)
        if self.word_accuracy:
            self.writer.add_scalar("eval/word_accuracy", word_acc, self.global_step)

        # Print metrics
        print(
            f"\nEval Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Char Accuracy: {char_acc:.4f}",
            end="",
        )
        if self.word_accuracy:
            print(f", Word Accuracy: {word_acc:.4f}")
        else:
            print()

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]):
        """
        Save model checkpoint using Lightning ModelCheckpoint.

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics (must include 'char_accuracy' for monitoring)
        """
        # Create checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "global_step": self.global_step,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save using Lightning callback (handles cleanup automatically)
        filepath = Path(self.train_config.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, filepath)

        # Update callback's best model tracking
        if "char_accuracy" in metrics:
            current_score = metrics["char_accuracy"]
            self.checkpoint_callback.update_best_and_save(
                current_score, trainer=None, pl_module=None
            )

        print(f"Saved checkpoint: {filepath}")

    def train(self, num_epochs: int):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        # Initialize best_accuracy for step-based validation
        self.best_accuracy = 0.0

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            # End-of-epoch evaluation
            print(f"\n[Epoch {epoch + 1}] Running end-of-epoch validation...")
            eval_metrics = self.evaluate(epoch)

            # Save epoch checkpoint (Lightning callback handles cleanup automatically)
            if (epoch + 1) % self.train_config.save_interval == 0:
                self.save_checkpoint(epoch, eval_metrics)

        self.writer.close()
        print(f"\nTraining complete! Best character accuracy: {self.best_accuracy:.4f}")
