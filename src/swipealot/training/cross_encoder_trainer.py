"""Trainer for cross-encoder model."""

import os

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from swipealot.utils import batch_to_device

from .loss import MultipleNegativesRankingLoss


class CrossEncoderTrainer:
    """Trainer for cross-encoder with gradual unfreezing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        loss_fn: MultipleNegativesRankingLoss,
        device: torch.device,
        config,
        scheduler=None,
    ):
        """
        Initialize cross-encoder trainer.

        Args:
            model: SwipeCrossEncoderModel
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_fn: MNR loss function
            device: Device to train on
            config: CrossEncoderConfig object
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

        # Training config shortcut
        self.train_config = config.training

        # TensorBoard logging
        os.makedirs(self.train_config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.train_config.log_dir)
        self.global_step = 0

        # Checkpointing
        os.makedirs(self.train_config.checkpoint_dir, exist_ok=True)
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.train_config.checkpoint_dir,
            filename="checkpoint_epoch_{epoch}",
            save_top_k=self.train_config.keep_n_checkpoints,
            monitor="val_accuracy",
            mode="max",
            save_last=True,
            verbose=True,
        )

        # Mixed precision
        self.amp_dtype = None
        if self.train_config.use_amp:
            if self.train_config.amp_dtype == "bfloat16":
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16

        self.scaler = (
            torch.cuda.amp.GradScaler()
            if (self.train_config.use_amp and self.train_config.amp_dtype == "float16")
            else None
        )

        # Track best validation accuracy
        self.best_val_accuracy = 0.0

        # Apply encoder freezing
        if self.train_config.freeze_encoder:
            print("Freezing encoder (only training classification head)")
            self.model.freeze_encoder()
        else:
            print("Training both encoder and classification head")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Average training loss
        """
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            # Move to device
            batch = batch_to_device(batch, self.device)

            # Reshape batch from collator format
            # batch["path_coords"]: [batch*(1+N), path_len, 3]
            # batch["char_tokens"]: [batch*(1+N), word_len]
            # batch["labels"]: [batch] (all zeros)
            # batch["group_sizes"]: [batch] (all 1+N)

            # Forward pass with mixed precision
            if self.train_config.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    # Get scores for all pairs
                    scores = self.model(
                        path_coords=batch["path_coords"],
                        char_tokens=batch["char_tokens"],
                        attention_mask=batch["attention_mask"],
                    )  # [batch*(1+N), 1]

                    # Reshape scores into groups: [batch, 1+N]
                    batch_size = len(batch["group_sizes"])
                    group_size = batch["group_sizes"][0].item()  # Assuming all same
                    scores_grouped = scores.view(batch_size, group_size)

                    # Compute MNR loss
                    loss = self.loss_fn(scores_grouped, batch["labels"])

                # Backward pass
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                # No mixed precision
                scores = self.model(
                    path_coords=batch["path_coords"],
                    char_tokens=batch["char_tokens"],
                    attention_mask=batch["attention_mask"],
                )
                batch_size = len(batch["group_sizes"])
                group_size = batch["group_sizes"][0].item()
                scores_grouped = scores.view(batch_size, group_size)
                loss = self.loss_fn(scores_grouped, batch["labels"])

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Compute accuracy (positive is index 0, highest score?)
            predictions = scores_grouped.argmax(dim=1)
            accuracy = (predictions == batch["labels"]).float().mean()

            # Logging
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy.item())

            if self.global_step % self.train_config.log_interval == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/accuracy", accuracy.item(), self.global_step)
                if self.scheduler:
                    self.writer.add_scalar(
                        "train/lr", self.scheduler.get_last_lr()[0], self.global_step
                    )

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}"})
            self.global_step += 1

            # Validation
            if self.global_step % self.train_config.val_interval == 0:
                val_loss, val_accuracy = self.validate()
                self.writer.add_scalar("val/loss", val_loss, self.global_step)
                self.writer.add_scalar("val/accuracy", val_accuracy, self.global_step)
                self.model.train()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

        return avg_loss

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """
        Run validation.

        Returns:
            (val_loss, val_accuracy)
        """
        self.model.eval()
        val_losses = []
        val_accuracies = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = batch_to_device(batch, self.device)

            # Forward pass
            if self.train_config.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    scores = self.model(
                        path_coords=batch["path_coords"],
                        char_tokens=batch["char_tokens"],
                        attention_mask=batch["attention_mask"],
                    )
                    batch_size = len(batch["group_sizes"])
                    group_size = batch["group_sizes"][0].item()
                    scores_grouped = scores.view(batch_size, group_size)
                    loss = self.loss_fn(scores_grouped, batch["labels"])
            else:
                scores = self.model(
                    path_coords=batch["path_coords"],
                    char_tokens=batch["char_tokens"],
                    attention_mask=batch["attention_mask"],
                )
                batch_size = len(batch["group_sizes"])
                group_size = batch["group_sizes"][0].item()
                scores_grouped = scores.view(batch_size, group_size)
                loss = self.loss_fn(scores_grouped, batch["labels"])

            # Compute accuracy
            predictions = scores_grouped.argmax(dim=1)
            accuracy = (predictions == batch["labels"]).float().mean()

            val_losses.append(loss.item())
            val_accuracies.append(accuracy.item())

        avg_loss = sum(val_losses) / len(val_losses)
        avg_accuracy = sum(val_accuracies) / len(val_accuracies)

        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        return avg_loss, avg_accuracy

    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("STARTING CROSS-ENCODER TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.train_config.num_epochs}")
        print(f"Freeze encoder: {self.train_config.freeze_encoder}")
        print("=" * 60)

        for epoch in range(self.train_config.num_epochs):
            self.train_epoch(epoch)

            # Validation at end of epoch
            val_loss, val_accuracy = self.validate()

            # Save checkpoint
            if (epoch + 1) % self.train_config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.train_config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self.save_checkpoint(checkpoint_path, epoch, val_accuracy)

                # Save best model
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    best_path = os.path.join(self.train_config.checkpoint_dir, "best_model.pt")
                    self.save_checkpoint(best_path, epoch, val_accuracy)
                    print(f"New best model saved! Accuracy: {val_accuracy:.4f}")

        print("=" * 60)
        print("TRAINING COMPLETE")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print("=" * 60)

        self.writer.close()

    def save_checkpoint(self, path: str, epoch: int, val_accuracy: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "global_step": self.global_step,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
