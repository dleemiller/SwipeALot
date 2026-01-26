"""Minimal HuggingFace Trainer subclass for SwipeALot with custom loss."""

import logging

import torch
from transformers import Trainer

from .metrics import CharacterAccuracy

logger = logging.getLogger(__name__)


class SwipeTrainer(Trainer):
    """
    Minimal trainer for SwipeALot with custom loss computation.

    This trainer subclass only overrides compute_loss() to use SwipeLoss
    instead of the default CrossEntropyLoss. Everything else uses the
    standard HuggingFace Trainer functionality.
    """

    def __init__(
        self, loss_fn=None, eval_collator=None, path_resample_mode: str = "time", **kwargs
    ):
        """
        Initialize SwipeTrainer.

        Args:
            loss_fn: SwipeLoss instance for computing loss
            eval_collator: Optional separate collator for evaluation
            **kwargs: All other arguments passed to transformers.Trainer
        """
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.eval_collator = eval_collator
        self.path_resample_mode = path_resample_mode
        self._train_collator = self.data_collator

    def _save(self, output_dir, state_dict=None):
        """
        Save model checkpoint with remote code files for AutoModel compatibility.
        """
        from .checkpoint_utils import prepare_checkpoint_for_hub

        # Save model and config
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )

        # Prepare checkpoint for HuggingFace Hub
        # Copies modeling files, fixes imports, adds auto_map
        prepare_checkpoint_for_hub(output_dir)

        # Save tokenizer and processor for hub-ready checkpoints
        try:
            from swipealot.huggingface import SwipeProcessor, SwipeTokenizer

            # Get CharacterTokenizer from data_collator
            if hasattr(self.data_collator, "tokenizer"):
                char_tokenizer = self.data_collator.tokenizer

                # Wrap in SwipeTokenizer
                hf_tokenizer = SwipeTokenizer()
                hf_tokenizer._tokenizer = char_tokenizer
                hf_tokenizer.save_pretrained(output_dir)

                # Save processor (auto_map is added automatically in save_pretrained)
                hf_processor = SwipeProcessor(
                    tokenizer=hf_tokenizer,
                    max_path_len=self.model.config.max_path_len,
                    max_char_len=self.model.config.max_char_len,
                    path_input_dim=getattr(self.model.config, "path_input_dim", 6),
                    path_resample_mode=self.path_resample_mode,
                )
                hf_processor.save_pretrained(output_dir)
        except Exception as e:
            # Don't fail checkpoint save if tokenizer/processor save fails
            logger.warning(f"Failed to save tokenizer/processor: {e}")

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Get evaluation dataloader, using eval_collator if provided.
        """
        if self.eval_collator is not None:
            # Temporarily swap collators
            original_collator = self.data_collator
            self.data_collator = self.eval_collator
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = original_collator
            return dataloader
        return super().get_eval_dataloader(eval_dataset)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform an evaluation step, extracting predictions for compute_metrics.
        """
        import torch

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract loss if available
        loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None

        if prediction_loss_only:
            return (loss, None, None)

        # Extract predictions
        if hasattr(outputs, "char_logits"):
            char_logits = outputs.char_logits
            length_logits = getattr(outputs, "length_logits", None)
        elif isinstance(outputs, dict):
            char_logits = outputs.get("char_logits")
            length_logits = outputs.get("length_logits")
        else:
            char_logits = None
            length_logits = None

        predictions = (char_logits, length_logits)

        # Extract labels (for metrics)
        char_labels = inputs.get("char_labels") if isinstance(inputs, dict) else None
        length_target = inputs.get("length_target") if isinstance(inputs, dict) else None
        length_supervise_mask = (
            inputs.get("length_supervise_mask") if isinstance(inputs, dict) else None
        )
        labels = (char_labels, length_target, length_supervise_mask)

        # Move to CPU for metrics computation
        def _to_cpu(x):
            if x is None:
                return None
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x

        predictions = tuple(_to_cpu(x) for x in predictions)
        labels = tuple(_to_cpu(x) for x in labels)

        return (loss, predictions, labels)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss using SwipeLoss.

        Args:
            model: SwipeTransformerModel
            inputs: Dict with input_ids, path_coords, attention_mask, labels
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for newer transformers)

        Returns:
            loss (torch.Tensor) or (loss, outputs) if return_outputs=True
        """
        # Forward pass through model
        outputs = model(**inputs)

        # Compute loss using SwipeLoss
        losses = self.loss_fn(outputs, inputs)
        loss = losses["total_loss"]

        # Log individual loss components (less frequently to avoid spam)
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in losses.items():
                if key != "total_loss":
                    self.log({f"train/{key}": value.item()})
            if (
                hasattr(self.data_collator, "stats")
                and getattr(self.args, "dataloader_num_workers", 0) == 0
            ):
                stats = self.data_collator.stats.summarize()
                for mode, values in stats.items():
                    self.log(
                        {
                            f"masking/{mode}_count": values.get("count", 0.0),
                            f"masking/{mode}_path_frac": values.get("path_mask_frac_mean", 0.0),
                            f"masking/{mode}_char_frac": values.get("char_mask_frac_mean", 0.0),
                        }
                    )
                self.data_collator.stats.reset()

        return (loss, outputs) if return_outputs else loss


def create_compute_metrics_fn(tokenizer):
    """
    Create a compute_metrics function for the Trainer.

    Args:
        tokenizer: CharacterTokenizer for word accuracy computation

    Returns:
        Function that computes metrics from EvalPrediction
    """

    def compute_metrics(eval_pred):
        """
        Compute character accuracy metrics, and (if available) length regression metrics.

        Args:
            eval_pred: EvalPrediction with predictions and label_ids
                predictions: Either:
                    - char_logits
                    - (char_logits, length_logits)
                label_ids: Either:
                    - char_labels
                    - (char_labels, length_target, length_supervise_mask)

        Returns:
            Dict with metrics
        """
        try:
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids

            # Unpack tuple predictions/labels (newer SwipeTrainer.prediction_step)
            if isinstance(predictions, (tuple, list)):
                char_pred = predictions[0] if len(predictions) > 0 else None
                length_pred = predictions[1] if len(predictions) > 1 else None
            else:
                char_pred = predictions
                length_pred = None

            if isinstance(labels, (tuple, list)):
                char_labels = labels[0] if len(labels) > 0 else None
                length_target = labels[1] if len(labels) > 1 else None
                length_mask = labels[2] if len(labels) > 2 else None
            else:
                char_labels = labels
                length_target = None
                length_mask = None

            # Convert to torch tensors if needed
            if char_pred is None or char_labels is None:
                return {}

            if not isinstance(char_pred, torch.Tensor):
                char_pred = torch.tensor(char_pred)
            if not isinstance(char_labels, torch.Tensor):
                char_labels = torch.tensor(char_labels)

            # Character logits: prefer new shape [batch, char_len, vocab_size].
            # Fall back to legacy full-sequence logits [batch, seq_len, vocab_size].
            char_len = char_labels.shape[1]
            if char_pred.dim() == 3 and char_pred.shape[1] == char_len:
                char_logits = char_pred
            else:
                total_seq_len = char_pred.shape[1]
                path_len = total_seq_len - 2 - char_len
                char_start = 1 + path_len + 1  # After [CLS] + path + [SEP]
                char_logits = char_pred[:, char_start : char_start + char_len, :]

            # Compute character accuracy
            char_accuracy_metric = CharacterAccuracy(vocab_size=tokenizer.vocab_size, device="cpu")
            char_accuracy_metric.update(char_logits, char_labels)
            char_acc = char_accuracy_metric.compute()

            metrics = {
                "char_accuracy": float(char_acc),
            }

            # Length regression metrics (if available)
            if length_pred is not None and length_target is not None:
                if not isinstance(length_pred, torch.Tensor):
                    length_pred = torch.tensor(length_pred)
                if not isinstance(length_target, torch.Tensor):
                    length_target = torch.tensor(length_target)
                if length_mask is None:
                    length_mask_t = torch.ones_like(length_target, dtype=torch.bool)
                else:
                    length_mask_t = (
                        length_mask.bool()
                        if isinstance(length_mask, torch.Tensor)
                        else torch.tensor(length_mask).bool()
                    )

                length_pred = length_pred.reshape(-1).float()
                length_target = length_target.reshape(-1).float()
                length_mask_t = length_mask_t.reshape(-1)

                if length_mask_t.any():
                    err = (length_pred - length_target).abs()[length_mask_t]
                    metrics["length_mae"] = float(err.mean().item())
                    metrics["length_rmse"] = float(
                        ((length_pred - length_target) ** 2)[length_mask_t].mean().sqrt().item()
                    )

                    pred_round = torch.round(length_pred).clamp(min=0.0)
                    diff_round = (pred_round - length_target).abs()[length_mask_t]
                    metrics["length_acc_within_1"] = float(
                        (diff_round <= 1.0).float().mean().item()
                    )

            return metrics
        except Exception as e:
            logger.error(f"Error in compute_metrics: {e}", exc_info=True)
            # Return empty dict on error
            return {}

    return compute_metrics
