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

    def __init__(self, loss_fn=None, eval_collator=None, **kwargs):
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

        # Extract predictions - we want char_logits for character accuracy
        if hasattr(outputs, "char_logits"):
            predictions = outputs.char_logits
        else:
            predictions = outputs.get("char_logits") if isinstance(outputs, dict) else None

        # Extract labels
        labels = inputs.get("char_labels") if isinstance(inputs, dict) else None

        # Move to CPU for metrics computation
        if predictions is not None:
            predictions = predictions.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()

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
        Compute character and word accuracy metrics.

        Args:
            eval_pred: EvalPrediction with predictions and label_ids
                predictions: Model outputs (SwipeTransformerOutput)
                label_ids: Dict with char_labels and optionally words

        Returns:
            Dict with metrics
        """
        try:
            # Extract predictions and labels
            predictions, labels = eval_pred.predictions, eval_pred.label_ids

            # Predictions are the full sequence char_logits: [batch, seq_len, vocab_size]
            # Labels are just the character portion: [batch, char_len]
            # Sequence structure: [CLS] + path + [SEP] + chars
            # We need to extract the character portion from predictions

            # Convert to torch tensors if needed
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            # Extract character portion from sequence
            # Sequence is: [CLS] + path + [SEP] + chars
            # Calculate where chars start dynamically
            total_seq_len = predictions.shape[1]
            char_len = labels.shape[1]
            # path_len = total_seq_len - 1 (CLS) - 1 (SEP) - char_len
            path_len = total_seq_len - 2 - char_len
            char_start = 1 + path_len + 1  # After [CLS] + path + [SEP]
            char_logits = predictions[
                :, char_start : char_start + char_len, :
            ]  # [batch, char_len, vocab_size]

            char_labels = labels

            # Compute character accuracy
            char_accuracy_metric = CharacterAccuracy(vocab_size=tokenizer.vocab_size, device="cpu")
            char_accuracy_metric.update(char_logits, char_labels)
            char_acc = char_accuracy_metric.compute()

            metrics = {
                "char_accuracy": float(char_acc),
            }

            # Compute word accuracy if words are available
            # Note: words are not typically available in the standard Trainer eval_pred
            # This would require a custom data collator that preserves words in labels

            return metrics
        except Exception as e:
            logger.error(f"Error in compute_metrics: {e}", exc_info=True)
            # Return empty dict on error
            return {}

    return compute_metrics
