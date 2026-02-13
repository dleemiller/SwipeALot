"""Trainer for distillation fine-tuning with CTC loss."""

from __future__ import annotations

import logging

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class SwipeDistillTrainer(Trainer):
    """Trainer with separate LR groups for encoder vs new modules."""

    def __init__(self, *, processor=None, eval_collator=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.eval_collator = eval_collator
        self._last_log_step = -1

    def get_eval_dataloader(self, eval_dataset=None):
        """Use eval_collator if provided."""
        if self.eval_collator is not None:
            original = self.data_collator
            self.data_collator = self.eval_collator
            loader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = original
            return loader
        return super().get_eval_dataloader(eval_dataset)

    def create_optimizer(self):
        """Create optimizer with separate LR groups."""
        if self.optimizer is not None:
            return self.optimizer

        encoder_lr_scale = float(getattr(self.model.config, "encoder_lr_scale", 0.1))
        base_lr = self.args.learning_rate

        encoder_params = self.model.get_encoder_params()
        new_params = self.model.get_new_params()

        # Only include encoder params that require grad
        encoder_params = [p for p in encoder_params if p.requires_grad]
        new_params = [p for p in new_params if p.requires_grad]

        param_groups = []
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": base_lr * encoder_lr_scale,
                    "name": "encoder",
                }
            )
        if new_params:
            param_groups.append(
                {
                    "params": new_params,
                    "lr": base_lr,
                    "name": "new_modules",
                }
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        # Remove lr from kwargs since we set it per-group
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, return_dict=True)
        loss = outputs.loss

        if loss is None:
            raise RuntimeError("Model did not return a loss.")

        if model.training:
            step = int(self.state.global_step)
            log_every = int(self.args.logging_steps) if int(self.args.logging_steps) > 0 else 1
            if step != self._last_log_step and step % log_every == 0:
                log_items = {}
                log_items["ctc_loss"] = float(loss.detach().cpu())

                # Projector output stats
                if outputs.projected is not None:
                    proj = outputs.projected.detach()
                    log_items["projector_mean"] = float(proj.mean().cpu())
                    log_items["projector_std"] = float(proj.std().cpu())
                    log_items["projector_norm"] = float(proj.norm(dim=1).mean().cpu())

                if log_items:
                    self.log(log_items)
                self._last_log_step = step

        if return_outputs:
            return loss, outputs
        return loss

    def _save(self, output_dir, state_dict=None):
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )
        if self.processor is not None:
            try:
                self.processor.save_pretrained(output_dir)
            except Exception as e:
                logger.warning(f"Failed to save processor: {e}")

        # Also save adapter+decoder weights separately for Phase 3
        try:
            adapter_decoder_state = self.model.get_adapter_decoder_state_dict()
            torch.save(adapter_decoder_state, f"{output_dir}/adapter_decoder.pt")
        except Exception as e:
            logger.warning(f"Failed to save adapter+decoder state: {e}")

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        loss = getattr(outputs, "loss", None)
        if prediction_loss_only:
            return (loss, None, None)

        preds = outputs.logits  # [B, T', num_chars+1]
        labels = (inputs.get("labels"), inputs.get("label_lengths"))

        def _to_cpu(x):
            if x is None:
                return None
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x

        preds = _to_cpu(preds)
        labels = tuple(_to_cpu(x) for x in labels)
        return (loss, preds, labels)


def create_compute_metrics_fn():
    """Return a compute_metrics function for CTC output."""

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        if preds is None or labels is None:
            return {}

        # For now, report average CTC loss (already in eval_loss)
        return {}

    return compute_metrics
