"""Trainer utilities for downstream text-to-path CVAE fine-tuning."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class SwipeTextToPathCVAETrainer(Trainer):
    """Trainer that exposes deterministic XY predictions for metrics."""

    def __init__(self, *, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self._last_cvae_log_step = -1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, return_dict=True)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss.")

        if model.training:
            step = int(self.state.global_step)
            log_every = int(self.args.logging_steps) if int(self.args.logging_steps) > 0 else 1
            if step != self._last_cvae_log_step and step % log_every == 0:
                log_items = {"kl_weight": float(getattr(model.config, "kl_weight", 0.0))}
                if outputs.recon_loss is not None:
                    log_items["recon_loss"] = float(outputs.recon_loss.detach().cpu())
                if outputs.kl_loss is not None:
                    log_items["kl_loss"] = float(outputs.kl_loss.detach().cpu())
                if outputs.smoothness_loss is not None:
                    log_items["smoothness_loss"] = float(outputs.smoothness_loss.detach().cpu())
                if outputs.speed_smoothness_loss is not None:
                    log_items["speed_smoothness_loss"] = float(
                        outputs.speed_smoothness_loss.detach().cpu()
                    )
                if getattr(outputs, "uncertainty_penalty", None) is not None:
                    log_items["uncertainty_penalty"] = float(
                        outputs.uncertainty_penalty.detach().cpu()
                    )
                if outputs.log_sigma is not None:
                    log_items["log_sigma_mean"] = float(outputs.log_sigma.detach().mean().cpu())
                if outputs.post_mu is not None:
                    log_items["post_mu_abs_mean"] = float(
                        outputs.post_mu.detach().abs().mean().cpu()
                    )
                if outputs.post_logvar is not None:
                    post_std = torch.exp(0.5 * outputs.post_logvar.detach())
                    log_items["post_std_mean"] = float(post_std.mean().cpu())
                if outputs.prior_logvar is not None:
                    prior_std = torch.exp(0.5 * outputs.prior_logvar.detach())
                    log_items["prior_std_mean"] = float(prior_std.mean().cpu())

                self.log(log_items)
                self._last_cvae_log_step = step
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

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs, temperature=0.0, sample_latent=False, return_dict=True)

        loss = getattr(outputs, "loss", None)
        if prediction_loss_only:
            return (loss, None, None)

        preds = outputs.path_xy
        labels = (inputs.get("labels_xy"), inputs.get("labels_mask"))

        def _to_cpu(x):
            if x is None:
                return None
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x

        preds = _to_cpu(preds)
        labels = tuple(_to_cpu(x) for x in labels)
        return (loss, preds, labels)


def create_compute_metrics_fn():
    """Return a compute_metrics function that reports masked XY MSE."""

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        if preds is None or labels is None:
            return {}

        if isinstance(labels, (tuple, list)) and len(labels) >= 1:
            labels_xy = labels[0]
            labels_mask = labels[1] if len(labels) > 1 else None
        else:
            labels_xy = labels
            labels_mask = None

        pred_xy = np.asarray(preds, dtype=np.float64)
        tgt_xy = np.asarray(labels_xy, dtype=np.float64)
        diff2 = np.sum((pred_xy - tgt_xy) ** 2, axis=-1)  # [N, P]

        if labels_mask is not None:
            m = np.asarray(labels_mask, dtype=np.float64)
            denom = float(np.sum(m)) if float(np.sum(m)) > 0 else 1.0
            mse = float(np.sum(diff2 * m) / denom)
        else:
            mse = float(np.mean(diff2))

        return {"xy_mse": mse}

    return compute_metrics
