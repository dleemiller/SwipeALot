"""Trainer utilities for downstream text→path fine-tuning."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class SwipeTextToPathTrainer(Trainer):
    """Trainer that exposes deterministic XY predictions for metrics."""

    def __init__(self, *, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self._last_ar_log_step = -1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_xy = inputs.get("labels_xy")
        labels_mask = inputs.get("labels_mask")
        ss_ratio = float(getattr(model.config, "scheduled_sampling_ratio", 0.0))
        ss_warmup = int(getattr(model.config, "scheduled_sampling_warmup_steps", 0))
        if model.training and ss_ratio > 0.0 and ss_warmup > 0:
            step = int(self.state.global_step)
            ss_ratio = min(ss_ratio, ss_ratio * step / float(ss_warmup))
        outputs = model(**inputs, temperature=0.0, return_dict=True)

        if model.training and ss_ratio > 0.0 and labels_xy is not None:
            pred_xy = outputs.path_xy.detach()
            if pred_xy is None:
                raise RuntimeError("Scheduled sampling requires model outputs.")
            prev_gt = labels_xy[:, :-1, :]
            prev_pred = pred_xy[:, :-1, :]
            use_pred = torch.rand(prev_gt.shape[:-1], device=prev_gt.device) < ss_ratio
            if labels_mask is not None:
                use_pred = use_pred & labels_mask[:, :-1].bool()
            use_pred = use_pred.unsqueeze(-1)
            mixed_prev = torch.where(use_pred, prev_pred, prev_gt)
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                path_coords=inputs.get("path_coords"),
                labels_xy=labels_xy,
                labels_mask=labels_mask,
                tgt_in_xy=mixed_prev,
                temperature=0.0,
                return_dict=True,
            )
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss.")

        if model.training:
            step = int(self.state.global_step)
            log_every = int(self.args.logging_steps) if int(self.args.logging_steps) > 0 else 1
            if step != self._last_ar_log_step and step % log_every == 0:
                log_items = {}
                if outputs.log_sigma is not None:
                    log_items["log_sigma_mean"] = float(outputs.log_sigma.detach().mean().cpu())
                if outputs.mu_logit is not None:
                    log_items["mu_logit_abs_mean"] = float(
                        outputs.mu_logit.detach().abs().mean().cpu()
                    )
                    path_mean = torch.sigmoid(outputs.mu_logit.detach())
                    log_items["path_mean_abs_step"] = float(
                        (path_mean[:, 1:, :] - path_mean[:, :-1, :]).abs().mean().cpu()
                    )
                if ss_ratio > 0.0:
                    log_items["scheduled_sampling_ratio"] = float(ss_ratio)
                if outputs.path_xy is not None and labels_xy is not None:
                    pred = outputs.path_xy.detach()
                    tgt = labels_xy.detach()
                    diff2 = ((pred - tgt) ** 2).sum(dim=-1)
                    if labels_mask is not None:
                        mask = labels_mask.to(dtype=diff2.dtype)
                        denom = mask.sum().clamp(min=1.0)
                        log_items["train_xy_mse"] = float((diff2 * mask).sum().cpu() / denom.cpu())
                    else:
                        log_items["train_xy_mse"] = float(diff2.mean().cpu())

                if log_items:
                    self.log(log_items)
                self._last_ar_log_step = step

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
            outputs = model(**inputs, temperature=0.0, return_dict=True)

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
