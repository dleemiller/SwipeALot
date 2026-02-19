"""Trainer for attention shaping self-distillation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import Trainer

from swipealot.downstream.attention_shaping.metrics import compute_attention_metrics

logger = logging.getLogger(__name__)


def create_compute_metrics_fn():
    """Return a compute_metrics function for attention shaping eval.

    prediction_step packs [pred(26x128) | target(26x128) | char_mask(26) | stats(8)]
    into predictions, flattened to [N, 6690]. We unpack and compute metrics here.
    """
    n_path = 128
    n_chars = 26
    pred_size = n_chars * n_path  # 3328
    target_size = n_chars * n_path  # 3328
    mask_size = n_chars  # 26
    stats_size = 8

    stat_keys = [
        "temporal_loss",
        "char_kl_loss",
        "valid_huber_loss",
        "length_loss",
        "pred_mean",
        "pred_std",
        "target_mean",
        "target_std",
    ]

    def compute_metrics(eval_pred):
        packed = eval_pred.predictions  # [N, 6691]
        if packed is None or len(packed) == 0:
            return {}

        n = packed.shape[0]
        idx = 0

        pred = packed[:, idx : idx + pred_size].reshape(n, n_chars, n_path)
        idx += pred_size
        target = packed[:, idx : idx + target_size].reshape(n, n_chars, n_path)
        idx += target_size
        char_mask = packed[:, idx : idx + mask_size]
        idx += mask_size
        stats = packed[:, idx : idx + stats_size]

        # Average the per-batch stats (loss components, pred/target summary)
        results = {k: float(stats[:, i].mean()) for i, k in enumerate(stat_keys)}

        # Compute attention quality metrics on full pred/target/char_mask
        attn_metrics = compute_attention_metrics(
            pred,
            target,
            char_mask.astype(bool),
        )
        results.update(attn_metrics)

        # Log sample metrics to console
        from rich.console import Console

        console = Console()
        console.print(
            f"[bold]Eval:[/bold] "
            f"kl=[cyan]{results.get('kl_mean', 0):.4f}[/cyan]  "
            f"peak_mae=[cyan]{results.get('peak_mae_mean', 0):.1f}[/cyan]  "
            f"top_k_iou=[cyan]{results.get('top_k_iou', 0):.3f}[/cyan]  "
            f"ordering_tau=[cyan]{results.get('ordering_tau', 0):.3f}[/cyan]  "
            f"char_auc=[cyan]{results.get('char_auc', 0):.3f}[/cyan]  "
            f"snr=[cyan]{results.get('snr_db', 0):.1f}[/cyan]dB"
        )

        return results

    return compute_metrics


class SwipeAttentionShapingTrainer(Trainer):
    """Trainer for attention shaping. Single LR — the head is tiny, we're training the encoder."""

    def __init__(self, *, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self._last_log_step = -1

    def _log_output_stats(self, outputs):
        """Build dict of stats from model outputs."""
        log_items = {}

        if outputs.temporal_loss is not None:
            log_items["temporal_loss"] = outputs.temporal_loss
        if outputs.char_kl_loss is not None:
            log_items["char_kl_loss"] = outputs.char_kl_loss
        if outputs.valid_huber_loss is not None:
            log_items["valid_huber_loss"] = outputs.valid_huber_loss
        if outputs.length_loss is not None:
            log_items["length_loss"] = outputs.length_loss

        if outputs.pred_attention is not None:
            pred = outputs.pred_attention.detach()
            log_items["pred_mean"] = float(pred.mean().cpu())
            log_items["pred_std"] = float(pred.std().cpu())

        if outputs.target_attention is not None:
            target = outputs.target_attention.detach()
            log_items["target_mean"] = float(target.mean().cpu())
            log_items["target_std"] = float(target.std().cpu())

        return log_items

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, return_dict=True)
        loss = outputs.loss

        if loss is None:
            raise RuntimeError("Model did not return a loss.")

        if model.training:
            step = int(self.state.global_step)
            log_every = int(self.args.logging_steps) if int(self.args.logging_steps) > 0 else 1
            if step != self._last_log_step and step % log_every == 0:
                log_items = {"total_loss": float(loss.detach().cpu())}
                log_items.update(self._log_output_stats(outputs))
                if log_items:
                    self.log(log_items)
                self._last_log_step = step

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        loss = outputs.loss
        if prediction_loss_only:
            return (loss, None, None)

        # Pack pred [B, 26, 128], target [B, 26, 128], char_mask [B, 26], stats [B, 8]
        # into a single flat tensor [B, 6690] for HF Trainer gather/concat
        pred = outputs.pred_attention.detach().cpu()  # [B, 26, 128]
        target = outputs.target_attention.detach().cpu()  # [B, 26, 128]
        batch = pred.shape[0]

        # Reconstruct char_mask from target (chars with any attention > 0)
        char_mask = (target.sum(dim=-1) > 1e-8).float()  # [B, 26]

        stats = self._log_output_stats(outputs)
        stat_vec = (
            torch.tensor(
                [
                    stats.get("temporal_loss", 0.0),
                    stats.get("char_kl_loss", 0.0),
                    stats.get("valid_huber_loss", 0.0),
                    stats.get("length_loss", 0.0),
                    stats.get("pred_mean", 0.0),
                    stats.get("pred_std", 0.0),
                    stats.get("target_mean", 0.0),
                    stats.get("target_std", 0.0),
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(batch, -1)
        )  # [B, 8]

        packed = torch.cat(
            [
                pred.reshape(batch, -1),  # [B, 3328]
                target.reshape(batch, -1),  # [B, 3328]
                char_mask,  # [B, 26]
                stat_vec,  # [B, 9]
            ],
            dim=1,
        )  # [B, 6691]

        return (loss, packed, packed)  # preds=labels=packed

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to keep teacher frozen."""
        model.teacher.eval()
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

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

        # Save student encoder separately for stage 1 loading
        try:
            student_dir = Path(output_dir) / "student_encoder"
            self.model.student.save_pretrained(
                str(student_dir),
                safe_serialization=self.args.save_safetensors,
            )
        except Exception as e:
            logger.warning(f"Failed to save student encoder: {e}")

        # Save projector + projector_norm weights for stage 1
        try:
            projector_state = {
                "projector": self.model.projector.state_dict(),
                "projector_norm": self.model.projector_norm.state_dict(),
            }
            torch.save(projector_state, f"{output_dir}/projector.pt")
        except Exception as e:
            logger.warning(f"Failed to save projector state: {e}")
