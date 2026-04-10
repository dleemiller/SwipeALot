"""Trainer for distillation fine-tuning with CTC loss."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


def _greedy_ctc_decode(logits: np.ndarray, blank_idx: int = 26) -> list[list[int]]:
    """Greedy CTC decode: argmax, collapse repeats, remove blanks.

    Args:
        logits: [B, T, C] raw logits
        blank_idx: index of the CTC blank token

    Returns:
        List of decoded index sequences per sample.
    """
    best = np.argmax(logits, axis=-1)  # [B, T]
    decoded = []
    for seq in best:
        chars = []
        prev = -1
        for idx in seq:
            if idx != prev:
                if idx != blank_idx:
                    chars.append(int(idx))
            prev = idx
        decoded.append(chars)
    return decoded


def _indices_to_word(indices: list[int]) -> str:
    return "".join(chr(i + ord("a")) for i in indices)


def _edit_distance_ops(a: list[int], b: list[int]) -> tuple[int, int, int, int]:
    """Levenshtein distance with insertion/deletion/substitution counts.

    Returns:
        (distance, insertions, deletions, substitutions)
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb, lb, 0, 0
    if lb == 0:
        return la, 0, la, 0

    # DP table: each cell is (dist, ins, del, sub)
    prev = [(j, j, 0, 0) for j in range(lb + 1)]
    for i in range(1, la + 1):
        curr = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            # deletion from ref (hyp too short)
            d_del = prev[j][0] + 1
            # insertion into hyp (hyp too long)
            d_ins = curr[j - 1][0] + 1
            # substitution
            d_sub = prev[j - 1][0] + cost

            if d_del <= d_ins and d_del <= d_sub:
                curr[j] = (d_del, prev[j][1], prev[j][2] + 1, prev[j][3])
            elif d_ins <= d_sub:
                curr[j] = (d_ins, curr[j - 1][1] + 1, curr[j - 1][2], curr[j - 1][3])
            else:
                curr[j] = (d_sub, prev[j - 1][1], prev[j - 1][2], prev[j - 1][3] + cost)
        prev = curr
    return prev[lb]


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
                if outputs.ctc_loss is not None:
                    log_items["ctc_loss"] = float(outputs.ctc_loss.detach().cpu())
                else:
                    log_items["ctc_loss"] = float(loss.detach().cpu())

                if outputs.logits is not None:
                    logits = outputs.logits.detach()
                    blank_idx = int(self.model.config.blank_idx)
                    probs = torch.softmax(logits, dim=-1)

                    # Blank probability
                    log_items["blank_prob"] = float(probs[:, :, blank_idx].mean().cpu())

                    # Active frame rate (argmax != blank)
                    best = logits.argmax(dim=-1)  # [B, T]
                    log_items["active_frame_rate"] = float((best != blank_idx).float().mean().cpu())

                    # CTC entropy (bits) — mean per-frame entropy of softmax distribution
                    log_probs = torch.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1)  # [B, T] nats
                    log_items["ctc_entropy"] = float(
                        (entropy / 0.6931).mean().cpu()
                    )  # convert to bits

                    # Greedy decode length
                    shifted = torch.cat([torch.full_like(best[:, :1], -1), best[:, :-1]], dim=1)
                    non_repeat = best != shifted
                    non_blank = best != blank_idx
                    decoded_lens = (non_repeat & non_blank).sum(dim=1).float()
                    log_items["decoded_len"] = float(decoded_lens.mean().cpu())

                if log_items:
                    self.log(log_items)
                self._last_log_step = step

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to log per-group grad norms after backward."""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        step = int(self.state.global_step)
        log_every = int(self.args.logging_steps) if int(self.args.logging_steps) > 0 else 1
        if step % log_every == 0 and self.optimizer is not None:
            log_items = {}
            for group in self.optimizer.param_groups:
                name = group.get("name", "unknown")
                total_norm = 0.0
                for p in group["params"]:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                log_items[f"grad_norm/{name}"] = total_norm**0.5
            if log_items:
                self.log(log_items)

        return loss

    def _save(self, output_dir, state_dict=None):
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=True,
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

        preds = outputs.logits.detach().cpu()  # [B, T', num_chars+1]

        # Pack labels and label_lengths into a single tensor [B, max_label_len+1]
        # so HF Trainer can gather them across processes correctly.
        # Last column holds the length.
        labels = inputs["labels"].detach().cpu()  # [B, L]
        label_lengths = inputs["label_lengths"].detach().cpu()  # [B]
        packed_labels = torch.cat([labels, label_lengths.unsqueeze(1)], dim=1)

        return (loss, preds, packed_labels)


def create_compute_metrics_fn(
    blank_idx: int = 26,
    trie=None,
    beam_width: int = 10,
):
    """Return a compute_metrics function for CTC output with word accuracy and CER.

    Args:
        blank_idx: CTC blank token index.
        trie: Optional Trie for vocab-constrained beam search eval.
        beam_width: Beam width for trie-constrained decoding.
    """
    beam_decoder = None
    if trie is not None:
        from swipealot.decoder import TrieBeamSearch

        beam_decoder = TrieBeamSearch(
            trie=trie,
            beam_width=beam_width,
            blank_idx=blank_idx,
        )
        logger.info(f"Vocab-constrained eval enabled: {len(trie)} words, beam_width={beam_width}")

    def compute_metrics(eval_pred):
        raw_logits = eval_pred.predictions  # [N, T'(+1), C]
        packed_labels = eval_pred.label_ids  # [N, L+1]

        if raw_logits is None or packed_labels is None:
            return {}

        # Unpack: last column is length
        label_lengths = packed_labels[:, -1].astype(int)
        labels = packed_labels[:, :-1]

        logits = raw_logits

        decoded = _greedy_ctc_decode(logits, blank_idx=blank_idx)

        n = len(decoded)
        exact_matches = 0
        total_edit_dist = 0
        total_ins = 0
        total_del = 0
        total_sub = 0
        total_ref_len = 0
        wrong_examples = []

        for i in range(n):
            ref_len = label_lengths[i]
            ref = labels[i, :ref_len].tolist()
            hyp = decoded[i]

            if hyp == ref:
                exact_matches += 1
            elif len(wrong_examples) < 20:
                wrong_examples.append((_indices_to_word(ref), _indices_to_word(hyp)))

            dist, ins, dels, subs = _edit_distance_ops(hyp, ref)
            total_edit_dist += dist
            total_ins += ins
            total_del += dels
            total_sub += subs
            total_ref_len += max(ref_len, 1)

        word_acc = exact_matches / max(n, 1)
        cer = total_edit_dist / max(total_ref_len, 1)

        # Blank / active frame stats
        best = np.argmax(logits, axis=-1)  # [N, T']
        blank_rate = float((best == blank_idx).mean())
        active_frame_rate = 1.0 - blank_rate

        # CTC entropy (bits)
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True) + 1e-12)
        probs = np.exp(log_probs)
        entropy_nats = -(probs * log_probs).sum(axis=-1)  # [N, T']
        ctc_entropy = float(entropy_nats.mean() / 0.6931)

        # Decoded vs reference lengths
        mean_decoded_len = sum(len(d) for d in decoded) / max(n, 1)
        mean_ref_len = float(label_lengths.mean())

        # Vocab-constrained beam search decoding
        vocab_metrics = {}
        if beam_decoder is not None:
            import torch as _torch

            vocab_correct = 0
            vocab_top3 = 0
            oov_words = 0

            for i in range(n):
                ref_len = label_lengths[i]
                ref_word = _indices_to_word(labels[i, :ref_len].tolist())

                if not beam_decoder.trie.search(ref_word):
                    oov_words += 1
                    continue

                # beam_decoder.decode expects [T, C] (time-first)
                sample_logits = _torch.from_numpy(logits[i])  # [T', C]

                candidates = beam_decoder.decode(sample_logits, top_k=3)
                if candidates and candidates[0][0] == ref_word:
                    vocab_correct += 1
                if any(w == ref_word for w, _ in candidates[:3]):
                    vocab_top3 += 1

            in_vocab = n - oov_words
            vocab_metrics["vocab_word_acc"] = vocab_correct / max(in_vocab, 1)
            vocab_metrics["vocab_top3_acc"] = vocab_top3 / max(in_vocab, 1)
            vocab_metrics["oov_rate"] = oov_words / max(n, 1)

        # Log sample errors to console with rich formatting
        if wrong_examples:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Sample Eval Errors", show_lines=False)
            table.add_column("Reference", style="green")
            table.add_column("Predicted", style="red")
            table.add_column("Edit Dist", justify="right", style="yellow")
            for ref_word, hyp_word in wrong_examples:
                dist, _, _, _ = _edit_distance_ops(
                    [ord(c) - ord("a") for c in ref_word],
                    [ord(c) - ord("a") for c in hyp_word],
                )
                table.add_row(ref_word, hyp_word, str(dist))
            console.print(table)

            vocab_str = ""
            if vocab_metrics:
                vocab_str = (
                    f"  vocab_acc=[cyan]{vocab_metrics['vocab_word_acc']:.3f}[/cyan]"
                    f"  vocab_top3=[cyan]{vocab_metrics['vocab_top3_acc']:.3f}[/cyan]"
                    f"  oov=[dim]{vocab_metrics['oov_rate']:.3f}[/dim]"
                )
            console.print(
                f"[bold]Eval:[/bold] word_acc=[cyan]{word_acc:.3f}[/cyan]  "
                f"cer=[cyan]{cer:.3f}[/cyan]  "
                f"ins=[yellow]{total_ins / max(total_ref_len, 1):.3f}[/yellow]  "
                f"del=[yellow]{total_del / max(total_ref_len, 1):.3f}[/yellow]  "
                f"sub=[yellow]{total_sub / max(total_ref_len, 1):.3f}[/yellow]  "
                f"blank=[dim]{blank_rate:.3f}[/dim]  "
                f"entropy=[dim]{ctc_entropy:.2f}b[/dim]  "
                f"len=[dim]{mean_decoded_len:.1f}/{mean_ref_len:.1f}[/dim]" + vocab_str
            )

        metrics = {
            "word_acc": word_acc,
            "cer": cer,
            "ins_rate": total_ins / max(total_ref_len, 1),
            "del_rate": total_del / max(total_ref_len, 1),
            "sub_rate": total_sub / max(total_ref_len, 1),
            "blank_rate": blank_rate,
            "active_frame_rate": active_frame_rate,
            "ctc_entropy": ctc_entropy,
            "decoded_len": mean_decoded_len,
            "ref_len": mean_ref_len,
        }
        metrics.update(vocab_metrics)
        return metrics

    return compute_metrics
