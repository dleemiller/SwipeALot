"""Evaluate a HuggingFace SwipeALot checkpoint via the customer-facing APIs.

This script intentionally loads the checkpoint using:
- `AutoModel.from_pretrained(..., trust_remote_code=True)`
- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`

so evaluation reflects how the model will be used after export.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from swipealot.text_utils import swipable_length, swipable_text


def _safe_token(tokenizer, token_id: int) -> str:
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        return str(token_id)


def _vocab_start_id(tokenizer) -> int | None:
    # SwipeTokenizer wraps CharacterTokenizer as `._tokenizer`.
    inner = getattr(tokenizer, "_tokenizer", None)
    special_tokens = getattr(inner, "special_tokens", None)
    if isinstance(special_tokens, list):
        return len(special_tokens)
    return None


@dataclass
class MaskedEvalResult:
    masked_token_accuracy: float
    masked_token_count: int
    masked_token_accuracy_vocab: float
    masked_token_count_vocab: int
    word_accuracy_reconstruct: float
    word_accuracy_argmax: float
    word_count: int


@dataclass
class LengthEvalResult:
    mae: float
    rmse: float
    acc_exact_rounded: float
    acc_within_1: float
    acc_within_2: float
    count: int


@dataclass
class MaskedPredictionMetrics:
    char_accuracy: float
    top3_accuracy: float
    word_accuracy: float
    n_chars: int
    n_words: int


@dataclass
class FullReconstructionMetrics:
    char_accuracy: float
    word_accuracy: float
    n_chars: int
    n_words: int


@dataclass
class BlindReconstructionMetrics:
    """Two-pass ('blind reconstruction'): predict length from path, then reconstruct masked text."""

    char_accuracy: float
    word_accuracy: float
    n_chars: int
    n_words: int


@dataclass
class PathReconstructionMetrics:
    masked_mse: float
    n_samples: int


def _iter_batches(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _iter_dataset_batches(dataset, batch_size: int) -> Iterable[dict[str, list]]:
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : start + batch_size]


def evaluate_masked_tokens(
    *,
    model,
    processor,
    dataset_items: list[dict],
    device: torch.device,
    batch_size: int,
    mask_mode: str,
    mask_prob: float,
    mask_eos: bool,
    seed: int,
    show_mismatches: int = 0,
    show_top_errors: int = 0,
) -> MaskedEvalResult:
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(getattr(tokenizer, "eos_token_id", -1))
    vocab_start_id = _vocab_start_id(tokenizer)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    masked_correct = 0
    masked_total = 0
    masked_correct_vocab = 0
    masked_total_vocab = 0
    word_correct = 0
    word_correct_argmax = 0
    word_total = 0

    mismatches: list[dict] = []
    error_pair_counts: dict[tuple[int, int], int] = {}
    true_token_counts: dict[int, int] = {}
    true_token_correct: dict[int, int] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_batches(dataset_items, batch_size),
            total=(len(dataset_items) + batch_size - 1) // batch_size,
            desc="Masked token eval",
        ):
            words = [ex["word"] for ex in batch]
            paths = [ex["data"] for ex in batch]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]  # [B, char_len]
            labels = input_ids.clone()

            maskable = input_ids.ne(pad_id)
            if not mask_eos and eos_id >= 0:
                maskable = maskable & input_ids.ne(eos_id)

            if mask_mode == "none":
                masked_positions = torch.zeros_like(maskable, dtype=torch.bool)
            elif mask_mode == "all":
                masked_positions = maskable
            elif mask_mode == "random":
                r = torch.rand(maskable.shape, generator=rng, device="cpu")
                masked_positions = (r < float(mask_prob)) & maskable.cpu()
                masked_positions = masked_positions.to(device)
            else:
                raise ValueError(f"Unknown mask_mode: {mask_mode}")

            labels[~masked_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_positions] = mask_id

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                labels=labels,
                return_dict=True,
            )
            logits = outputs.char_logits  # [B, char_len, V] (text segment only)
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)
            masked = labels.ne(-100)
            if masked.any():
                masked_correct += int((preds[masked] == labels[masked]).sum().item())
                masked_total += int(masked.sum().item())

                if vocab_start_id is not None:
                    masked_vocab = masked & labels.ge(int(vocab_start_id))
                    if masked_vocab.any():
                        masked_correct_vocab += int(
                            (preds[masked_vocab] == labels[masked_vocab]).sum().item()
                        )
                        masked_total_vocab += int(masked_vocab.sum().item())

                if show_top_errors > 0:
                    true_ids = labels[masked].detach().cpu().numpy().astype(np.int64)
                    pred_ids = preds[masked].detach().cpu().numpy().astype(np.int64)
                    for t, p in zip(true_ids.tolist(), pred_ids.tolist(), strict=False):
                        true_token_counts[t] = true_token_counts.get(t, 0) + 1
                        if p == t:
                            true_token_correct[t] = true_token_correct.get(t, 0) + 1
                        else:
                            key = (t, p)
                            error_pair_counts[key] = error_pair_counts.get(key, 0) + 1

            # Word accuracy (swipable): compare alphanumeric-only reconstruction.
            # - `reconstruct`: fill only masked positions from the model, keep unmasked ground truth.
            # - `argmax`: take the model argmax at every position (includes unmasked positions).
            preds_cpu = preds.detach().cpu().numpy()
            input_ids_cpu = input_ids.detach().cpu().numpy()
            masked_positions_cpu = masked_positions.detach().cpu().numpy()

            for i, target_word in enumerate(words):
                target_ids = input_ids_cpu[i].tolist()
                try:
                    eos_pos = target_ids.index(eos_id) if eos_id >= 0 else len(target_ids)
                except ValueError:
                    eos_pos = len(target_ids)

                target_swipable = swipable_text(target_word)

                # Reconstruct by only changing masked positions.
                recon_ids = target_ids[:eos_pos]
                pred_seq = preds_cpu[i][:eos_pos].tolist()
                mask_seq = masked_positions_cpu[i][:eos_pos].tolist()
                for pos, is_masked in enumerate(mask_seq):
                    if is_masked:
                        recon_ids[pos] = int(pred_seq[pos])

                pred_swipable = swipable_text(tokenizer.decode(recon_ids))
                if pred_swipable == target_swipable:
                    word_correct += 1

                pred_argmax_swipable = swipable_text(tokenizer.decode(pred_seq))
                if pred_argmax_swipable == target_swipable:
                    word_correct_argmax += 1

                word_total += 1

                if show_mismatches > 0 and pred_swipable != target_swipable:
                    diffs = []
                    for pos, is_masked in enumerate(mask_seq):
                        if not is_masked:
                            continue
                        true_id = int(target_ids[pos])
                        pred_id = int(pred_seq[pos])
                        if pred_id != true_id:
                            diffs.append(
                                {
                                    "pos": pos,
                                    "true": _safe_token(tokenizer, true_id),
                                    "pred": _safe_token(tokenizer, pred_id),
                                }
                            )

                    mismatches.append(
                        {
                            "target": target_word,
                            "target_swipable": target_swipable,
                            "pred_reconstruct_swipable": pred_swipable,
                            "pred_argmax_swipable": pred_argmax_swipable,
                            "n_masked": int(sum(mask_seq)),
                            "n_wrong_masked": len(diffs),
                            "wrong_masked": diffs,
                        }
                    )

    if show_mismatches > 0 and mismatches:
        mismatches.sort(key=lambda d: (d["n_wrong_masked"], d["n_masked"]), reverse=True)
        print("\n" + "=" * 72)
        print(f"Mismatches (showing up to {min(show_mismatches, len(mismatches))})")
        print("=" * 72)
        for ex in mismatches[:show_mismatches]:
            print(f"target:    {ex['target']!r}  (swipable={ex['target_swipable']!r})")
            print(f"pred(re):  {ex['pred_reconstruct_swipable']!r}")
            print(f"pred(arg): {ex['pred_argmax_swipable']!r}")
            print(
                f"masked: {ex['n_masked']}  wrong_masked: {ex['n_wrong_masked']}  details: {ex['wrong_masked']}"
            )
            print("-" * 72)

    if show_top_errors > 0 and error_pair_counts:
        top_pairs = sorted(error_pair_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :show_top_errors
        ]
        top_tokens = sorted(true_token_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :show_top_errors
        ]
        print("\n" + "=" * 72)
        print(f"Top masked-token errors (up to {len(top_pairs)})")
        print("=" * 72)
        for (true_id, pred_id), n in top_pairs:
            print(
                f"{n:6d}  true={_safe_token(tokenizer, true_id):>8s}  pred={_safe_token(tokenizer, pred_id):>8s}"
            )

        print("\n" + "=" * 72)
        print(f"Most-masked tokens + accuracy (up to {len(top_tokens)})")
        print("=" * 72)
        for tok_id, n in top_tokens:
            correct = true_token_correct.get(tok_id, 0)
            acc = correct / n if n else 0.0
            print(f"{n:6d}  acc={acc:6.3f}  token={_safe_token(tokenizer, tok_id)}")

    return MaskedEvalResult(
        masked_token_accuracy=(masked_correct / masked_total if masked_total else 0.0),
        masked_token_count=masked_total,
        masked_token_accuracy_vocab=(
            masked_correct_vocab / masked_total_vocab if masked_total_vocab else 0.0
        ),
        masked_token_count_vocab=masked_total_vocab,
        word_accuracy_reconstruct=(word_correct / word_total if word_total else 0.0),
        word_accuracy_argmax=(word_correct_argmax / word_total if word_total else 0.0),
        word_count=word_total,
    )


def evaluate_masked_prediction_30pct(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
    mask_prob: float,
    seed: int,
) -> MaskedPredictionMetrics:
    """Model-card metric: 30% random token masking; report char/top3/word accuracy."""
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    vocab_start_id = _vocab_start_id(tokenizer)
    if vocab_start_id is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    char_correct = 0
    char_top3 = 0
    char_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Masked prediction (30%)",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]  # [B, char_len]
            labels = input_ids.clone()

            # Randomly mask tokens (skip PAD; punctuation is a special token and will be excluded later).
            maskable = input_ids.ne(pad_id)
            r = torch.rand(maskable.shape, generator=rng, device="cpu")
            masked_positions = (r < float(mask_prob)) & maskable.cpu()
            masked_positions = masked_positions.to(device)

            # Evaluate only alphanumeric vocabulary tokens (a-z, 0-9).
            evaluable = labels.ge(int(vocab_start_id)) & maskable
            eval_mask = masked_positions & evaluable

            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_positions] = mask_id

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)

            if eval_mask.any():
                char_correct += int((preds[eval_mask] == labels[eval_mask]).sum().item())
                char_total += int(eval_mask.sum().item())

                top3 = logits.topk(k=3, dim=-1).indices  # [B, char_len, 3]
                hit_top3 = (top3 == labels.unsqueeze(-1)).any(dim=-1)
                char_top3 += int(hit_top3[eval_mask].sum().item())

            # Word accuracy: all masked alnum tokens correct (skip words with no masked-alnum positions).
            per_word_total = eval_mask.sum(dim=1)
            if per_word_total.any():
                per_word_correct = ((preds == labels) & eval_mask).sum(dim=1)
                word_mask = per_word_total > 0
                word_total += int(word_mask.sum().item())
                word_correct += int(
                    (per_word_correct[word_mask] == per_word_total[word_mask]).sum().item()
                )

    return MaskedPredictionMetrics(
        char_accuracy=(char_correct / char_total if char_total else 0.0),
        top3_accuracy=(char_top3 / char_total if char_total else 0.0),
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        n_chars=char_total,
        n_words=word_total,
    )


def evaluate_full_reconstruction_100pct(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
) -> FullReconstructionMetrics:
    """Model-card metric: 100% masked text reconstruction; report char/word accuracy."""
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    vocab_start_id = _vocab_start_id(tokenizer)
    if vocab_start_id is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Full reconstruction (100%)",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]  # [B, char_len]
            labels = input_ids.clone()

            maskable = input_ids.ne(pad_id)
            masked_input_ids = input_ids.clone()
            masked_input_ids[maskable] = mask_id  # include EOS, punctuation, etc.

            evaluable = labels.ge(int(vocab_start_id)) & maskable

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)

            if evaluable.any():
                char_correct += int((preds[evaluable] == labels[evaluable]).sum().item())
                char_total += int(evaluable.sum().item())

            per_word_total = evaluable.sum(dim=1)
            if per_word_total.any():
                per_word_correct = ((preds == labels) & evaluable).sum(dim=1)
                word_mask = per_word_total > 0
                word_total += int(word_mask.sum().item())
                word_correct += int(
                    (per_word_correct[word_mask] == per_word_total[word_mask]).sum().item()
                )

    return FullReconstructionMetrics(
        char_accuracy=(char_correct / char_total if char_total else 0.0),
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        n_chars=char_total,
        n_words=word_total,
    )


def evaluate_blind_reconstruction_two_pass(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
) -> BlindReconstructionMetrics:
    """Model-card metric: 2-pass reconstruction with unknown length ("blind reconstruction")."""
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(tokenizer.eos_token_id)

    vocab_start_id = _vocab_start_id(tokenizer)
    if vocab_start_id is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    max_path_len = int(processor.max_path_len)
    max_char_len = int(processor.max_char_len)
    char_start = 1 + max_path_len + 1  # [CLS] + path + [SEP]

    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Blind reconstruction (2-pass)",
        ):
            words = batch["word"]
            paths = batch["data"]

            # Target alnum-only tokens via text-only encoding.
            text_inputs = processor.encode_text(words, return_tensors="pt")
            text_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in text_inputs.items()
            }
            labels_full = text_inputs["input_ids"]  # [B, char_len]
            alnum_mask = (
                labels_full.ge(int(vocab_start_id))
                & labels_full.ne(pad_id)
                & labels_full.ne(eos_id)
            )

            # 1) Predict length from path only.
            path_inputs = processor.encode_path(paths, return_tensors="pt")
            path_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in path_inputs.items()
            }
            length_out = model(**path_inputs, return_dict=True)
            pred_len = length_out.length_logits.squeeze(-1)  # [B]
            pred_len_rounded = pred_len.round().long().clamp(min=0, max=max_char_len - 1)

            # 2) Create masked text segment of length `pred_len_rounded` (+ EOS).
            batch_size_actual = int(pred_len_rounded.shape[0])
            pos = torch.arange(max_char_len, device=device).unsqueeze(0)  # [1, C]

            input_ids = torch.full(
                (batch_size_actual, max_char_len), pad_id, dtype=torch.long, device=device
            )
            mask_positions = pos < pred_len_rounded.unsqueeze(1)
            eos_positions = pos == pred_len_rounded.unsqueeze(1)
            input_ids[mask_positions] = mask_id
            input_ids[eos_positions] = eos_id

            attention_mask = path_inputs["attention_mask"].clone()
            attention_mask[:, char_start:] = (pos < (pred_len_rounded + 1).unsqueeze(1)).long()

            outputs = model(
                path_coords=path_inputs["path_coords"],
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")
            preds = logits.argmax(dim=-1)  # [B, char_len]

            # Score per example with a length penalty (denom=max(true_len, pred_len)).
            for i in range(batch_size_actual):
                target_ids = labels_full[i][alnum_mask[i]].detach().cpu().tolist()
                pred_ids = preds[i, : int(pred_len_rounded[i].item())].detach().cpu().tolist()

                target_len = len(target_ids)
                pred_len_i = len(pred_ids)
                denom = max(target_len, pred_len_i)
                if denom == 0:
                    continue

                correct = 0
                for j in range(min(target_len, pred_len_i)):
                    if int(pred_ids[j]) == int(target_ids[j]):
                        correct += 1

                char_correct += correct
                char_total += denom

                word_total += 1
                if target_len == pred_len_i and correct == target_len:
                    word_correct += 1

    return BlindReconstructionMetrics(
        char_accuracy=(char_correct / char_total if char_total else 0.0),
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        n_chars=char_total,
        n_words=word_total,
    )


def evaluate_path_reconstruction_masked_mse(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
    mask_prob: float,
    mse_dims: list[int] | None,
    seed: int,
) -> PathReconstructionMetrics:
    """Model-card metric: 30% random path-point masking; report masked MSE."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    masked_mse_sum = 0.0
    masked_mse_count = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Path reconstruction (masked MSE)",
        ):
            words = batch["word"]
            paths = batch["data"]

            base = processor(path_coords=paths, text=words, return_tensors="pt")
            base = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in base.items()}

            path_coords = base["path_coords"]  # [B, max_path_len, D]
            max_path_len = path_coords.shape[1]

            attn = base.get("attention_mask")
            if attn is None:
                raise RuntimeError("Processor did not return attention_mask")
            path_mask = attn[:, 1 : 1 + max_path_len].bool()  # [B, max_path_len]

            r = torch.rand((path_coords.shape[0], max_path_len), generator=rng, device="cpu")
            mask = (r < float(mask_prob)) & path_mask.detach().cpu()
            mask = mask.to(device)

            if not mask.any():
                continue

            masked_path = path_coords.clone()
            masked_path[mask] = 0.0

            outputs = model(
                path_coords=masked_path,
                input_ids=base["input_ids"],
                attention_mask=base.get("attention_mask"),
                return_dict=True,
            )
            path_logits = getattr(outputs, "path_logits", None)
            if path_logits is None:
                raise RuntimeError("Model did not return `path_logits`; is predict_path disabled?")

            if path_logits.shape[1] == max_path_len:
                pred = path_logits
            else:
                pred = path_logits[:, 1 : 1 + max_path_len, :]

            if mse_dims:
                pred = pred[..., mse_dims]
                true = path_coords[..., mse_dims]
            else:
                true = path_coords

            diff2 = (true - pred) ** 2  # [B, max_path_len, D']
            for i in range(diff2.shape[0]):
                m = mask[i]
                if not bool(m.any()):
                    continue
                masked_mse_sum += float(diff2[i][m].mean().item())
                masked_mse_count += 1

    return PathReconstructionMetrics(
        masked_mse=(masked_mse_sum / masked_mse_count if masked_mse_count else 0.0),
        n_samples=masked_mse_count,
    )


def evaluate_length(
    *,
    model,
    processor,
    dataset_items: list[dict],
    device: torch.device,
    batch_size: int,
) -> LengthEvalResult:
    preds: list[float] = []
    targets: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_batches(dataset_items, batch_size),
            total=(len(dataset_items) + batch_size - 1) // batch_size,
            desc="Length eval",
        ):
            words = [ex["word"] for ex in batch]
            paths = [ex["data"] for ex in batch]

            inputs = processor(path_coords=paths, text=None, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }
            outputs = model(**inputs, return_dict=True)

            length_logits = getattr(outputs, "length_logits", None)
            if length_logits is None:
                raise RuntimeError(
                    "Model did not return `length_logits`; is predict_length disabled?"
                )

            length_pred = length_logits.reshape(-1).detach().cpu().numpy().astype(np.float64)
            preds.extend(length_pred.tolist())
            targets.extend([swipable_length(w) for w in words])

    pred_arr = np.array(preds, dtype=np.float64)
    tgt_arr = np.array(targets, dtype=np.float64)
    err = pred_arr - tgt_arr

    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0

    pred_round = np.round(pred_arr).clip(min=0.0)
    acc_exact = float(np.mean(pred_round == tgt_arr)) if len(err) else 0.0
    acc_w1 = float(np.mean(np.abs(pred_round - tgt_arr) <= 1.0)) if len(err) else 0.0
    acc_w2 = float(np.mean(np.abs(pred_round - tgt_arr) <= 2.0)) if len(err) else 0.0

    return LengthEvalResult(
        mae=mae,
        rmse=rmse,
        acc_exact_rounded=acc_exact,
        acc_within_1=acc_w1,
        acc_within_2=acc_w2,
        count=len(targets),
    )


def evaluate_length_dataset(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
) -> LengthEvalResult:
    preds: list[float] = []
    targets: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Length eval",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=None, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }
            outputs = model(**inputs, return_dict=True)

            length_logits = getattr(outputs, "length_logits", None)
            if length_logits is None:
                raise RuntimeError(
                    "Model did not return `length_logits`; is predict_length disabled?"
                )

            length_pred = length_logits.reshape(-1).detach().cpu().numpy().astype(np.float64)
            preds.extend(length_pred.tolist())
            targets.extend([swipable_length(w) for w in words])

    pred_arr = np.array(preds, dtype=np.float64)
    tgt_arr = np.array(targets, dtype=np.float64)
    err = pred_arr - tgt_arr

    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0

    pred_round = np.round(pred_arr).clip(min=0.0)
    acc_exact = float(np.mean(pred_round == tgt_arr)) if len(err) else 0.0
    acc_w1 = float(np.mean(np.abs(pred_round - tgt_arr) <= 1.0)) if len(err) else 0.0
    acc_w2 = float(np.mean(np.abs(pred_round - tgt_arr) <= 2.0)) if len(err) else 0.0

    return LengthEvalResult(
        mae=mae,
        rmse=rmse,
        acc_exact_rounded=acc_exact,
        acc_within_1=acc_w1,
        acc_within_2=acc_w2,
        count=len(targets),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SwipeALot HuggingFace checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a HuggingFace checkpoint directory (e.g. .../checkpoint-6000 or .../final)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="futo-org/swipe.futo.org",
        help="HuggingFace dataset name (default: futo-org/swipe.futo.org)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples to evaluate (0 = full split; default: 5000)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Optional HF cache root (sets HF_HOME / HF_DATASETS_CACHE); useful in restricted envs",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable HF offline mode (sets HF_HUB_OFFLINE / HF_DATASETS_OFFLINE / TRANSFORMERS_OFFLINE)",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["random", "all", "none"],
        default="random",
        help="Text masking mode (default: random)",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.3,
        help="Mask probability for masked-text metrics when --mask-mode=random (default: 0.3)",
    )
    parser.add_argument(
        "--path-mse-dims",
        type=str,
        default="0,1",
        help="Comma-separated path dims for masked MSE in --model-card (default: 0,1 for x/y)",
    )
    parser.add_argument(
        "--mask-eos",
        action="store_true",
        help="Also mask EOS tokens (default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=0,
        help="Print up to N word mismatches with masked character diffs (default: 0)",
    )
    parser.add_argument(
        "--show-top-errors",
        type=int,
        default=0,
        help="Print top N masked-token error pairs + most-masked tokens (default: 0)",
    )
    parser.add_argument(
        "--model-card",
        action="store_true",
        help="Print model-card style metrics table (masked pred, full recon, length, path MSE)",
    )
    parser.add_argument(
        "--skip-length",
        action="store_true",
        help="Skip length evaluation (default: False)",
    )

    args = parser.parse_args()

    # Configure HF caches before importing transformers (affects dynamic module cache path).
    hf_home = args.hf_home

    from swipealot.utils import configure_hf_env

    if hf_home is not None or args.offline:
        configure_hf_env(
            hf_home,
            offline=bool(args.offline),
            disable_telemetry=hf_home is not None,
            overwrite=False,
            set_hub_cache=False,
        )

    from datasets import load_dataset
    from transformers import AutoModel, AutoProcessor

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(str(ckpt), trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)

    if int(args.n_samples) <= 0:
        dataset = load_dataset(args.dataset, split=str(args.split))
    else:
        dataset = load_dataset(args.dataset, split=f"{args.split}[:{args.n_samples}]")

    if args.model_card:
        mse_dims: list[int] | None
        if args.path_mse_dims.strip():
            mse_dims = [int(x) for x in args.path_mse_dims.split(",") if x.strip() != ""]
        else:
            mse_dims = None

        masked30 = evaluate_masked_prediction_30pct(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            mask_prob=float(args.mask_prob),
            seed=args.seed,
        )
        full100 = evaluate_full_reconstruction_100pct(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        blind = evaluate_blind_reconstruction_two_pass(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        length = evaluate_length_dataset(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        path = evaluate_path_reconstruction_masked_mse(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            mask_prob=float(args.mask_prob),
            mse_dims=mse_dims,
            seed=args.seed,
        )

        print("\n## Performance Metrics\n")
        print(f"Evaluated on {len(dataset):,} {args.split} samples:\n")
        print("| Task | Metric | Score |")
        print("|------|--------|-------|")
        print(f"| Masked Prediction (30%) | Character Accuracy | {masked30.char_accuracy:.1%} |")
        print(f"|  | Top-3 Accuracy | {masked30.top3_accuracy:.1%} |")
        print(f"|  | Word Accuracy | {masked30.word_accuracy:.1%} |")
        print(f"| Full Reconstruction (100%) | Character Accuracy | {full100.char_accuracy:.1%} |")
        print(f"|  | Word Accuracy | {full100.word_accuracy:.1%} |")
        print(f"| Blind Reconstruction (2-pass) | Character Accuracy | {blind.char_accuracy:.1%} |")
        print(f"|  | Word Accuracy | {blind.word_accuracy:.1%} |")
        print(f"| Length Prediction | Exact Accuracy | {length.acc_exact_rounded:.1%} |")
        print(f"|  | Within ±1 | {length.acc_within_1:.1%} |")
        print(f"|  | Within ±2 | {length.acc_within_2:.1%} |")
        dim_label = (
            "x/y"
            if mse_dims == [0, 1]
            else ("all" if not mse_dims else ",".join(map(str, mse_dims)))
        )
        print(f"| Path Reconstruction | MSE (masked; dims={dim_label}) | {path.masked_mse:.6f} |")
        return

    dataset_items = list(dataset)

    masked = evaluate_masked_tokens(
        model=model,
        processor=processor,
        dataset_items=dataset_items,
        device=device,
        batch_size=args.batch_size,
        mask_mode=args.mask_mode,
        mask_prob=args.mask_prob,
        mask_eos=args.mask_eos,
        seed=args.seed,
        show_mismatches=args.show_mismatches,
        show_top_errors=args.show_top_errors,
    )

    print("\n" + "=" * 72)
    print("Masked token evaluation")
    print("=" * 72)
    print(
        f"masked_token_accuracy: {masked.masked_token_accuracy:.4f} (n={masked.masked_token_count})"
    )
    if masked.masked_token_count_vocab:
        print(
            f"masked_token_acc(vocab): {masked.masked_token_accuracy_vocab:.4f} (n={masked.masked_token_count_vocab})"
        )
    print(f"word_accuracy(recon):  {masked.word_accuracy_reconstruct:.4f} (n={masked.word_count})")
    print(f"word_accuracy(argmax): {masked.word_accuracy_argmax:.4f} (n={masked.word_count})")

    if not args.skip_length:
        length = evaluate_length(
            model=model,
            processor=processor,
            dataset_items=dataset_items,
            device=device,
            batch_size=args.batch_size,
        )
        print("\n" + "=" * 72)
        print("Length evaluation (regression)")
        print("=" * 72)
        print(f"mae:             {length.mae:.4f}")
        print(f"rmse:            {length.rmse:.4f}")
        print(f"acc_exact_round: {length.acc_exact_rounded:.4f}")
        print(f"acc_within_1:    {length.acc_within_1:.4f}")
        print(f"acc_within_2:    {length.acc_within_2:.4f}")


if __name__ == "__main__":
    main()
