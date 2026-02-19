"""Evaluate a distill checkpoint with beam search A/B (static vs length-pred)."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

from swipealot.decoder import Trie, TrieBeamSearch
from swipealot.downstream.distill import SwipeDistillConfig, SwipeDistillModel
from swipealot.downstream.distill.collator import HFToWordDataset, SwipeDistillCollator
from swipealot.huggingface import SwipeProcessor, SwipeTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)


def _indices_to_word(indices: list[int]) -> str:
    return "".join(chr(i + ord("a")) for i in indices)


def _greedy_decode(logits: np.ndarray, blank_idx: int = 26) -> list[int]:
    best = np.argmax(logits, axis=-1)  # [T]
    chars = []
    prev = -1
    for idx in best:
        if idx != prev and idx != blank_idx:
            chars.append(int(idx))
        prev = idx
    return chars


def main():
    parser = argparse.ArgumentParser(description="Evaluate distill checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--vocab", type=str, default="data/vocabulary.txt")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="futo-org/swipe.futo.org")
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    ckpt_path = Path(args.checkpoint)
    logger.info(f"Loading model from [cyan]{ckpt_path}[/cyan]")

    with open(ckpt_path / "config.json") as f:
        config_dict = json.load(f)
    config = SwipeDistillConfig(
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ("model_type", "transformers_version", "torch_dtype")
        }
    )
    model = SwipeDistillModel(config)

    # Load weights
    safetensors_file = ckpt_path / "model.safetensors"
    if safetensors_file.exists():
        state_dict = load_file(str(safetensors_file))
    else:
        state_dict = torch.load(ckpt_path / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    has_length = model.config.predict_length
    logger.info(f"predict_length: [yellow]{has_length}[/yellow]")

    # Load processor — bypass super().__init__ which crashes on transformers 5.0
    tokenizer = SwipeTokenizer.from_pretrained(str(ckpt_path), local_files_only=True)
    processor = SwipeProcessor.__new__(SwipeProcessor)
    processor.tokenizer = tokenizer
    processor.chat_template = None
    processor.max_path_len = 128
    processor.max_char_len = 48
    processor.path_input_dim = 8
    processor.path_resample_mode = "time"
    for attr in getattr(SwipeProcessor, "optional_attributes", []):
        if not hasattr(processor, attr):
            setattr(processor, attr, None)
    logger.info("Loaded processor")

    # Load trie
    vocab_path = Path(args.vocab)
    if vocab_path.suffix == ".trie":
        trie = Trie.load(vocab_path)
    else:
        trie = Trie.from_file(vocab_path)
    logger.info(f"Vocabulary: [green]{len(trie):,}[/green] words")

    beam_decoder = TrieBeamSearch(
        trie=trie,
        beam_width=args.beam_width,
        blank_idx=model.config.blank_idx,
        length_weight=args.length_weight,
    )

    # Load data
    logger.info(f"Loading {args.split} split...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.n_samples and args.n_samples < len(ds):
        ds = ds.select(range(args.n_samples))

    dataset = HFToWordDataset(ds)
    collator = SwipeDistillCollator(processor=processor, resample_mode="time")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=2
    )

    # Run eval
    all_logits = []
    all_labels = []
    all_label_lengths = []
    all_length_means = []
    all_length_sigmas = []

    logger.info("Running inference...")
    for batch in tqdm(loader, desc="Eval"):
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        with torch.no_grad():
            out = model(**batch_gpu, return_dict=True)

        all_logits.append(out.logits.cpu().numpy())
        all_labels.append(batch["labels"].numpy())
        all_label_lengths.append(batch["label_lengths"].numpy())

        if has_length and out.length_mean is not None:
            sigma_min = model.config.length_sigma_min
            sigma = torch.clamp(torch.exp(out.length_log_sigma), min=sigma_min)
            all_length_means.append(out.length_mean.cpu().numpy())
            all_length_sigmas.append(sigma.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    label_lengths = np.concatenate(all_label_lengths)
    length_means = np.concatenate(all_length_means) if all_length_means else None
    length_sigmas = np.concatenate(all_length_sigmas) if all_length_sigmas else None

    n = len(logits)
    logger.info(f"Evaluating [green]{n:,}[/green] samples...")

    # Beam search: static vs LP
    static_correct = 0
    lp_correct = 0
    static_top3 = 0
    lp_top3 = 0
    oov = 0

    # Track cases where LP and static disagree
    lp_wins = []  # LP correct, static wrong
    static_wins = []  # static correct, LP wrong

    for i in tqdm(range(n), desc="Beam search"):
        ref_len = label_lengths[i]
        ref_word = _indices_to_word(labels[i, :ref_len].tolist())

        if not trie.search(ref_word):
            oov += 1
            continue

        sample_logits = torch.from_numpy(logits[i])

        # Static
        cands_static = beam_decoder.decode(sample_logits, top_k=3)
        s_correct = cands_static and cands_static[0][0] == ref_word
        s_top3 = any(w == ref_word for w, _ in cands_static[:3])
        if s_correct:
            static_correct += 1
        if s_top3:
            static_top3 += 1

        # LP
        if length_means is not None:
            lp = (float(length_means[i]), float(length_sigmas[i]))
            cands_lp = beam_decoder.decode(sample_logits, top_k=3, length_pred=lp)
            l_correct = cands_lp and cands_lp[0][0] == ref_word
            l_top3 = any(w == ref_word for w, _ in cands_lp[:3])
            if l_correct:
                lp_correct += 1
            if l_top3:
                lp_top3 += 1

            # Track disagreements
            if l_correct and not s_correct and len(lp_wins) < 30:
                lp_wins.append(
                    (
                        ref_word,
                        cands_static[0][0] if cands_static else "???",
                        cands_lp[0][0] if cands_lp else "???",
                        float(length_means[i]),
                        float(length_sigmas[i]),
                    )
                )
            elif s_correct and not l_correct and len(static_wins) < 30:
                static_wins.append(
                    (
                        ref_word,
                        cands_static[0][0] if cands_static else "???",
                        cands_lp[0][0] if cands_lp else "???",
                        float(length_means[i]),
                        float(length_sigmas[i]),
                    )
                )

    in_vocab = n - oov

    # Print results
    console.print()
    console.print(f"[bold]Results ({n:,} samples, {in_vocab:,} in-vocab, {oov:,} OOV)[/bold]")
    console.print(
        f"  Static:  word_acc=[cyan]{static_correct / max(in_vocab, 1):.4f}[/cyan]  top3=[cyan]{static_top3 / max(in_vocab, 1):.4f}[/cyan]"
    )
    if length_means is not None:
        console.print(
            f"  LP:      word_acc=[cyan]{lp_correct / max(in_vocab, 1):.4f}[/cyan]  top3=[cyan]{lp_top3 / max(in_vocab, 1):.4f}[/cyan]"
        )
        delta = lp_correct - static_correct
        console.print(
            f"  Delta:   [{'green' if delta > 0 else 'red'}]{delta:+d}[/] samples ({delta / max(in_vocab, 1):+.4f})"
        )

    # Length prediction quality
    if length_means is not None:
        actual = label_lengths.astype(float)
        errors = length_means - actual
        abs_errors = np.abs(errors)
        sigmas = length_sigmas

        console.print()
        console.print("[bold]Length Prediction Quality[/bold]")
        console.print(f"  MAE:          [cyan]{abs_errors.mean():.3f}[/cyan]")
        console.print(f"  RMSE:         [cyan]{np.sqrt((errors**2).mean()):.3f}[/cyan]")
        console.print(f"  Mean sigma:   [cyan]{sigmas.mean():.3f}[/cyan]")
        console.print(f"  Within 1σ:    [cyan]{(abs_errors < sigmas).mean():.3f}[/cyan]")
        console.print(f"  Within 2σ:    [cyan]{(abs_errors < 2 * sigmas).mean():.3f}[/cyan]")
        console.print(f"  Correlation:  [cyan]{np.corrcoef(length_means, actual)[0, 1]:.4f}[/cyan]")

    # Show disagreement examples
    if lp_wins:
        console.print()
        table = Table(title=f"LP Wins ({len(lp_wins)} examples)", show_lines=False)
        table.add_column("Reference", style="green")
        table.add_column("Static Pick", style="red")
        table.add_column("LP Pick", style="cyan")
        table.add_column("Pred Len", justify="right")
        table.add_column("σ", justify="right")
        for ref, s_pick, l_pick, mean, sigma in lp_wins:
            table.add_row(ref, s_pick, l_pick, f"{mean:.1f}", f"{sigma:.2f}")
        console.print(table)

    if static_wins:
        console.print()
        table = Table(title=f"Static Wins ({len(static_wins)} examples)", show_lines=False)
        table.add_column("Reference", style="green")
        table.add_column("Static Pick", style="cyan")
        table.add_column("LP Pick", style="red")
        table.add_column("Pred Len", justify="right")
        table.add_column("σ", justify="right")
        for ref, s_pick, l_pick, mean, sigma in static_wins:
            table.add_row(ref, s_pick, l_pick, f"{mean:.1f}", f"{sigma:.2f}")
        console.print(table)


if __name__ == "__main__":
    main()
