"""Evaluate a distill checkpoint with beam search."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
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

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    label_lengths = np.concatenate(all_label_lengths)

    n = len(logits)
    logger.info(f"Evaluating [green]{n:,}[/green] samples...")

    # Beam search evaluation
    correct = 0
    top3 = 0
    oov = 0

    for i in tqdm(range(n), desc="Beam search"):
        ref_len = label_lengths[i]
        ref_word = _indices_to_word(labels[i, :ref_len].tolist())

        if not trie.search(ref_word):
            oov += 1
            continue

        sample_logits = torch.from_numpy(logits[i])

        candidates = beam_decoder.decode(sample_logits, top_k=3)
        if candidates and candidates[0][0] == ref_word:
            correct += 1
        if any(w == ref_word for w, _ in candidates[:3]):
            top3 += 1

    in_vocab = n - oov

    # Print results
    console.print()
    console.print(f"[bold]Results ({n:,} samples, {in_vocab:,} in-vocab, {oov:,} OOV)[/bold]")
    console.print(
        f"  word_acc=[cyan]{correct / max(in_vocab, 1):.4f}[/cyan]  "
        f"top3=[cyan]{top3 / max(in_vocab, 1):.4f}[/cyan]"
    )


if __name__ == "__main__":
    main()
