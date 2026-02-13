#!/usr/bin/env python3
"""Extract distillation targets from a trained Phase 1 model.

Runs training data through the fine-tuned SwipeALot encoder + projector
and saves the projector outputs as NPZ shards for Phase 3 TCN training.

Output format per shard:
    projector_output: [N, D, 128] float16
    path_features: [N, 8, 128] float16
    words: [N] str
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.utils.data import DataLoader

from swipealot.downstream.distill import SwipeDistillModel
from swipealot.downstream.distill.collator import HFToWordDataset, SwipeDistillCollator
from swipealot.huggingface import SwipeProcessor, SwipeTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)


def extract_targets(
    model: SwipeDistillModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Run data through encoder + projector and collect outputs.

    Returns:
        (projector_outputs, path_features, words)
    """
    model.eval()
    all_projected = []
    all_path_features = []
    all_words = []

    with (
        torch.no_grad(),
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress,
    ):
        task = progress.add_task("Extracting targets...", total=len(dataloader))
        for batch in dataloader:
            # Move to device
            path_coords = batch["path_coords"].to(device)
            path_features = batch["path_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward through encoder + projector only
            outputs = model(
                path_coords=path_coords,
                path_features=path_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            # projected: [B, D, 128]
            projected = outputs.projected.cpu().numpy().astype(np.float16)
            pf = path_features.cpu().numpy().astype(np.float16)

            all_projected.append(projected)
            all_path_features.append(pf)

            # Words come directly from the collator batch
            all_words.extend(batch["words"])
            progress.update(task, advance=1)

    return all_projected, all_path_features, all_words


def save_shards(
    projected: list[np.ndarray],
    path_features: list[np.ndarray],
    words: list[str],
    output_dir: Path,
    shard_size: int = 50_000,
) -> None:
    """Save extraction results as NPZ shards."""
    # Concatenate all batches
    all_projected = np.concatenate(projected, axis=0)
    all_path_features = np.concatenate(path_features, axis=0)
    all_words = np.array(words, dtype=object)

    total = len(all_words)
    num_shards = math.ceil(total / shard_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, total)

        shard_path = output_dir / f"shard_{i:04d}.npz"
        np.savez_compressed(
            shard_path,
            projector_output=all_projected[start:end],
            path_features=all_path_features[start:end],
            words=all_words[start:end],
        )
        logger.info(f"  Saved shard {i + 1}/{num_shards}: {shard_path} ({end - start} samples)")

    logger.info(f"Total: {total} samples in {num_shards} shard(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract distillation targets from Phase 1 model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Phase 1 model checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for NPZ shards",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="futo-org/swipe.futo.org",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to extract from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=50_000,
        help="Samples per NPZ shard",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to extract (default: all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    logger.info(f"Loading model from: [cyan]{args.checkpoint}[/cyan]")

    # Load model
    model = SwipeDistillModel.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = model.to(args.device)
    model.eval()

    projector_dim = model.config.projector_dim
    logger.info(f"Projector dim: [yellow]{projector_dim}[/yellow]")

    # Load processor
    try:
        processor = SwipeProcessor.from_pretrained(args.checkpoint)
    except Exception:
        tokenizer = SwipeTokenizer.from_pretrained(args.checkpoint)
        processor = SwipeProcessor(
            tokenizer=tokenizer,
            max_path_len=128,
            max_char_len=48,
            path_input_dim=8,
        )

    # Load dataset
    logger.info(f"Loading dataset: [cyan]{args.dataset}[/cyan] split=[cyan]{args.split}[/cyan]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info(f"Samples: [green]{len(dataset):,}[/green]")

    wrapped = HFToWordDataset(dataset)
    collator = SwipeDistillCollator(processor=processor)

    dataloader = DataLoader(
        wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    # Extract
    logger.info("Extracting projector outputs...")
    projected, path_features, words = extract_targets(model, dataloader, torch.device(args.device))

    # Sanity check
    sample = projected[0]
    logger.info(f"Projector output shape: {sample.shape} (expected [B, {projector_dim}, 128])")
    all_proj = np.concatenate(projected, axis=0)
    logger.info(
        f"Projector stats: mean={all_proj.mean():.4f}, std={all_proj.std():.4f}, "
        f"norm={np.linalg.norm(all_proj, axis=1).mean():.4f}"
    )

    # Save
    output_dir = Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"distill_D{projector_dim}_{args.split}_{timestamp}"

    logger.info(f"Saving to: [blue]{output_dir}[/blue]")
    save_shards(projected, path_features, words, output_dir, args.shard_size)

    logger.info("[bold green]Extraction complete![/bold green]")


if __name__ == "__main__":
    main()
