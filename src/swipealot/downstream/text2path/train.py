"""Train a downstream text→path model using HuggingFace Trainer."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from rich.logging import RichHandler
from transformers import TrainingArguments

from swipealot.downstream.text2path import SwipeTextToPathConfig, SwipeTextToPathModel
from swipealot.downstream.text2path.collator import SwipeTextToPathCollator
from swipealot.downstream.text2path.config import Text2PathConfig
from swipealot.downstream.text2path.trainer import SwipeTextToPathTrainer, create_compute_metrics_fn
from swipealot.huggingface import SwipeProcessor, SwipeTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)


def _maybe_slice(dataset, n: int | None):
    if n is None:
        return dataset
    if n <= 0:
        return dataset
    return dataset.select(range(min(int(n), len(dataset))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text→path generator (downstream fine-tune)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/text2path/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--debug", action="store_true", help="Use small subset of data")
    args = parser.parse_args()

    cfg = Text2PathConfig.from_yaml(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split("/")[-1].replace(".yaml", "")
    run_name = f"text2path_{config_name}_{timestamp}"

    base_output_dir = cfg.training.training_args.get("output_dir", "checkpoints/text2path")
    base_log_dir = cfg.training.training_args.get("logging_dir", "logs/text2path")
    output_dir = f"{base_output_dir}/{run_name}"
    log_dir = f"{base_log_dir}/{run_name}"

    logger.info(f"Loading config from: [cyan]{args.config}[/cyan]")
    logger.info(f"Run name: [yellow]{run_name}[/yellow]")
    logger.info(f"Logs: [blue]{log_dir}[/blue]")
    logger.info(f"Checkpoints: [blue]{output_dir}[/blue]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [green]{device}[/green]")

    # Load tokenizer+processor from the encoder checkpoint for consistent text encoding.
    logger.info(f"Loading encoder tokenizer/processor from: [cyan]{cfg.model.encoder_path}[/cyan]")
    try:
        processor = SwipeProcessor.from_pretrained(cfg.model.encoder_path)
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = SwipeTokenizer.from_pretrained(cfg.model.encoder_path)
        processor = SwipeProcessor(
            tokenizer=tokenizer,
            max_path_len=128,
            max_char_len=48,
            path_input_dim=6,
            path_resample_mode=str(cfg.data.path_resample_mode),
        )

    logger.info("Loading datasets...")
    max_samples = 2_000 if args.debug else None
    train_ds = load_dataset(cfg.data.dataset_name, split=cfg.data.train_split)
    val_ds = load_dataset(cfg.data.dataset_name, split=cfg.data.val_split)
    train_ds = _maybe_slice(train_ds, cfg.data.max_train_samples or max_samples)
    val_ds = _maybe_slice(
        val_ds, cfg.data.max_eval_samples or (max_samples // 10 if max_samples else None)
    )

    logger.info(f"Train samples: [green]{len(train_ds):,}[/green]")
    logger.info(f"Val samples: [green]{len(val_ds):,}[/green]")

    collator = SwipeTextToPathCollator(
        processor=processor, resample_mode=cfg.data.path_resample_mode
    )

    logger.info("Creating model...")
    model_cfg = SwipeTextToPathConfig(
        encoder_config=None,
        decoder_n_layers=cfg.model.decoder_n_layers,
        decoder_n_heads=cfg.model.decoder_n_heads,
        decoder_d_ff=cfg.model.decoder_d_ff,
        dropout=cfg.model.dropout,
        out_dim=cfg.model.out_dim,
        sigma_min=cfg.model.sigma_min,
        target_eps=cfg.model.target_eps,
        scheduled_sampling_ratio=cfg.model.scheduled_sampling_ratio,
        scheduled_sampling_warmup_steps=cfg.model.scheduled_sampling_warmup_steps,
    )
    model = SwipeTextToPathModel.from_encoder_pretrained(
        cfg.model.encoder_path,
        config=model_cfg,
        freeze_encoder=bool(cfg.model.freeze_encoder),
    )

    training_args = dict(cfg.training.training_args)
    training_args["output_dir"] = output_dir
    # Always make run-specific logging dirs to avoid overwriting TensorBoard runs.
    training_args["logging_dir"] = log_dir
    training_args["run_name"] = run_name
    training_args.setdefault("remove_unused_columns", False)
    training_args.setdefault("save_safetensors", True)

    hf_args = TrainingArguments(**training_args)

    trainer = SwipeTextToPathTrainer(
        model=model,
        args=hf_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=create_compute_metrics_fn(),
        processor=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    logger.info(f"Saved model to: [blue]{output_dir}[/blue]")


if __name__ == "__main__":
    main()
