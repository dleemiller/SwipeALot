"""Train a downstream text-to-path CVAE model using HuggingFace Trainer."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from rich.logging import RichHandler
from transformers import TrainingArguments

from swipealot.downstream.pathVAE import SwipeTextToPathCVAEConfig, SwipeTextToPathCVAEModel
from swipealot.downstream.pathVAE.collator import SwipeTextToPathCVAECollator
from swipealot.downstream.pathVAE.config import PathVAEConfig
from swipealot.downstream.pathVAE.trainer import (
    SwipeTextToPathCVAETrainer,
    create_compute_metrics_fn,
)
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
    parser = argparse.ArgumentParser(description="Train text-to-path CVAE (downstream fine-tune)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pathVAE/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--debug", action="store_true", help="Use small subset of data")
    args = parser.parse_args()

    cfg = PathVAEConfig.from_yaml(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split("/")[-1].replace(".yaml", "")
    run_name = f"pathvae_{config_name}_{timestamp}"

    base_output_dir = cfg.training.training_args.get("output_dir", "checkpoints/pathVAE")
    base_log_dir = cfg.training.training_args.get("logging_dir", "logs/pathVAE")
    output_dir = f"{base_output_dir}/{run_name}"
    log_dir = f"{base_log_dir}/{run_name}"

    logger.info(f"Loading config from: [cyan]{args.config}[/cyan]")
    logger.info(f"Run name: [yellow]{run_name}[/yellow]")
    logger.info(f"Logs: [blue]{log_dir}[/blue]")
    logger.info(f"Checkpoints: [blue]{output_dir}[/blue]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [green]{device}[/green]")

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

    collator = SwipeTextToPathCVAECollator(
        processor=processor,
        resample_mode=cfg.data.path_resample_mode,
        reverse_prob=cfg.data.reverse_prob,
    )

    logger.info("Creating model...")
    model_cfg = SwipeTextToPathCVAEConfig(
        encoder_config=None,
        decoder_n_layers=cfg.model.decoder_n_layers,
        decoder_n_heads=cfg.model.decoder_n_heads,
        decoder_d_ff=cfg.model.decoder_d_ff,
        dropout=cfg.model.dropout,
        out_dim=cfg.model.out_dim,
        sigma_min=cfg.model.sigma_min,
        target_eps=cfg.model.target_eps,
        path_loss_radial_weight=cfg.model.path_loss_radial_weight,
        path_sigma_target_min=cfg.model.path_sigma_target_min,
        path_sigma_target_max=cfg.model.path_sigma_target_max,
        uncertainty_reg_weight=cfg.model.uncertainty_reg_weight,
        latent_dim=cfg.model.latent_dim,
        latent_hidden_dim=cfg.model.latent_hidden_dim,
        kl_weight=cfg.model.kl_weight,
        smoothness_weight=cfg.model.smoothness_weight,
        smoothness_order=cfg.model.smoothness_order,
        speed_smoothness_weight=cfg.model.speed_smoothness_weight,
        spline_enabled=cfg.model.spline_enabled,
        spline_num_ctrl=cfg.model.spline_num_ctrl,
        spline_degree=cfg.model.spline_degree,
        spline_adaptive=cfg.model.spline_adaptive,
        spline_min_ctrl=cfg.model.spline_min_ctrl,
        spline_max_ctrl=cfg.model.spline_max_ctrl,
        spline_ctrl_per_char=cfg.model.spline_ctrl_per_char,
        path_encoder_n_layers=cfg.model.path_encoder_n_layers,
        path_encoder_n_heads=cfg.model.path_encoder_n_heads,
        path_encoder_d_ff=cfg.model.path_encoder_d_ff,
        path_encoder_dropout=cfg.model.path_encoder_dropout,
    )
    model = SwipeTextToPathCVAEModel.from_encoder_pretrained(
        cfg.model.encoder_path,
        config=model_cfg,
        freeze_encoder=bool(cfg.model.freeze_encoder),
    )
    model.to(device)

    training_args = dict(cfg.training.training_args)
    training_args["output_dir"] = output_dir
    training_args["logging_dir"] = log_dir
    training_args["run_name"] = run_name
    training_args.setdefault("remove_unused_columns", False)
    training_args.setdefault("save_safetensors", True)

    hf_args = TrainingArguments(**training_args)

    trainer = SwipeTextToPathCVAETrainer(
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
