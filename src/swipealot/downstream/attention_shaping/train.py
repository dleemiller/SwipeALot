"""Train attention shaping model (stage 0: self-distill encoder attention)."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from datasets import load_dataset
from rich.logging import RichHandler
from transformers import TrainingArguments

from swipealot.downstream.attention_shaping.config import AttentionShapingConfig
from swipealot.downstream.attention_shaping.configuration import SwipeAttentionShapingConfig
from swipealot.downstream.attention_shaping.modeling import SwipeAttentionShapingModel
from swipealot.downstream.attention_shaping.trainer import (
    SwipeAttentionShapingTrainer,
    create_compute_metrics_fn,
)
from swipealot.downstream.distill.collator import HFToWordDataset, SwipeDistillCollator
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
    parser = argparse.ArgumentParser(
        description="Train attention shaping model (stage 0: self-distill encoder attention)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/attention_shaping/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Use small subset of data")
    args = parser.parse_args()

    cfg = AttentionShapingConfig.from_yaml(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split("/")[-1].replace(".yaml", "")
    run_name = f"attn_shaping_{config_name}_{timestamp}"

    base_output_dir = cfg.training.training_args.get("output_dir", "checkpoints/attention_shaping")
    base_log_dir = cfg.training.training_args.get("logging_dir", "logs/attention_shaping")
    output_dir = f"{base_output_dir}/{run_name}"
    log_dir = f"{base_log_dir}/{run_name}"

    logger.info(f"Loading config from: [cyan]{args.config}[/cyan]")
    logger.info(f"Run name: [yellow]{run_name}[/yellow]")
    logger.info(f"Logs: [blue]{log_dir}[/blue]")
    logger.info(f"Checkpoints: [blue]{output_dir}[/blue]")

    # Load tokenizer+processor
    logger.info(f"Loading encoder from: [cyan]{cfg.model.encoder_path}[/cyan]")
    try:
        processor = SwipeProcessor.from_pretrained(cfg.model.encoder_path)
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = SwipeTokenizer.from_pretrained(cfg.model.encoder_path)
        processor = SwipeProcessor(
            tokenizer=tokenizer,
            max_path_len=128,
            max_char_len=48,
            path_input_dim=8,
            path_resample_mode=str(cfg.data.path_resample_mode),
        )

    # Load HF datasets only (teacher needs properly processed inputs)
    logger.info("Loading datasets...")
    max_samples = 2_000 if args.debug else None

    hf_train = load_dataset(cfg.data.dataset_name, split=cfg.data.train_split)
    hf_val = load_dataset(cfg.data.dataset_name, split=cfg.data.val_split)
    hf_train = _maybe_slice(hf_train, cfg.data.max_train_samples or max_samples)
    hf_val = _maybe_slice(
        hf_val, cfg.data.max_eval_samples or (max_samples // 10 if max_samples else None)
    )

    train_dataset = HFToWordDataset(hf_train)
    val_dataset = HFToWordDataset(hf_val)
    collator = SwipeDistillCollator(processor=processor, resample_mode=cfg.data.path_resample_mode)

    logger.info(f"Train samples: [green]{len(train_dataset):,}[/green]")
    logger.info(f"Val samples: [green]{len(val_dataset):,}[/green]")

    # Create model
    logger.info("Creating model...")
    logger.info(f"  Projector dim: [yellow]{cfg.model.projector_dim}[/yellow]")
    logger.info(f"  Target layers: [yellow]{cfg.model.attention_target_layers}[/yellow]")
    logger.info(f"  Softplus beta: [yellow]{cfg.model.softplus_beta}[/yellow]")
    logger.info(
        f"  Loss weights: huber_beta={cfg.model.huber_beta} "
        f"char_kl={cfg.model.char_kl_weight} valid_huber={cfg.model.valid_huber_weight} "
        f"length={cfg.model.length_loss_weight}"
    )

    model_cfg = SwipeAttentionShapingConfig(
        encoder_config=None,
        projector_dim=cfg.model.projector_dim,
        target_layers=cfg.model.attention_target_layers,
        softplus_beta=cfg.model.softplus_beta,
        huber_beta=cfg.model.huber_beta,
        char_kl_weight=cfg.model.char_kl_weight,
        valid_huber_weight=cfg.model.valid_huber_weight,
        length_loss_weight=cfg.model.length_loss_weight,
    )
    model = SwipeAttentionShapingModel.from_encoder_pretrained(
        cfg.model.encoder_path,
        config=model_cfg,
    )

    # Count parameters
    teacher_params = sum(p.numel() for p in model.teacher.parameters())
    student_params = sum(p.numel() for p in model.student.parameters())
    new_params = sum(p.numel() for p in model.get_new_params())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Teacher params (frozen): [green]{teacher_params:,}[/green]")
    logger.info(f"  Student params: [green]{student_params:,}[/green]")
    logger.info(f"  New params: [green]{new_params:,}[/green]")
    logger.info(f"  Trainable params: [green]{trainable_params:,}[/green]")

    # Training args
    training_args = dict(cfg.training.training_args)
    training_args["output_dir"] = output_dir
    training_args["logging_dir"] = log_dir
    training_args["run_name"] = run_name
    training_args.setdefault("remove_unused_columns", False)
    training_args.setdefault("save_safetensors", True)

    hf_args = TrainingArguments(**training_args)

    # Create trainer
    trainer = SwipeAttentionShapingTrainer(
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=create_compute_metrics_fn(),
        processor=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    logger.info(f"Saved model to: [blue]{output_dir}[/blue]")
    logger.info(f"Student encoder: [blue]{output_dir}/student_encoder/[/blue]")
    logger.info(f"Projector weights: [blue]{output_dir}/projector.pt[/blue]")


if __name__ == "__main__":
    main()
