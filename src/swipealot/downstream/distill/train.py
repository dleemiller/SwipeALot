"""Train a downstream distillation model using HuggingFace Trainer."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rich.logging import RichHandler
from torch.utils.data import ConcatDataset, Dataset
from transformers import TrainingArguments

from swipealot.downstream.distill import SwipeDistillConfig, SwipeDistillModel
from swipealot.downstream.distill.collator import (
    HFToWordDataset,
    NPZDistillCollator,
    SwipeDistillCollator,
)
from swipealot.downstream.distill.config import DistillConfig
from swipealot.downstream.distill.trainer import SwipeDistillTrainer, create_compute_metrics_fn
from swipealot.huggingface import SwipeProcessor, SwipeTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)


class NPZWordDataset(Dataset):
    """Simple dataset wrapping NPZ files that have path_features + words."""

    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if path.is_dir():
            files = sorted(path.rglob("*.npz"))
        else:
            files = [path]

        all_features = []
        all_words = []
        for f in files:
            data = np.load(f, allow_pickle=True)
            all_features.append(data["path_features"])  # [N, 8, 128]
            all_words.append(data["words"])  # [N]

        self.path_features = np.concatenate(all_features, axis=0)
        self.words = np.concatenate(all_words, axis=0)
        logger.info(f"  NPZ: loaded {len(self)} samples from {len(files)} file(s) at {npz_path}")

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return {
            "path_features": torch.from_numpy(self.path_features[idx].astype(np.float32)),
            "word": str(self.words[idx]),
        }


def _maybe_slice(dataset, n: int | None):
    if n is None:
        return dataset
    if n <= 0:
        return dataset
    return dataset.select(range(min(int(n), len(dataset))))


class MixedCollator:
    """Dispatches to HF or NPZ collator based on item keys.

    Handles mixed batches from ConcatDataset where some items come from
    HF (with ``data``) and others from NPZ (with ``path_features``).
    """

    def __init__(self, *, hf_collator: SwipeDistillCollator, npz_collator: NPZDistillCollator):
        self.hf_collator = hf_collator
        self.npz_collator = npz_collator

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        hf_items = [it for it in items if "data" in it]
        npz_items = [it for it in items if "path_features" in it]

        batches = []
        if hf_items:
            batches.append(self.hf_collator(hf_items))
        if npz_items:
            batches.append(self.npz_collator(npz_items))

        if len(batches) == 1:
            return batches[0]

        merged = {}
        for key in batches[0]:
            vals = [b[key] for b in batches]
            if isinstance(vals[0], torch.Tensor):
                merged[key] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], list):
                merged[key] = [x for v in vals for x in v]
            else:
                merged[key] = vals[0]
        return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train distillation model (SwipeALot -> projector -> CTC)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/distill/dfsmn.yaml", help="Path to YAML config"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Use small subset of data")
    args = parser.parse_args()

    cfg = DistillConfig.from_yaml(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split("/")[-1].replace(".yaml", "")
    run_name = f"distill_{config_name}_{timestamp}"

    base_output_dir = cfg.training.training_args.get("output_dir", "checkpoints/distill")
    base_log_dir = cfg.training.training_args.get("logging_dir", "logs/distill")
    output_dir = f"{base_output_dir}/{run_name}"
    log_dir = f"{base_log_dir}/{run_name}"

    logger.info(f"Loading config from: [cyan]{args.config}[/cyan]")
    logger.info(f"Run name: [yellow]{run_name}[/yellow]")
    logger.info(f"Logs: [blue]{log_dir}[/blue]")
    logger.info(f"Checkpoints: [blue]{output_dir}[/blue]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [green]{device}[/green]")

    # Load tokenizer+processor
    processor_path = cfg.model.init_checkpoint or cfg.model.encoder_path
    logger.info(f"Loading processor from: [cyan]{processor_path}[/cyan]")
    try:
        processor = SwipeProcessor.from_pretrained(processor_path)
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = SwipeTokenizer.from_pretrained(processor_path)
        processor = SwipeProcessor(
            tokenizer=tokenizer,
            max_path_len=128,
            max_char_len=48,
            path_input_dim=8,
            path_resample_mode=str(cfg.data.path_resample_mode),
        )

    # Load datasets
    logger.info("Loading datasets...")
    max_samples = 2_000 if args.debug else None

    # HuggingFace dataset
    hf_train = load_dataset(
        cfg.data.dataset_name, cfg.data.dataset_config, split=cfg.data.train_split
    )
    hf_val = load_dataset(cfg.data.dataset_name, cfg.data.dataset_config, split=cfg.data.val_split)
    if cfg.data.exclude_sources:
        exclude = set(cfg.data.exclude_sources)
        hf_train = hf_train.filter(lambda r: r["source"] not in exclude)
        hf_val = hf_val.filter(lambda r: r["source"] not in exclude)
        logger.info(f"After excluding {exclude}: train={len(hf_train):,}, val={len(hf_val):,}")
    hf_train = _maybe_slice(hf_train, cfg.data.max_train_samples or max_samples)
    hf_val = _maybe_slice(
        hf_val, cfg.data.max_eval_samples or (max_samples // 10 if max_samples else None)
    )

    # Build combined training dataset
    train_datasets: list[Dataset] = [HFToWordDataset(hf_train)]
    collator = SwipeDistillCollator(processor=processor, resample_mode=cfg.data.path_resample_mode)

    # Add NPZ datasets if configured
    npz_datasets: list[Dataset] = []
    if cfg.data.extra_npz_paths:
        # For NPZ data, we need a separate collator
        for npz_path in cfg.data.extra_npz_paths:
            try:
                npz_ds = NPZWordDataset(npz_path)
                npz_datasets.append(npz_ds)
            except Exception as e:
                logger.warning(f"Failed to load NPZ data from {npz_path}: {e}")

    # If we have NPZ data, use ConcatDataset with the HF collator
    # (NPZ items also have {word, data/path_features} format)
    if npz_datasets:
        logger.info(f"Adding {len(npz_datasets)} NPZ dataset(s)")
        # NPZ datasets return path_features directly, need to handle both formats
        all_train = train_datasets + npz_datasets
        train_dataset = ConcatDataset(all_train)
        # Use a mixed collator that handles both formats
        collator = MixedCollator(
            hf_collator=SwipeDistillCollator(
                processor=processor, resample_mode=cfg.data.path_resample_mode
            ),
            npz_collator=NPZDistillCollator(processor=processor),
        )
    else:
        train_dataset = train_datasets[0]

    val_dataset = HFToWordDataset(hf_val)

    logger.info(f"Train samples: [green]{len(train_dataset):,}[/green]")
    logger.info(f"Val samples: [green]{len(val_dataset):,}[/green]")

    # For validation, always use HF collator (val is always HF data)
    val_collator = SwipeDistillCollator(
        processor=processor, resample_mode=cfg.data.path_resample_mode
    )

    # Create model
    logger.info("Creating model...")
    logger.info(f"  Projector dim: [yellow]{cfg.model.projector_dim}[/yellow]")
    logger.info(f"  Adapter stages: [yellow]{cfg.model.adapter_num_stages}[/yellow]")
    logger.info(
        f"  DFSMN: [yellow]layers={cfg.model.dfsmn_num_layers} "
        f"proj_dim={cfg.model.dfsmn_proj_dim} "
        f"context={cfg.model.dfsmn_context} "
        f"h={cfg.model.rnn_hidden}[/yellow]"
    )
    logger.info(f"  Text mask prob: [yellow]{cfg.model.text_mask_prob}[/yellow]")
    logger.info(f"  Encoder LR scale: [yellow]{cfg.model.encoder_lr_scale}[/yellow]")
    logger.info(f"  Freeze encoder: [yellow]{cfg.model.freeze_encoder}[/yellow]")

    model_cfg = SwipeDistillConfig(
        encoder_config=None,
        projector_dim=cfg.model.projector_dim,
        adapter_num_stages=cfg.model.adapter_num_stages,
        adapter_kernel_size=cfg.model.adapter_kernel_size,
        adapter_stride=cfg.model.adapter_stride,
        adapter_double_channels=cfg.model.adapter_double_channels,
        adapter_fold_last_stage=cfg.model.adapter_fold_last_stage,
        rnn_hidden=cfg.model.rnn_hidden,
        dfsmn_num_layers=cfg.model.dfsmn_num_layers,
        dfsmn_proj_dim=cfg.model.dfsmn_proj_dim,
        dfsmn_context=cfg.model.dfsmn_context,
        num_chars=cfg.model.num_chars,
        blank_idx=cfg.model.blank_idx,
        encoder_lr_scale=cfg.model.encoder_lr_scale,
        text_mask_prob=cfg.model.text_mask_prob,
    )
    if cfg.model.init_checkpoint:
        # Load full distill model from a previous run, override config fields
        logger.info(f"Loading distill model from: [cyan]{cfg.model.init_checkpoint}[/cyan]")
        init_cfg = SwipeDistillConfig.from_pretrained(cfg.model.init_checkpoint)
        init_cfg.decoder_type = cfg.model.decoder_type
        init_cfg.encoder_lr_scale = cfg.model.encoder_lr_scale
        model = SwipeDistillModel.from_pretrained(cfg.model.init_checkpoint, config=init_cfg)
        if cfg.model.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
    else:
        model = SwipeDistillModel.from_encoder_pretrained(
            cfg.model.encoder_path,
            config=model_cfg,
            freeze_encoder=bool(cfg.model.freeze_encoder),
        )

    # Load stage 0 (attention shaping) weights if configured
    if getattr(cfg.model, "stage0_checkpoint", None):
        stage0_path = Path(cfg.model.stage0_checkpoint)
        logger.info(f"Loading stage 0 weights from: [cyan]{stage0_path}[/cyan]")

        # Load student encoder weights
        student_encoder_path = stage0_path / "student_encoder"
        if student_encoder_path.exists():
            from swipealot.huggingface.modeling_swipe import SwipeTransformerModel

            stage0_encoder = SwipeTransformerModel.from_pretrained(str(student_encoder_path))
            model.encoder.load_state_dict(stage0_encoder.state_dict())
            del stage0_encoder
            logger.info("  Loaded student encoder weights")
        else:
            logger.warning(f"  Student encoder not found at {student_encoder_path}")

        # Load projector + projector_norm weights
        projector_path = stage0_path / "projector.pt"
        if projector_path.exists():
            projector_state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
            model.projector.load_state_dict(projector_state["projector"])
            model.projector_norm.load_state_dict(projector_state["projector_norm"])
            logger.info("  Loaded projector + projector_norm weights")
        else:
            logger.warning(f"  Projector weights not found at {projector_path}")

    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    new_params = sum(p.numel() for p in model.get_new_params())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Encoder params: [green]{encoder_params:,}[/green]")
    logger.info(f"  New params: [green]{new_params:,}[/green]")
    logger.info(f"  Trainable params: [green]{trainable_params:,}[/green]")

    # Training args
    training_args = dict(cfg.training.training_args)
    training_args["output_dir"] = output_dir
    training_args.pop("logging_dir", None)
    os.environ["TENSORBOARD_LOGGING_DIR"] = log_dir
    training_args["run_name"] = run_name
    training_args.setdefault("remove_unused_columns", False)

    hf_args = TrainingArguments(**training_args)

    # Load vocabulary trie for constrained beam search eval
    trie = None
    if cfg.data.vocab_path:
        from swipealot.decoder import Trie

        vocab_path = Path(cfg.data.vocab_path)
        if vocab_path.suffix == ".trie":
            trie = Trie.load(vocab_path)
        else:
            trie = Trie.from_file(vocab_path)
        logger.info(f"Loaded vocabulary trie: [green]{len(trie):,}[/green] words from {vocab_path}")

    # Create trainer
    trainer = SwipeDistillTrainer(
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        eval_collator=val_collator if npz_datasets else None,
        compute_metrics=create_compute_metrics_fn(
            blank_idx=cfg.model.blank_idx,
            trie=trie,
        ),
        processor=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    logger.info(f"Saved model to: [blue]{output_dir}[/blue]")
    logger.info(f"Adapter+decoder weights: [blue]{output_dir}/adapter_decoder.pt[/blue]")


if __name__ == "__main__":
    main()
