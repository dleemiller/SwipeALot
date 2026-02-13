"""Main training script for swipe keyboard model using HuggingFace Trainer."""

import argparse
import logging
from datetime import datetime

import torch
from rich.logging import RichHandler
from transformers import TrainingArguments

from swipealot.config import Config
from swipealot.data import (
    CharacterTokenizer,
    MaskedCollator,
    NPZSwipeDataset,
    PairwiseMaskedCollator,
    SwipeDataset,
    ValidationCollator,
    vocab_hash,
)
from swipealot.huggingface import SwipeTransformerConfig, SwipeTransformerModel
from swipealot.training import SwipeLoss
from swipealot.training.trainer import SwipeTrainer, create_compute_metrics_fn

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train swipe keyboard model")
    parser.add_argument(
        "--config", type=str, default="configs/tiny.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use small subset of data for debugging"
    )
    parser.add_argument(
        "--char-weights",
        type=str,
        default=None,
        help="Path to precomputed character frequency weights (.pt)",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from: [cyan]{args.config}[/cyan]")
    config = Config.from_yaml(args.config)

    # Create unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split("/")[-1].replace(".yaml", "")
    run_name = f"{config_name}_{timestamp}"

    # Get base directories from training_args or use defaults
    base_output_dir = config.training.training_args.get("output_dir", "checkpoints")
    base_log_dir = config.training.training_args.get("logging_dir", "logs")

    # Update with unique run name
    output_dir = f"{base_output_dir}/{run_name}"
    log_dir = f"{base_log_dir}/{run_name}"

    logger.info(f"Run name: [yellow]{run_name}[/yellow]")
    logger.info(f"Logs: [blue]{log_dir}[/blue]")
    logger.info(f"Checkpoints: [blue]{output_dir}[/blue]")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [green]{device}[/green]")

    # Tokenizer is deterministic (not dataset-derived)
    logger.info("Creating tokenizer...")
    tokenizer = CharacterTokenizer()
    config.model.vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: [magenta]{tokenizer.vocab_size}[/magenta]")

    # Optional inverse-log character frequency weights (pre-computed only)
    char_freq_weights = None
    if config.training.use_char_freq_weights:
        weights_path = args.char_weights or config.training.char_weights_path
        if not weights_path:
            raise ValueError(
                "use_char_freq_weights is True but no char_weights_path provided. "
                "Run scripts/compute_char_weights.py first."
            )

        logger.info(f"Loading precomputed character weights from: [cyan]{weights_path}[/cyan]")
        loaded = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "weights" in loaded:
            char_freq_weights = loaded["weights"]

            # Optional vocab consistency check
            if "vocab" in loaded:
                saved_vocab = loaded["vocab"]
                current_vocab = [tokenizer.id_to_char[i] for i in range(tokenizer.vocab_size)]
                if saved_vocab != current_vocab:
                    raise ValueError(
                        "Tokenizer vocabulary does not match saved weight vocabulary; regenerate weights or align tokenizer"
                    )
            if "vocab_hash" in loaded:
                current_hash = vocab_hash(tokenizer)
                if loaded["vocab_hash"] != current_hash:
                    raise ValueError(
                        "Tokenizer hash mismatch: weights were built with a different tokenizer ordering"
                    )
        else:
            char_freq_weights = loaded

        if char_freq_weights.shape[0] != tokenizer.vocab_size:
            raise ValueError(
                f"Loaded weights vocab ({char_freq_weights.shape[0]}) != tokenizer vocab ({tokenizer.vocab_size})"
            )
        char_freq_weights = char_freq_weights.to(device)
        logger.info(
            f"Loaded weights tensor shape: {tuple(char_freq_weights.shape)} (mean {char_freq_weights.mean():.4f})"
        )
    else:
        logger.info(
            "Character frequency weights: [dim]disabled (use_char_freq_weights=False)[/dim]"
        )

    # Create datasets
    logger.info("Loading datasets...")
    max_samples = 1000 if args.debug else None

    train_dataset = SwipeDataset(
        split=config.data.train_split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=max_samples,
        path_resample_mode=config.data.path_resample_mode,
    )

    # Optionally concatenate extra NPZ datasets
    if config.data.extra_npz_paths:
        from torch.utils.data import ConcatDataset

        npz_datasets = []
        for npz_path in config.data.extra_npz_paths:
            npz_ds = NPZSwipeDataset(
                npz_paths=npz_path,
                tokenizer=tokenizer,
                max_word_len=config.data.max_char_len,
                max_samples=max_samples,
            )
            logger.info(
                f"  Extra NPZ dataset: [cyan]{npz_path}[/cyan] → [green]{len(npz_ds):,}[/green] samples"
            )
            npz_datasets.append(npz_ds)
        train_dataset = ConcatDataset([train_dataset] + npz_datasets)
        logger.info(f"Combined train dataset: [green]{len(train_dataset):,}[/green] samples")

    val_dataset = SwipeDataset(
        split=config.data.val_split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=max_samples // 10 if max_samples else 1000,  # Use 1k samples for validation
        path_resample_mode=config.data.path_resample_mode,
    )

    logger.info(f"Train samples: [green]{len(train_dataset):,}[/green]")
    logger.info(f"Val samples: [green]{len(val_dataset):,}[/green]")

    # Create collators
    logger.info("Creating data collators...")
    if config.training.use_pairwise_masking:
        train_collator = PairwiseMaskedCollator(
            tokenizer=tokenizer,
            mask_path=config.data.mask_path,
            modality_prob=config.training.pairwise_modality_prob,
            zero_attention_prob=config.training.pairwise_zero_attention_prob,
            path_mask_block_max_len=config.data.path_mask_block_max_len,
            inverted_char_prob_heavy=config.training.pairwise_inverted_char_prob_heavy,
            inverted_path_prob_heavy=config.training.pairwise_inverted_path_prob_heavy,
            inverted_char_prob_light=config.training.pairwise_inverted_char_prob_light,
            inverted_path_prob_light=config.training.pairwise_inverted_path_prob_light,
            right_half_prob=config.training.pairwise_right_half_prob,
            right_half_path_prob=config.training.pairwise_right_half_path_prob,
            right_half_reverse_prob=config.training.pairwise_right_half_reverse_prob,
            inverted_reverse_prob=config.training.pairwise_inverted_reverse_prob,
        )
        # Use unmasked validation for true accuracy metrics
        val_collator = ValidationCollator(tokenizer=tokenizer)
        modality_pct = int(config.training.pairwise_modality_prob * 100)
        inverted_pct = 100 - modality_pct
        logger.info(
            f"Using pairwise masking for training ([yellow]{inverted_pct}%[/yellow] inverted, [yellow]{modality_pct}%[/yellow] modality)"
        )
        logger.info("Using unmasked evaluation for validation")
        logger.info(
            "Note: data.char_mask_prob/path_mask_prob are ignored when use_pairwise_masking=True"
        )
    else:
        train_collator = MaskedCollator(
            tokenizer=tokenizer,
            char_mask_prob=config.data.char_mask_prob,
            path_mask_prob=config.data.path_mask_prob,
            mask_path=config.data.mask_path,
            mask_vocab_only=config.data.mask_vocab_only,
            path_mask_block_max_len=config.data.path_mask_block_max_len,
        )
        val_collator = train_collator
        logger.info("Using standard masked collator")

    # Create HuggingFace model configuration
    logger.info("Creating HuggingFace model configuration...")
    hf_config = SwipeTransformerConfig(
        vocab_size=config.model.vocab_size,
        max_path_len=config.data.max_path_len,
        max_char_len=config.data.max_char_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        path_input_dim=config.model.path_input_dim,
        predict_char=config.model.predict_char,
        predict_path=config.model.predict_path,
        predict_path_uncertainty=config.model.predict_path_uncertainty,
        predict_length=config.model.predict_length,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        unk_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Create model
    logger.info("Creating HuggingFace model...")
    model = SwipeTransformerModel(hf_config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [cyan]{num_params:,}[/cyan]")
    logger.info(f"Trainable parameters: [cyan]{num_trainable:,}[/cyan]")

    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = SwipeLoss(
        char_weight=config.training.char_loss_weight,
        path_weight=config.training.path_loss_weight,
        length_weight=config.training.length_loss_weight,
        path_loss_dims=config.training.path_loss_dims,
        path_loss_end_weight=config.training.path_loss_end_weight,
        path_loss_radial_weight=config.training.path_loss_radial_weight,
        path_sigma_min=config.training.path_sigma_min,
        path_best_of_k=config.training.path_best_of_k,
        path_sigma_target_min=config.training.path_sigma_target_min,
        path_sigma_target_max=config.training.path_sigma_target_max,
        uncertainty_reg_weight=config.training.uncertainty_reg_weight,
        focal_gamma=config.training.focal_gamma if config.training.use_focal_loss else 0.0,
        char_class_weights=char_freq_weights,
        contrastive_weight=config.training.contrastive_weight,
        contrastive_temperature=config.training.contrastive_temperature,
        matryoshka_dims=config.training.matryoshka_dims,
        matryoshka_weights=config.training.matryoshka_weights,
    )

    # Create compute_metrics function
    compute_metrics = create_compute_metrics_fn(tokenizer)

    # Create TrainingArguments from config
    logger.info("Creating training arguments...")

    # Start with training_args from config (this contains all standard HF parameters)
    training_args_dict = config.training.training_args.copy()

    # Override with run-specific settings
    training_args_dict["output_dir"] = output_dir
    training_args_dict["logging_dir"] = log_dir

    # Override num_workers if specified in data config
    if config.data.num_workers:
        training_args_dict.setdefault("dataloader_num_workers", config.data.num_workers)

    # Create TrainingArguments instance
    training_args = TrainingArguments(**training_args_dict)

    # Calculate total training steps for logging
    batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    total_train_steps = (len(train_dataset) // batch_size) * training_args.num_train_epochs
    warmup_steps = training_args_dict.get("warmup_steps", 0)
    warmup_ratio = (
        training_args.warmup_ratio
        if warmup_steps == 0
        else (warmup_steps / total_train_steps if total_train_steps > 0 else 0)
    )

    logger.info(f"Training for [yellow]{training_args.num_train_epochs}[/yellow] epochs")
    logger.info(f"Total training steps: ~[yellow]{total_train_steps:,}[/yellow]")
    logger.info(
        f"Warmup steps: [yellow]{int(warmup_ratio * total_train_steps)}[/yellow] ([dim]{warmup_ratio:.2%} of total[/dim])"
    )
    logger.info(f"Learning rate: [yellow]{training_args.learning_rate:.2e}[/yellow]")
    logger.info(f"Mixed precision: bf16={training_args.bf16}, fp16={training_args.fp16}")
    logger.info(f"Reporting to: [cyan]{', '.join(training_args.report_to)}[/cyan]")

    # Create Trainer
    logger.info("Initializing SwipeTrainer...")
    trainer = SwipeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        eval_collator=val_collator,  # Use separate collator for evaluation
        compute_metrics=compute_metrics,
        loss_fn=loss_fn,
        path_resample_mode=config.data.path_resample_mode,
    )

    # Resume from checkpoint if specified
    resume_from_checkpoint = None
    if args.resume:
        logger.info(f"Resuming from checkpoint: [cyan]{args.resume}[/cyan]")
        resume_from_checkpoint = args.resume

    # Train
    logger.info("[bold green]Starting training...[/bold green]")
    logger.info("=" * 60)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    final_path = output_dir + "/final"
    trainer.save_model(final_path)

    # Prepare checkpoint for HuggingFace Hub
    from swipealot.training.checkpoint_utils import prepare_checkpoint_for_hub

    logger.info("Preparing checkpoint for HuggingFace Hub...")
    prepare_checkpoint_for_hub(final_path)
    logger.info("  [green]✓[/green] Copied modeling files and dependencies")
    logger.info("  [green]✓[/green] Fixed imports for standalone loading")
    logger.info("  [green]✓[/green] Updated config.json with auto_map")

    # Save tokenizer and processor (auto_map is added automatically in save_pretrained)
    from swipealot.huggingface import SwipeProcessor, SwipeTokenizer

    hf_tokenizer = SwipeTokenizer()
    hf_tokenizer._tokenizer = tokenizer
    hf_tokenizer.save_pretrained(final_path)
    logger.info("  [green]✓[/green] Saved tokenizer with auto_map")

    hf_processor = SwipeProcessor(
        tokenizer=hf_tokenizer,
        max_path_len=config.data.max_path_len,
        max_char_len=config.data.max_char_len,
        path_input_dim=config.model.path_input_dim,
        path_resample_mode=config.data.path_resample_mode,
    )
    hf_processor.save_pretrained(final_path)
    logger.info("  [green]✓[/green] Saved processor with auto_map")

    logger.info("=" * 60)
    logger.info("[bold green]✅ Training complete![/bold green]")
    logger.info(f"[bold green]✅[/bold green] Model saved to: [cyan]{output_dir}/final[/cyan]")
    logger.info(
        "[bold green]✅[/bold green] Model is HuggingFace-compatible - no conversion needed!"
    )
    logger.info("\n[bold]To load the model:[/bold]")
    logger.info("  [dim]from transformers import AutoModel, AutoProcessor[/dim]")
    logger.info(
        f"  [dim]model = AutoModel.from_pretrained('{output_dir}/final', trust_remote_code=True)[/dim]"
    )
    logger.info(
        f"  [dim]processor = AutoProcessor.from_pretrained('{output_dir}/final', trust_remote_code=True)[/dim]"
    )


if __name__ == "__main__":
    main()
