# Cross Encoder Training Archive

This directory contains the original cross encoder training implementation for SwipeALot.

## Why Archived?

The cross encoder training code has been archived to streamline the main codebase to focus on:
- Base SwipeALot model training
- HuggingFace model conversion and distribution

Cross encoder training will be revisited in a dedicated repository that uses the HuggingFace base model ('dleemiller/SwipeALot-base') as a starting point with transformers trainers.

## What's Here?

### Core Training (src/ structure preserved)
- `models/cross_encoder.py` - SwipeCrossEncoderModel architecture
- `training/cross_encoder_trainer.py` - Custom trainer with freeze/unfreeze
- `training/loss.py` - MultipleNegativesRankingLoss + SwipeLoss (copied)
- `data/cross_encoder_dataset.py` - Dataset with negative mining
- `data/collators.py` - CrossEncoderCollator for batching (copied)

### HuggingFace Integration
- `huggingface/modeling_swipe.py` - HF-compatible model class (copied, contains both base and cross encoder)
- `huggingface/cross_encoder_wrapper.py` - Inference wrapper API

### Configuration
- `configs/cross_encoder.yaml` - Training hyperparameters
- Config classes remain in main codebase (src/swipealot/config.py)

### Tests
- `tests/test_cross_encoder_dataset.py`

## How It Worked

1. Train base SwipeTransformerModel first
2. Save checkpoint: `checkpoints/base_YYYYMMDD_HHMMSS/best_model.pt`
3. Load base checkpoint in SwipeCrossEncoderModel
4. Train classification head with MultipleNegativesRankingLoss
5. Use hard negative mining for contrastive learning

## Architecture Details

- **Reused from base**: Embeddings layer, encoder (transformer)
- **New component**: Classification head (Dense → GELU → LayerNorm → Linear)
- **Pooling**: SEP token embedding at position `1 + path_len`
- **Loss**: MNR loss with in-batch negatives + hard negatives

## Future Direction

Future cross encoder training will:
- Start from HuggingFace model: 'dleemiller/SwipeALot-base'
- Use transformers.Trainer with standard ecosystem
- Live in dedicated repository for cross encoder fine-tuning

## Archived
Date: $(date +%Y-%m-%d)
Commit: $(git rev-parse HEAD)
