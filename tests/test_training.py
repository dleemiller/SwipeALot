"""Quick test training run to verify everything works."""

import torch
from torch.utils.data import DataLoader

from swipealot.data import CharacterTokenizer, MaskedCollator
from swipealot.huggingface import SwipeTransformerConfig, SwipeTransformerModel
from swipealot.training import SwipeLoss


def _make_sample(tokenizer: CharacterTokenizer, word: str, path_len: int, char_len: int):
    path_coords = torch.randn(path_len, 8)
    path_mask = torch.ones(path_len, dtype=torch.long)

    token_ids = tokenizer.encode(word) + [tokenizer.eos_token_id]
    token_ids = token_ids[: char_len - 1] + [tokenizer.eos_token_id]
    token_ids = token_ids + [tokenizer.pad_token_id] * (char_len - len(token_ids))

    char_mask = torch.tensor([1 if t != tokenizer.pad_token_id else 0 for t in token_ids])

    return {
        "path_coords": path_coords,
        "path_mask": path_mask,
        "char_tokens": torch.tensor(token_ids, dtype=torch.long),
        "char_mask": char_mask,
        "word": word,
    }


def test_training(num_steps: int = 3, batch_size: int = 2):
    """Run a minimal training smoke test on synthetic data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharacterTokenizer()
    config = SwipeTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_path_len=16,
        max_char_len=12,
        predict_path=True,
    )

    # Synthetic dataset
    words = ["hello", "world", "swipe", "keyboard", "test", "model"]
    samples = [
        _make_sample(tokenizer, w, path_len=config.max_path_len, char_len=config.max_char_len)
        for w in words
    ]
    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=1.0,  # ensure supervised tokens for the smoke test
        path_mask_prob=0.1,
        mask_path=True,
    )
    train_loader = DataLoader(samples, batch_size=batch_size, shuffle=True, collate_fn=collator)

    model = SwipeTransformerModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = SwipeLoss(char_weight=1.0, path_weight=0.1)

    model.train()
    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(
            path_coords=batch["path_coords"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        losses = loss_fn(outputs, batch)
        loss = losses["total_loss"]

        assert loss.requires_grad, "Loss should require gradients for backprop"
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        assert torch.isfinite(loss), "Loss should be finite"
        step += 1

    assert step == num_steps


if __name__ == "__main__":
    test_training()
