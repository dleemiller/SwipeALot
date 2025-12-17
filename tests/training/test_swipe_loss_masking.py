import torch

from swipealot.training.loss import SwipeLoss


def test_char_loss_uses_only_masked_tokens():
    torch.manual_seed(0)
    loss_fn = SwipeLoss(char_weight=1.0, path_weight=0.0)

    batch_size = 2
    path_len = 4
    char_len = 5
    vocab = 11

    outputs = {
        "char_logits": torch.randn(batch_size, char_len, vocab, requires_grad=True),
    }

    # Only 1 supervised token total; everything else ignored.
    char_labels = torch.full((batch_size, char_len), -100, dtype=torch.long)
    char_labels[0, 2] = 3

    batch = {
        "path_coords": torch.zeros(batch_size, path_len, 6),
        "char_labels": char_labels,
    }

    losses = loss_fn(outputs, batch)
    assert losses["char_loss"].isfinite()
    assert losses["char_loss"].item() > 0.0

    # Perturb logits at an ignored position; loss should not change.
    logits2 = outputs["char_logits"].detach().clone()
    logits2[1, 4] = logits2[1, 4] + 1000.0
    outputs2 = {"char_logits": logits2.requires_grad_(True)}
    losses2 = loss_fn(outputs2, batch)
    assert torch.allclose(losses["char_loss"], losses2["char_loss"])


def test_path_loss_uses_only_masked_points():
    torch.manual_seed(0)
    loss_fn = SwipeLoss(char_weight=0.0, path_weight=1.0)

    batch_size = 1
    path_len = 3
    dim = 6

    # Predict path segment only.
    path_pred = torch.zeros(batch_size, path_len, dim, requires_grad=True)
    path_labels = torch.zeros(batch_size, path_len, dim)
    path_mask_indices = torch.zeros(batch_size, path_len, dtype=torch.long)

    # Only index 1 is masked/supervised.
    path_mask_indices[0, 1] = 1
    path_labels[0, 1] = torch.arange(dim).float()

    # Make unmasked points wildly wrong; should not affect loss.
    path_pred.data[0, 0] = 999.0
    path_pred.data[0, 2] = -999.0

    outputs = {"path_logits": path_pred}
    batch = {
        "path_coords": torch.zeros(batch_size, path_len, dim),
        "path_labels": path_labels,
        "path_mask_indices": path_mask_indices,
    }

    losses = loss_fn(outputs, batch)
    assert losses["path_loss"].isfinite()
    assert losses["path_loss"].item() > 0.0

    # If nothing is masked, path loss should be 0.
    batch_none_masked = {
        **batch,
        "path_mask_indices": torch.zeros_like(path_mask_indices),
    }
    losses0 = loss_fn(outputs, batch_none_masked)
    assert losses0["path_loss"].item() == 0.0
