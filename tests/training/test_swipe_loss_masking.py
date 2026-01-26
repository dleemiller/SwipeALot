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


def test_path_loss_dims_can_restrict_to_xy_only():
    torch.manual_seed(0)
    loss_fn = SwipeLoss(char_weight=0.0, path_weight=1.0, path_loss_dims=[0, 1])

    batch_size = 1
    path_len = 3
    dim = 6

    path_pred = torch.zeros(batch_size, path_len, dim, requires_grad=True)
    path_labels = torch.zeros(batch_size, path_len, dim)
    path_mask_indices = torch.zeros(batch_size, path_len, dtype=torch.long)

    # Only index 1 is masked/supervised.
    path_mask_indices[0, 1] = 1

    # Change only non-xy dims at the supervised point: should NOT affect xy-only loss.
    path_labels[0, 1, 2:] = torch.arange(dim - 2).float() + 10.0
    losses = loss_fn(
        {"path_logits": path_pred},
        {
            "path_coords": torch.zeros(batch_size, path_len, dim),
            "path_labels": path_labels,
            "path_mask_indices": path_mask_indices,
        },
    )
    assert losses["path_loss"].item() == 0.0

    # Now change x/y at the supervised point: should affect loss.
    path_labels[0, 1, 0] = 0.25
    path_labels[0, 1, 1] = 0.75
    losses2 = loss_fn(
        {"path_logits": path_pred},
        {
            "path_coords": torch.zeros(batch_size, path_len, dim),
            "path_labels": path_labels,
            "path_mask_indices": path_mask_indices,
        },
    )
    assert losses2["path_loss"].item() > 0.0


def test_path_loss_radial_weight_scales_with_distance():
    loss_fn = SwipeLoss(
        char_weight=0.0,
        path_weight=1.0,
        path_loss_dims=[0, 1],
        path_loss_radial_weight=1.0,
    )

    batch_size = 1
    path_len = 2
    dim = 6

    path_mask_indices = torch.ones(batch_size, path_len, dtype=torch.long)

    labels = torch.zeros(batch_size, path_len, dim)
    labels[0, 0, 0] = 0.5
    labels[0, 0, 1] = 0.5
    labels[0, 1, 0] = 1.0
    labels[0, 1, 1] = 1.0

    center_pred = labels.clone()
    center_pred[0, 0, 0] = center_pred[0, 0, 0] + 0.1
    center_loss = loss_fn(
        {"path_logits": center_pred},
        {
            "path_coords": torch.zeros(batch_size, path_len, dim),
            "path_labels": labels,
            "path_mask_indices": path_mask_indices,
        },
    )["path_loss"]

    corner_pred = labels.clone()
    corner_pred[0, 1, 0] = corner_pred[0, 1, 0] + 0.1
    corner_loss = loss_fn(
        {"path_logits": corner_pred},
        {
            "path_coords": torch.zeros(batch_size, path_len, dim),
            "path_labels": labels,
            "path_mask_indices": path_mask_indices,
        },
    )["path_loss"]

    assert corner_loss.item() > center_loss.item()


def test_path_loss_uses_log_sigma_when_provided():
    loss_fn = SwipeLoss(char_weight=0.0, path_weight=1.0, path_loss_dims=[0, 1])

    batch_size = 1
    path_len = 2
    dim = 6

    path_labels = torch.zeros(batch_size, path_len, dim)
    path_pred = path_labels.clone()
    path_mask_indices = torch.ones(batch_size, path_len, dtype=torch.long)

    batch = {
        "path_coords": torch.zeros(batch_size, path_len, dim),
        "path_labels": path_labels,
        "path_mask_indices": path_mask_indices,
    }

    log_sigma_zero = torch.zeros(batch_size, path_len, dim)
    loss_zero = loss_fn(
        {"path_logits": path_pred, "path_log_sigma": log_sigma_zero},
        batch,
    )["path_loss"]

    log_sigma_one = torch.ones(batch_size, path_len, dim)
    loss_one = loss_fn(
        {"path_logits": path_pred, "path_log_sigma": log_sigma_one},
        batch,
    )["path_loss"]

    assert loss_one.item() > loss_zero.item()
