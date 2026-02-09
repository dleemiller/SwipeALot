import numpy as np
import torch

from swipealot.downstream.pathVAE import SwipeTextToPathCVAEConfig, SwipeTextToPathCVAEModel
from swipealot.downstream.pathVAE.modeling_pathvae import _bspline_basis_matrix
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


def _make_model(*, adaptive: bool) -> SwipeTextToPathCVAEModel:
    enc_cfg = SwipeTransformerConfig(
        vocab_size=50,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_path_len=16,
        max_char_len=10,
        path_input_dim=6,
        predict_char=False,
        predict_path=False,
        predict_length=False,
    )
    cfg = SwipeTextToPathCVAEConfig(
        encoder_config=enc_cfg,
        decoder_n_layers=1,
        out_dim=2,
        latent_dim=4,
        path_encoder_n_layers=1,
        spline_enabled=True,
        spline_num_ctrl=6,
        spline_degree=3,
        spline_adaptive=adaptive,
        spline_min_ctrl=4,
        spline_max_ctrl=10,
        spline_ctrl_per_char=0.5,
    )
    return SwipeTextToPathCVAEModel(cfg)


def test_bspline_basis_rows_sum_to_one():
    basis = _bspline_basis_matrix(num_points=16, num_ctrl=6, degree=3)
    assert basis.shape == (16, 6)
    assert np.all(basis >= 0.0)
    row_sums = basis.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_constant_control_points_stay_constant():
    basis = _bspline_basis_matrix(num_points=12, num_ctrl=5, degree=3)
    ctrl = np.full((5, 2), 0.42, dtype=np.float32)
    path = basis @ ctrl
    assert path.shape == (12, 2)
    assert np.allclose(path, 0.42, atol=1e-4)


def test_adaptive_num_ctrl_points_bounds():
    model = _make_model(adaptive=True)
    device = torch.device("cpu")
    short_ids = torch.randint(0, 10, (2, 10), device=device)
    short_mask = torch.zeros((2, 10), device=device, dtype=torch.long)
    short_mask[:, :2] = 1
    long_ids = torch.randint(0, 10, (2, 10), device=device)
    long_mask = torch.ones((2, 10), device=device, dtype=torch.long)

    num_short = model._num_ctrl_points(attention_mask=short_mask, input_ids=short_ids)
    num_long = model._num_ctrl_points(attention_mask=long_mask, input_ids=long_ids)

    assert 4 <= num_short <= 10
    assert 4 <= num_long <= 10
    assert num_long >= num_short


def test_spline_forward_backward_has_gradients():
    model = _make_model(adaptive=False)
    model.train()

    batch_size = 2
    input_ids = torch.randint(0, 50, (batch_size, 10))
    labels_xy = torch.rand(batch_size, 16, 2).clamp(1e-3, 1 - 1e-3)
    labels_mask = torch.ones(batch_size, 16, dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels_xy=labels_xy,
        labels_mask=labels_mask,
        return_dict=True,
    )
    assert out.loss is not None
    out.loss.backward()
    assert model.out_mu.weight.grad is not None
