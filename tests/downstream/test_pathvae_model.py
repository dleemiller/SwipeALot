import torch

from swipealot.downstream.pathVAE import SwipeTextToPathCVAEConfig, SwipeTextToPathCVAEModel
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


def test_pathvae_forward_shapes_and_losses():
    enc_cfg = SwipeTransformerConfig(
        vocab_size=50,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_path_len=12,
        max_char_len=8,
        path_input_dim=6,
        predict_char=False,
        predict_path=False,
        predict_length=False,
    )
    cfg = SwipeTextToPathCVAEConfig(
        encoder_config=enc_cfg,
        decoder_n_layers=1,
        out_dim=2,
        sigma_min=1e-3,
        latent_dim=4,
        kl_weight=0.2,
        smoothness_weight=0.1,
        smoothness_order=2,
        speed_smoothness_weight=0.05,
        spline_enabled=True,
        spline_num_ctrl=6,
        spline_degree=3,
        path_encoder_n_layers=1,
    )
    model = SwipeTextToPathCVAEModel(cfg)
    model.eval()

    batch_size = 2
    input_ids = torch.randint(0, enc_cfg.vocab_size, (batch_size, enc_cfg.max_char_len))
    text_attention_mask = torch.ones(batch_size, enc_cfg.max_char_len, dtype=torch.long)

    labels_xy = torch.rand(batch_size, enc_cfg.max_path_len, 2).clamp(1e-3, 1 - 1e-3)
    labels_mask = torch.ones(batch_size, enc_cfg.max_path_len, dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=text_attention_mask,
            labels_xy=labels_xy,
            labels_mask=labels_mask,
            return_dict=True,
        )

    assert out.loss is not None
    assert out.recon_loss is not None
    assert out.kl_loss is not None
    assert out.smoothness_loss is not None
    assert out.speed_smoothness_loss is not None
    assert out.mu_logit is not None and out.mu_logit.shape == (batch_size, enc_cfg.max_path_len, 2)
    assert out.log_sigma is not None and out.log_sigma.shape == (
        batch_size,
        enc_cfg.max_path_len,
        2,
    )
    assert out.path_xy is not None and out.path_xy.shape == (batch_size, enc_cfg.max_path_len, 2)
    assert out.z is not None and out.z.shape == (batch_size, cfg.latent_dim)
    assert out.prior_mu is not None and out.prior_mu.shape == (batch_size, cfg.latent_dim)
    assert out.prior_logvar is not None and out.prior_logvar.shape == (batch_size, cfg.latent_dim)
    assert out.post_mu is not None and out.post_mu.shape == (batch_size, cfg.latent_dim)
    assert out.post_logvar is not None and out.post_logvar.shape == (batch_size, cfg.latent_dim)
    assert out.encoder_last_hidden_state is not None
    assert out.encoder_last_hidden_state.shape[0] == batch_size


def test_generate_path_returns_unit_interval():
    enc_cfg = SwipeTransformerConfig(
        vocab_size=30,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_path_len=8,
        max_char_len=6,
        path_input_dim=6,
        predict_char=False,
        predict_path=False,
        predict_length=False,
    )
    model = SwipeTextToPathCVAEModel(
        SwipeTextToPathCVAEConfig(
            encoder_config=enc_cfg,
            decoder_n_layers=1,
            latent_dim=3,
            path_encoder_n_layers=1,
        )
    )
    model.eval()

    input_ids = torch.randint(0, enc_cfg.vocab_size, (1, enc_cfg.max_char_len))
    path = model.generate_path(
        input_ids=input_ids,
        attention_mask=None,
        temperature=0.0,
        sample_latent=False,
    )
    assert path.shape == (1, enc_cfg.max_path_len, 2)
    assert float(path.min()) >= 0.0
    assert float(path.max()) <= 1.0
