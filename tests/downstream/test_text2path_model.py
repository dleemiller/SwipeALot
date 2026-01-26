import torch

from swipealot.downstream.text2path import SwipeTextToPathConfig, SwipeTextToPathModel
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


def test_text2path_forward_shapes_and_loss():
    enc_cfg = SwipeTransformerConfig(
        vocab_size=50,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_path_len=16,
        max_char_len=8,
        path_input_dim=6,
        predict_char=False,
        predict_path=False,
        predict_length=False,
    )
    cfg = SwipeTextToPathConfig(
        encoder_config=enc_cfg,
        decoder_n_layers=1,
        out_dim=2,
        sigma_min=1e-3,
    )
    model = SwipeTextToPathModel(cfg)
    model.eval()

    batch_size = 2
    input_ids = torch.randint(0, enc_cfg.vocab_size, (batch_size, enc_cfg.max_char_len))
    text_attention_mask = torch.ones(batch_size, enc_cfg.max_char_len, dtype=torch.long)

    labels_xy = torch.rand(batch_size, enc_cfg.max_path_len, 2).clamp(1e-3, 1 - 1e-3)
    labels_mask = torch.ones(batch_size, enc_cfg.max_path_len, dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=text_attention_mask,  # text-only mask should be accepted
            labels_xy=labels_xy,
            labels_mask=labels_mask,
            return_dict=True,
        )

    assert out.loss is not None
    assert out.mu_logit is not None and out.mu_logit.shape == (batch_size, enc_cfg.max_path_len, 2)
    assert out.log_sigma is not None and out.log_sigma.shape == (
        batch_size,
        enc_cfg.max_path_len,
        2,
    )
    assert out.path_xy is not None and out.path_xy.shape == (batch_size, enc_cfg.max_path_len, 2)
    assert out.encoder_last_hidden_state is not None
    assert out.encoder_last_hidden_state.shape[0] == batch_size

    with torch.no_grad():
        out_ss = model(
            input_ids=input_ids,
            attention_mask=text_attention_mask,
            labels_xy=labels_xy,
            labels_mask=labels_mask,
            tgt_in_xy=labels_xy[:, :-1, :],
            return_dict=True,
        )
    assert out_ss.loss is not None


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
    model = SwipeTextToPathModel(SwipeTextToPathConfig(encoder_config=enc_cfg, decoder_n_layers=1))
    model.eval()

    input_ids = torch.randint(0, enc_cfg.vocab_size, (1, enc_cfg.max_char_len))
    path = model.generate_path(input_ids=input_ids, attention_mask=None, temperature=0.7)
    assert path.shape == (1, enc_cfg.max_path_len, 2)
    assert float(path.min()) >= 0.0
    assert float(path.max()) <= 1.0
