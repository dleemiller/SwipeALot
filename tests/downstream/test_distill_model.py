import torch
import torch.nn as nn

from swipealot.downstream.distill import SwipeDistillConfig, SwipeDistillModel
from swipealot.downstream.distill.collator import _encode_ctc_batch, _word_to_ctc_target
from swipealot.downstream.distill.modeling_distill import CTCDecoder, TemporalAdapter
from swipealot.huggingface.configuration_swipe import SwipeTransformerConfig


def _small_enc_cfg():
    return SwipeTransformerConfig(
        vocab_size=50,
        d_model=64,
        n_layers=1,
        n_heads=2,
        d_ff=128,
        max_path_len=16,
        max_char_len=8,
        path_input_dim=6,
        predict_char=False,
        predict_path=False,
        predict_length=False,
    )


def _small_distill_cfg(enc_cfg=None):
    enc_cfg = enc_cfg or _small_enc_cfg()
    return SwipeDistillConfig(
        encoder_config=enc_cfg,
        projector_dim=32,
        adapter_num_stages=2,
        adapter_kernel_size=3,
        adapter_stride=2,
        rnn_hidden=32,
        rnn_layers=1,
        rnn_bidirectional=True,
        rnn_type="lstm",
        rnn_dropout=0.0,
        num_chars=26,
        blank_idx=26,
    )


# ── Model tests ──────────────────────────────────────────────────


def test_distill_forward_shapes_and_loss():
    cfg = _small_distill_cfg()
    model = SwipeDistillModel(cfg)
    model.eval()

    batch_size = 2
    enc_cfg = _small_enc_cfg()
    path_coords = torch.randn(batch_size, enc_cfg.max_path_len, enc_cfg.path_input_dim)
    input_ids = torch.randint(0, enc_cfg.vocab_size, (batch_size, enc_cfg.max_char_len))
    attention_mask = torch.ones(batch_size, enc_cfg.max_char_len, dtype=torch.long)
    labels = torch.randint(0, 26, (batch_size, 5))
    label_lengths = torch.tensor([5, 3])

    with torch.no_grad():
        out = model(
            path_coords=path_coords,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            label_lengths=label_lengths,
            return_dict=True,
        )

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape[0] == batch_size
    assert out.logits.shape[2] == 27  # 26 chars + blank
    assert out.projected is not None
    assert out.projected.shape == (batch_size, cfg.projector_dim, enc_cfg.max_path_len)
    assert out.encoder_last_hidden_state is not None


def test_distill_forward_backward():
    cfg = _small_distill_cfg()
    model = SwipeDistillModel(cfg)
    model.train()

    batch_size = 2
    enc_cfg = _small_enc_cfg()
    path_coords = torch.randn(batch_size, enc_cfg.max_path_len, enc_cfg.path_input_dim)
    input_ids = torch.randint(0, enc_cfg.vocab_size, (batch_size, enc_cfg.max_char_len))
    labels = torch.randint(0, 26, (batch_size, 5))
    label_lengths = torch.tensor([5, 3])

    out = model(
        path_coords=path_coords,
        input_ids=input_ids,
        labels=labels,
        label_lengths=label_lengths,
        return_dict=True,
    )

    out.loss.backward()

    # Verify gradients flow to all named parameter groups
    has_encoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()
    )
    has_projector_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters()
    )
    has_adapter_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal_adapter.parameters()
    )
    has_decoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.ctc_decoder.parameters()
    )

    assert has_encoder_grad, "No gradients in encoder"
    assert has_projector_grad, "No gradients in projector"
    assert has_adapter_grad, "No gradients in temporal adapter"
    assert has_decoder_grad, "No gradients in CTC decoder"


def test_distill_param_groups():
    cfg = _small_distill_cfg()
    model = SwipeDistillModel(cfg)

    encoder_ids = {id(p) for p in model.get_encoder_params()}
    new_ids = {id(p) for p in model.get_new_params()}

    # Disjoint
    assert encoder_ids & new_ids == set(), "Encoder and new params overlap"

    # Cover all params
    all_ids = {id(p) for p in model.parameters()}
    assert encoder_ids | new_ids == all_ids, "Some params not covered by groups"


def test_distill_forward_absorbs_extra_kwargs():
    """Model forward absorbs extra collator keys (path_features, words) via **kwargs."""
    cfg = _small_distill_cfg()
    model = SwipeDistillModel(cfg)
    model.eval()

    batch_size = 2
    enc_cfg = _small_enc_cfg()
    path_coords = torch.randn(batch_size, enc_cfg.max_path_len, enc_cfg.path_input_dim)
    input_ids = torch.randint(0, enc_cfg.vocab_size, (batch_size, enc_cfg.max_char_len))

    with torch.no_grad():
        out = model(
            path_coords=path_coords,
            input_ids=input_ids,
            path_features=torch.randn(batch_size, enc_cfg.path_input_dim, enc_cfg.max_path_len),
            words=["hello", "world"],
            return_dict=True,
        )

    assert out.logits is not None


# ── Component tests ──────────────────────────────────────────────


def test_temporal_adapter_shapes():
    adapter = TemporalAdapter(input_channels=128, num_stages=2, kernel_size=5, stride=2)
    x = torch.randn(4, 128, 128)  # [B, C, T]
    y = adapter(x)
    assert y.shape[0] == 4
    assert y.shape[1] == 128  # constant channels
    assert y.shape[2] == 32  # 128 / 2 / 2


def test_ctc_decoder_identity_proj():
    decoder = CTCDecoder(input_dim=128, hidden_size=128)
    assert isinstance(decoder.input_proj, nn.Identity)


def test_ctc_decoder_linear_proj():
    decoder = CTCDecoder(input_dim=64, hidden_size=128)
    assert isinstance(decoder.input_proj, nn.Linear)
    assert decoder.input_proj.in_features == 64
    assert decoder.input_proj.out_features == 128


# ── Collator tests ───────────────────────────────────────────────


def test_word_to_ctc_target():
    indices, length = _word_to_ctc_target("hello")
    assert indices == [7, 4, 11, 11, 14]
    assert length == 5


def test_ctc_target_encoding_batch():
    words = ["hi", "abc"]
    labels, lengths = _encode_ctc_batch(words, max_label_len=8)
    assert labels.shape == (2, 8)
    assert lengths.tolist() == [2, 3]
    # "hi" -> [7, 8]
    assert labels[0, :2].tolist() == [7, 8]
    assert labels[0, 2:].tolist() == [0] * 6  # padding
    # "abc" -> [0, 1, 2]
    assert labels[1, :3].tolist() == [0, 1, 2]


# ── Config tests ─────────────────────────────────────────────────


def test_distill_config_roundtrip():
    cfg = _small_distill_cfg()
    d = cfg.to_dict()
    cfg2 = SwipeDistillConfig(**d)
    assert cfg2.projector_dim == cfg.projector_dim
    assert cfg2.rnn_hidden == cfg.rnn_hidden
    assert cfg2.adapter_num_stages == cfg.adapter_num_stages
    assert cfg2.num_chars == cfg.num_chars
