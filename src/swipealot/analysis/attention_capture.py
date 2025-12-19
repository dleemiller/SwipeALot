"""Shared attention-capture utilities for SwipeALot scripts.

Some exported checkpoints may not support `output_attentions=True` in their remote code.
In that case, we fall back to hook-based capture from the encoder layers.
"""

from __future__ import annotations

from typing import Any

import torch


def get_all_layer_attentions(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    print_fallback_note: bool = False,
) -> tuple[object, tuple[torch.Tensor, ...]]:
    """Return `(outputs, attentions)` for all encoder layers.

    Prefers `output_attentions=True` when available, otherwise uses forward hooks to capture
    per-layer self-attention weights.
    """
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    attentions = getattr(outputs, "attentions", None)
    if attentions is not None:
        return outputs, attentions

    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        raise RuntimeError(
            "Model did not return `attentions` and does not expose `model.encoder.layers` "
            "for hook-based attention extraction."
        )

    if print_fallback_note:
        print(
            "   Note: checkpoint remote code does not support `output_attentions`; using hook capture."
        )

    layers = list(encoder.layers)
    buffers: list[torch.Tensor | None] = [None] * len(layers)
    hooks = []
    original_forwards: dict[int, Any] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inp, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                buffers[layer_idx] = output[1].detach()

        return hook

    def make_patched_forward(original_forward):
        def patched_forward(
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=False,
            is_causal=False,
        ):
            return original_forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=False,
                is_causal=is_causal,
            )

        return patched_forward

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        original_forwards[idx] = attn.forward
        attn.forward = make_patched_forward(original_forwards[idx])
        hooks.append(attn.register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
    finally:
        for h in hooks:
            h.remove()
        for idx, layer in enumerate(layers):
            layer.self_attn.forward = original_forwards[idx]

    if any(b is None for b in buffers):
        missing = [i for i, b in enumerate(buffers) if b is None]
        raise RuntimeError(f"Failed to capture attention weights for layers: {missing}")

    return outputs, tuple(buffers)  # type: ignore[return-value]
