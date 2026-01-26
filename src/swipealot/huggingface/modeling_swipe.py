"""HuggingFace-compatible model classes for SwipeTransformer."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_swipe import SwipeTransformerConfig


@dataclass
class SwipeTransformerOutput(ModelOutput):
    """
    Output type for SwipeTransformerModel.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (character prediction).
        char_logits (`torch.FloatTensor` of shape `(batch_size, char_length, vocab_size)`):
            Prediction scores of the character prediction head (text segment only).
        path_logits (`torch.FloatTensor` of shape `(batch_size, path_length, path_input_dim)`, *optional*):
            Prediction scores of the path prediction head (path segment only, if enabled).
        path_log_sigma (`torch.FloatTensor` of shape `(batch_size, path_length, path_input_dim)`, *optional*):
            Predicted log sigma for path coordinates (path segment only, if enabled).
        length_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Predicted length from the length head (if enabled).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            SEP token embeddings for similarity/embedding tasks.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`.
            When requested, this includes the input embeddings plus one entry per encoder layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of attention tensors (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    loss: torch.FloatTensor | None = None
    char_logits: torch.FloatTensor | None = None
    path_logits: torch.FloatTensor | None = None
    path_log_sigma: torch.FloatTensor | None = None
    length_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class SwipeTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = SwipeTransformerConfig
    base_model_prefix = "swipe_transformer"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class SwipeTransformerModel(SwipeTransformerPreTrainedModel):
    """
    HuggingFace-compatible SwipeTransformerModel.

    This model reuses the existing components from src/swipealot/models/
    and wraps them in a HuggingFace-compatible interface.

    Args:
        config (SwipeTransformerConfig): Model configuration
    """

    def __init__(self, config: SwipeTransformerConfig):
        super().__init__(config)
        self.config = config

        # Import existing components
        from ..models.embeddings import MixedEmbedding
        from ..models.heads import (
            CharacterPredictionHead,
            LengthPredictionHead,
            PathPredictionHead,
            PathUncertaintyHead,
        )

        # Embeddings
        self.embeddings = MixedEmbedding(
            vocab_size=config.vocab_size,
            max_path_len=config.max_path_len,
            max_char_len=config.max_char_len,
            d_model=config.d_model,
            dropout=config.dropout,
            path_input_dim=config.path_input_dim,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Prediction heads
        self.char_head = (
            CharacterPredictionHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
            )
            if config.predict_char
            else None
        )

        if config.predict_path:
            self.path_head = PathPredictionHead(
                d_model=config.d_model, output_dim=config.path_input_dim
            )
            self.path_log_sigma_head = (
                PathUncertaintyHead(d_model=config.d_model, output_dim=config.path_input_dim)
                if config.predict_path_uncertainty
                else None
            )
        else:
            self.path_head = None
            self.path_log_sigma_head = None

        # Length prediction head (predicts word length from path)
        # Max length is max_char_len (including EOS)
        self.length_head = (
            LengthPredictionHead(d_model=config.d_model) if config.predict_length else None
        )

        # Initialize weights
        self.post_init()

    def _make_special_tokens(
        self, batch_size: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token = torch.full(
            (batch_size, 1),
            fill_value=self.config.cls_token_id,
            dtype=torch.long,
            device=device,
        )
        sep_token = torch.full(
            (batch_size, 1),
            fill_value=self.config.sep_token_id,
            dtype=torch.long,
            device=device,
        )
        return cls_token, sep_token

    def _src_key_padding_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return attention_mask == 0

    def _encode(
        self,
        embeddings: torch.Tensor,
        *,
        src_key_padding_mask: torch.Tensor | None,
        output_hidden_states: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]:
        """Run encoder with optional per-layer hidden-state + attention capture."""
        attentions: tuple[torch.Tensor, ...] | None = None
        hidden_states_by_layer: list[torch.Tensor] | None = [] if output_hidden_states else None

        hooks = []
        original_forwards: dict[int, callable] = {}
        attentions_buffer: list[torch.Tensor | None] | None = None

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

        def make_hook(layer_idx: int):
            def hook(_module: nn.Module, _input: tuple, output: tuple):
                if (
                    attentions_buffer is not None
                    and isinstance(output, tuple)
                    and len(output) > 1
                    and output[1] is not None
                ):
                    attentions_buffer[layer_idx] = output[1]

            return hook

        if output_attentions:
            attentions_buffer = [None] * len(self.encoder.layers)
            for idx, layer in enumerate(self.encoder.layers):
                attn_module = layer.self_attn
                original_forwards[idx] = attn_module.forward
                attn_module.forward = make_patched_forward(original_forwards[idx])
                hooks.append(attn_module.register_forward_hook(make_hook(idx)))

        try:
            x = embeddings
            for layer in self.encoder.layers:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
                if hidden_states_by_layer is not None:
                    hidden_states_by_layer.append(x)
            hidden_states = x

            if attentions_buffer is not None:
                if any(a is None for a in attentions_buffer):
                    missing = [i for i, a in enumerate(attentions_buffer) if a is None]
                    raise RuntimeError(
                        f"Failed to capture attention weights for layers: {missing}."
                    )
                attentions = tuple(attentions_buffer)  # type: ignore[assignment]
        finally:
            for hook in hooks:
                hook.remove()
            for idx, layer in enumerate(self.encoder.layers):
                if idx in original_forwards:
                    layer.self_attn.forward = original_forwards[idx]

        all_hidden_states = None
        if hidden_states_by_layer is not None:
            all_hidden_states = (embeddings,) + tuple(hidden_states_by_layer)

        return hidden_states, all_hidden_states, attentions

    def _heads(
        self,
        hidden_states: torch.Tensor,
        *,
        path_len: int,
        char_len: int,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        char_logits = None
        if self.char_head is not None:
            char_start = 1 + path_len + 1
            char_hidden = hidden_states[:, char_start : char_start + char_len, :]
            char_logits = self.char_head(char_hidden)

        path_logits = None
        path_log_sigma = None
        if self.path_head is not None:
            path_hidden = hidden_states[:, 1 : 1 + path_len, :]
            path_logits = self.path_head(path_hidden)
            if self.path_log_sigma_head is not None:
                path_log_sigma = self.path_log_sigma_head(path_hidden)

        cls_hidden = hidden_states[:, 0, :]
        length_logits = self.length_head(cls_hidden) if self.length_head is not None else None

        sep_position = 1 + path_len
        pooler_output = hidden_states[:, sep_position, :]
        return char_logits, path_logits, path_log_sigma, length_logits, pooler_output

    def _char_loss(
        self,
        *,
        char_logits: torch.Tensor | None,
        char_labels: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if char_labels is None or self.char_head is None or char_logits is None:
            return None

        labels_flat = char_labels.reshape(-1)
        mask = labels_flat != -100
        if mask.any():
            logits_flat = char_logits.reshape(-1, self.config.vocab_size)[mask]
            labels_flat = labels_flat[mask]
            return nn.functional.cross_entropy(logits_flat, labels_flat, reduction="mean")
        return torch.tensor(0.0, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        path_coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | dict | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Character token IDs [batch, char_len]
            path_coords (torch.Tensor): Path features [batch, path_len, path_input_dim]
                                       Default: [batch, path_len, 6] for (x, y, dx, dy, ds, log_dt)
            attention_mask (torch.Tensor, optional): Attention mask [batch, seq_len]
            labels (torch.Tensor or dict, optional): Labels for loss calculation
                Can be tensor [batch, char_len] or dict with keys like char_labels, path_labels
            return_dict (bool, optional): Whether to return ModelOutput object
            output_hidden_states (bool, optional): Whether to output hidden states
            output_attentions (bool, optional): Whether to output attention weights
            **kwargs: Additional arguments (for compatibility)

        Returns:
            SwipeTransformerOutput or tuple: Model outputs with:
                - loss: Optional loss value
                - char_logits: Character prediction logits [batch, char_len, vocab_size] (if enabled)
                - path_logits: Path prediction logits [batch, path_len, path_input_dim] (if enabled)
                - path_log_sigma: Path log sigma [batch, path_len, path_input_dim] (if enabled)
                - length_logits: Length regression output [batch] (if enabled)
                - last_hidden_state: Hidden states [batch, seq_len, d_model]
                - pooler_output: SEP token embedding [batch, d_model] for similarity/embedding tasks
                - hidden_states: Tuple of per-layer hidden states (if output_hidden_states=True)
                - attentions: Tuple of per-layer attention weights (if output_attentions=True)
        """
        if input_ids is None or path_coords is None:
            raise ValueError("Both input_ids and path_coords are required")

        if isinstance(labels, dict):
            char_labels = labels.get("char_labels")
        else:
            char_labels = labels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )

        batch_size = int(path_coords.shape[0])
        device = path_coords.device
        cls_token, sep_token = self._make_special_tokens(batch_size, device=device)
        embeddings = self.embeddings(path_coords, input_ids, cls_token, sep_token)

        src_key_padding_mask = self._src_key_padding_mask(attention_mask)
        hidden_states, all_hidden_states, attentions = self._encode(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
            output_hidden_states=bool(output_hidden_states),
            output_attentions=bool(output_attentions),
        )

        path_len = int(path_coords.shape[1])
        char_len = int(input_ids.shape[1])
        char_logits, path_logits, path_log_sigma, length_logits, pooler_output = self._heads(
            hidden_states,
            path_len=path_len,
            char_len=char_len,
        )

        loss = self._char_loss(
            char_logits=char_logits,
            char_labels=char_labels,
            device=hidden_states.device,
        )

        if not return_dict:
            output = (
                char_logits,
                path_logits,
                length_logits,
                hidden_states,
                pooler_output,
                all_hidden_states,
                attentions,
                path_log_sigma,
            )
            return (loss,) + output if loss is not None else output

        return SwipeTransformerOutput(
            loss=loss,
            char_logits=char_logits,
            path_logits=path_logits,
            path_log_sigma=path_log_sigma,
            length_logits=length_logits,
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=all_hidden_states,
            attentions=attentions,
        )


#
# Legacy note:
# `SwipeModel` (embeddings-only) has been removed; use `SwipeTransformerModel` and read
# `outputs.pooler_output` for embeddings.
