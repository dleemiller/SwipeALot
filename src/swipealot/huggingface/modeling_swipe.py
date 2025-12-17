"""HuggingFace-compatible model classes for SwipeTransformer."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    ModelOutput,
    SequenceClassifierOutput,
)

from .configuration_swipe import SwipeTransformerConfig


@dataclass
class SwipeTransformerOutput(ModelOutput):
    """
    Output type for SwipeTransformerModel.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (character prediction).
        char_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the character prediction head.
        path_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 3)`, *optional*):
            Prediction scores of the path prediction head (if enabled).
        length_logits (`torch.FloatTensor` of shape `(batch_size, max_length+1)`, *optional*):
            Prediction scores of the length prediction head (if enabled).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            SEP token embeddings for similarity/embedding tasks.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    """

    loss: torch.FloatTensor | None = None
    char_logits: torch.FloatTensor = None
    path_logits: torch.FloatTensor | None = None
    length_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


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
        from ..models.heads import CharacterPredictionHead, LengthPredictionHead, PathPredictionHead

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
            self.path_head = PathPredictionHead(d_model=config.d_model)
        else:
            self.path_head = None

        # Length prediction head (predicts word length from path)
        # Max length is max_char_len (including EOS)
        self.length_head = LengthPredictionHead(d_model=config.d_model) if config.predict_length else None

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        path_coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | dict | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
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
            **kwargs: Additional arguments (for compatibility)

        Returns:
            SwipeTransformerOutput or tuple: Model outputs with:
                - loss: Optional loss value
                - char_logits: Character prediction logits [batch, seq_len, vocab_size]
                - path_logits: Path prediction logits [batch, seq_len, 3] (if predict_path=True)
                - length_logits: Length prediction logits [batch, max_length]
                - last_hidden_state: Hidden states [batch, seq_len, d_model]
                - pooler_output: SEP token embeddings [batch, d_model] for similarity/embedding tasks
                - hidden_states: Tuple of hidden states (if output_hidden_states=True)
        """
        # Validate required inputs
        if input_ids is None or path_coords is None:
            raise ValueError("Both input_ids and path_coords are required")

        # Extract labels if dict (used by custom trainers)
        if isinstance(labels, dict):
            char_labels = labels.get("char_labels")
            # Can handle other label types in the future (path_labels, etc.)
        else:
            char_labels = labels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens
        cls_token = torch.full(
            (batch_size, 1), fill_value=self.config.cls_token_id, dtype=torch.long, device=device
        )
        sep_token = torch.full(
            (batch_size, 1), fill_value=self.config.sep_token_id, dtype=torch.long, device=device
        )

        # Get embeddings
        embeddings = self.embeddings(path_coords, input_ids, cls_token, sep_token)

        # Prepare attention mask for encoder
        if attention_mask is not None:
            # Convert attention mask: 1 = attend, 0 = ignore
            # PyTorch expects: False = attend, True = ignore
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode (batch_first=True is set in TransformerEncoderLayer)
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Character prediction
        char_logits = self.char_head(hidden_states) if self.char_head is not None else None

        # Path prediction (if enabled)
        path_logits = None
        if self.path_head is not None:
            path_logits = self.path_head(hidden_states)

        # Length prediction from CLS token
        cls_hidden = hidden_states[:, 0, :]  # [batch, d_model] - CLS at position 0
        length_logits = self.length_head(cls_hidden) if self.length_head is not None else None

        # Extract SEP token embedding for pooler output (embeddings/similarity tasks)
        # SEP is at position 1 + path_len
        path_len = path_coords.shape[1]
        sep_position = 1 + path_len
        pooler_output = hidden_states[:, sep_position, :]  # [batch, d_model]

        # Compute loss if labels provided
        loss = None
        if char_labels is not None and self.char_head is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            # Extract character positions from hidden states
            # Sequence is: [CLS] + path + [SEP] + chars
            char_start = 1 + path_len + 1  # After [CLS], path, and [SEP]
            char_hidden = hidden_states[:, char_start : char_start + char_labels.shape[1], :]
            char_pred = self.char_head(char_hidden)
            loss = loss_fct(char_pred.reshape(-1, self.config.vocab_size), char_labels.reshape(-1))

        if not return_dict:
            output = (hidden_states, char_logits, length_logits, pooler_output)
            if path_logits is not None:
                output = output + (path_logits,)
            return ((loss,) + output) if loss is not None else output

        return SwipeTransformerOutput(
            loss=loss,
            char_logits=char_logits,
            path_logits=path_logits,
            length_logits=length_logits,
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )



class SwipeModel(SwipeTransformerPreTrainedModel):
    """
    Base Swipe model for extracting embeddings.

    .. deprecated::
        This class is deprecated. Use SwipeTransformerModel instead, which now
        includes pooler_output for embeddings alongside prediction heads.
        SwipeTransformerModel provides both predictions AND embeddings in a single model.

    This model returns the SEP token embedding, which can be used for:
    - Vector databases
    - Semantic search
    - Similarity computation

    The SEP token embedding represents the joint encoding of the path and text.

    Usage (Deprecated - use SwipeTransformerModel instead):
        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "your-username/swipe-model",
            trust_remote_code=True
        )

        # Get embeddings
        outputs = model(path_coords=paths, input_ids=tokens)
        embeddings = outputs.pooler_output  # SEP token embeddings
        ```

    Args:
        config (SwipeTransformerConfig): Model configuration
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Import existing components
        from ..models.embeddings import MixedEmbedding

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

        # Initialize weights
        self.post_init()

    def forward(
        self,
        path_coords: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
    ):
        """
        Forward pass that returns embeddings.

        Args:
            path_coords (torch.Tensor): Path features [batch, path_len, path_input_dim]
                                       Default: [batch, path_len, 6] for (x, y, dx, dy, ds, log_dt)
            input_ids (torch.Tensor): Character token IDs [batch, char_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch, seq_len]
            return_dict (bool, optional): Whether to return ModelOutput object
            output_hidden_states (bool, optional): Whether to output all hidden states

        Returns:
            BaseModelOutputWithPooling with:
                - last_hidden_state: Full sequence hidden states [batch, seq_len, d_model]
                - pooler_output: SEP token embeddings [batch, d_model]
                - hidden_states: Tuple of hidden states (if output_hidden_states=True)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens
        cls_token = torch.full(
            (batch_size, 1), fill_value=self.config.cls_token_id, dtype=torch.long, device=device
        )
        sep_token = torch.full(
            (batch_size, 1), fill_value=self.config.sep_token_id, dtype=torch.long, device=device
        )

        # Get embeddings
        embeddings = self.embeddings(path_coords, input_ids, cls_token, sep_token)

        # Prepare attention mask
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode (batch_first=True is set in TransformerEncoderLayer)
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Extract SEP token embedding (pooler output)
        # SEP is at position 1 + path_len
        path_len = path_coords.shape[1]
        sep_position = 1 + path_len
        pooler_output = hidden_states[:, sep_position, :]  # [batch, d_model]

        if not return_dict:
            return (hidden_states, pooler_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )
