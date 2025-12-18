"""Main transformer model for swipe keyboard prediction."""

import torch
import torch.nn as nn

from .embeddings import MixedEmbedding
from .heads import CharacterPredictionHead, LengthPredictionHead, PathPredictionHead


class SwipeTransformerModel(nn.Module):
    """Complete swipe keyboard transformer model."""

    def __init__(self, config):
        """
        Initialize model from configuration.

        Args:
            config: ModelConfig object
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = MixedEmbedding(
            vocab_size=config.vocab_size,
            max_path_len=config.max_path_len,
            max_char_len=config.max_char_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm architecture
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Prediction heads
        self.char_head = CharacterPredictionHead(config.d_model, config.vocab_size)
        self.length_head = (
            LengthPredictionHead(config.d_model, max_length=config.max_char_len)
            if getattr(config, "predict_length", False)
            else None
        )

        if config.predict_path:
            self.path_head = PathPredictionHead(config.d_model)
        else:
            self.path_head = None

        # Initialize weights
        self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     """Initialize weights following standard practice."""
    #     if isinstance(module, nn.Linear):
    #         nn.init.normal_(module.weight, std=0.02)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         nn.init.normal_(module.weight, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         nn.init.ones_(module.weight)
    #         nn.init.zeros_(module.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            path_coords: [batch, path_len, 3] path coordinates
            char_tokens: [batch, char_len] character token IDs
            attention_mask: [batch, seq_len] attention mask (1 = attend, 0 = ignore)

        Returns:
            Dictionary with:
                - char_logits: [batch, seq_len, vocab_size]
                - path_coords_pred: [batch, seq_len, 3] (if predict_path=True)
                - hidden_states: [batch, seq_len, d_model]
        """
        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens (special token IDs: CLS=1, SEP=2)
        cls_token = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=device)
        sep_token = torch.full((batch_size, 1), fill_value=2, dtype=torch.long, device=device)

        # Get embeddings
        embeddings = self.embeddings(path_coords, char_tokens, cls_token, sep_token)

        # Prepare attention mask for transformer
        # PyTorch TransformerEncoder expects mask where True = ignore
        if attention_mask is not None:
            # Convert from (1 = attend, 0 = ignore) to (True = ignore, False = attend)
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Predictions
        char_logits = self.char_head(hidden_states)

        outputs = {"char_logits": char_logits, "hidden_states": hidden_states}

        if self.path_head is not None:
            path_coords_pred = self.path_head(hidden_states)
            outputs["path_coords_pred"] = path_coords_pred

        if self.length_head is not None:
            cls_states = hidden_states[:, 0, :]  # CLS position
            length_logits = self.length_head(cls_states)
            outputs["length_logits"] = length_logits

        return outputs
