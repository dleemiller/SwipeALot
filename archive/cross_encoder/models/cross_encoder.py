"""Cross-encoder model for swipe path/word matching.

Wraps the pretrained SwipeTransformerModel and adds a classification head
for binary similarity scoring.
"""

import torch
import torch.nn as nn

from .heads import ClassificationHead
from .transformer import SwipeTransformerModel


class SwipeCrossEncoderModel(nn.Module):
    """
    Cross-encoder for swipe path and word matching.

    Uses the pretrained SwipeTransformerModel encoder and adds a
    classification head on top of the SEP token embedding.
    """

    def __init__(self, base_model: SwipeTransformerModel, num_labels: int = 1):
        """
        Initialize cross-encoder from pretrained base model.

        Args:
            base_model: Pretrained SwipeTransformerModel
            num_labels: Number of output labels (1 for similarity score)
        """
        super().__init__()

        # Store base model components
        self.embeddings = base_model.embeddings
        self.encoder = base_model.encoder
        self.config = base_model.config

        # Add classification head
        self.classifier = ClassificationHead(
            d_model=base_model.config.d_model, num_labels=num_labels
        )

        # Initialize classification head weights
        self.classifier.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for classification head."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def freeze_encoder(self):
        """Freeze all encoder parameters (embeddings + transformer)."""
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.embeddings.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, num_layers: int = 2):
        """
        Unfreeze only the top N transformer layers.

        Args:
            num_layers: Number of top layers to unfreeze (default: 2)
        """
        # Keep embeddings frozen
        for param in self.embeddings.parameters():
            param.requires_grad = False

        # Freeze all encoder layers first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze top N layers
        total_layers = len(self.encoder.layers)
        for i in range(total_layers - num_layers, total_layers):
            for param in self.encoder.layers[i].parameters():
                param.requires_grad = True

        # Unfreeze final LayerNorm if present
        if hasattr(self.encoder, "norm") and self.encoder.norm is not None:
            for param in self.encoder.norm.parameters():
                param.requires_grad = True

    def forward(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-encoder.

        Args:
            path_coords: [batch, path_len, 3] path coordinates
            char_tokens: [batch, char_len] character token IDs
            attention_mask: [batch, seq_len] attention mask (1 = attend, 0 = ignore)

        Returns:
            [batch, num_labels] similarity scores
        """
        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens
        cls_token = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=device)
        sep_token = torch.full((batch_size, 1), fill_value=2, dtype=torch.long, device=device)

        # Get embeddings
        embeddings = self.embeddings(path_coords, char_tokens, cls_token, sep_token)

        # Prepare attention mask
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Extract SEP token embedding
        # SEP is at position 1 + path_len
        path_len = path_coords.shape[1]
        sep_position = 1 + path_len
        sep_embedding = hidden_states[:, sep_position, :]  # [batch, d_model]

        # Classification
        logits = self.classifier(sep_embedding)  # [batch, num_labels]

        return logits

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str, config, num_labels: int = 1, device: str = "cpu"
    ) -> "SwipeCrossEncoderModel":
        """
        Load cross-encoder from pretrained SwipeTransformerModel checkpoint.

        Args:
            checkpoint_path: Path to pretrained model checkpoint
            config: ModelConfig for base model architecture
            num_labels: Number of output labels (default: 1)
            device: Device to load model on

        Returns:
            SwipeCrossEncoderModel with loaded weights
        """
        # Create base model
        base_model = SwipeTransformerModel(config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load weights into base model
        base_model.load_state_dict(state_dict, strict=False)

        # Create cross-encoder
        cross_encoder = cls(base_model, num_labels=num_labels)

        return cross_encoder
