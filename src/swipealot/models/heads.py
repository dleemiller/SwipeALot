"""Prediction heads for SwipeTransformer."""

import torch
import torch.nn as nn


class CharacterPredictionHead(nn.Module):
    """Prediction head for masked characters."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, vocab_size] logits
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class PathPredictionHead(nn.Module):
    """Prediction head for masked path coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, 3)  # Predict (x, y, t)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, 3] coordinates in [0, 1] range
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        coords = self.decoder(x)
        coords = torch.sigmoid(coords)  # Ensure [0, 1] range
        return coords


class ClassificationHead(nn.Module):
    """
    Classification head for cross-encoder.

    Follows SBERT architecture: Dense → GELU → LayerNorm → Linear(→1)
    Outputs a single similarity score per input.
    """

    def __init__(self, d_model: int, num_labels: int = 1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, d_model] - typically SEP token embeddings

        Returns:
            [batch, num_labels] similarity scores
        """
        x = self.dense(features)
        x = self.activation(x)  # GELU
        x = self.norm(x)  # LayerNorm
        logits = self.classifier(x)  # [batch, 1] or [batch, num_labels]
        return logits
