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

    def __init__(self, d_model: int, output_dim: int = 6):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, output_dim)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, output_dim] path features.
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        features = self.decoder(x)

        # Per-feature constraints:
        # - x, y are normalized to [0,1]
        # - dx, dy are signed deltas (roughly [-1,1])
        # - ds is non-negative
        # - log_dt is non-negative
        if features.shape[-1] == 6:
            x_y = torch.sigmoid(features[..., 0:2])
            dx_dy = torch.tanh(features[..., 2:4])
            ds = torch.nn.functional.softplus(features[..., 4:5])
            log_dt = torch.nn.functional.softplus(features[..., 5:6])
            return torch.cat([x_y, dx_dy, ds, log_dt], dim=-1)

        # Fallback: unconstrained regression for other output dims.
        return features


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


class LengthPredictionHead(nn.Module):
    """Regress sequence length (e.g., swipable character count) from CLS embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.regressor = nn.Linear(d_model, 1)  # predict expected length directly

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_features: [batch, d_model] CLS embeddings

        Returns:
            [batch, 1] predicted length
        """
        x = self.dense(cls_features)
        x = self.activation(x)
        x = self.norm(x)
        return self.regressor(x).squeeze(-1)
