"""Embedding layers for SwipeTransformer."""

import torch
import torch.nn as nn


class PathEmbedding(nn.Module):
    """Embeds path features (x, y, dx, dy, ds, log_dt) to d_model dimension."""

    def __init__(self, d_model: int = 256, input_dim: int = 6):
        """
        Initialize path embedding layer.

        Args:
            d_model: Output dimension
            input_dim: Input feature dimension (default: 6 for x, y, dx, dy, ds, log_dt)
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, path_coords: torch.Tensor) -> torch.Tensor:
        """
        Project path features to d_model dimension.

        Args:
            path_coords: [batch, seq_len, input_dim] - path features
                         Default: (x, y, dx, dy, ds, log_dt) with input_dim=6

        Returns:
            [batch, seq_len, d_model] embeddings
        """
        return self.projection(path_coords)


class CharacterEmbedding(nn.Module):
    """Embeds character tokens."""

    def __init__(self, vocab_size: int, d_model: int = 256, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, char_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_tokens: [batch, seq_len] character token IDs

        Returns:
            [batch, seq_len, d_model] embeddings
        """
        return self.embedding(char_tokens)


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, max_position: int, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(max_position, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch, seq_len] position indices

        Returns:
            [batch, seq_len, d_model] positional embeddings
        """
        return self.embedding(positions)


class TypeEmbedding(nn.Module):
    """Token type embeddings to distinguish PATH (0) vs TEXT (1) tokens."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        # 0 = PATH, 1 = TEXT
        self.embedding = nn.Embedding(2, d_model)

    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_types: [batch, seq_len] type indices (0 or 1)

        Returns:
            [batch, seq_len, d_model] type embeddings
        """
        return self.embedding(token_types)


class MixedEmbedding(nn.Module):
    """
    Combines path and character embeddings with positional and type information.
    Constructs sequence: [CLS] + path_tokens + [SEP] + char_tokens
    """

    def __init__(
        self,
        vocab_size: int,
        max_path_len: int,
        max_char_len: int,
        d_model: int = 256,
        dropout: float = 0.1,
        path_input_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model

        # Content embeddings
        self.path_embedding = PathEmbedding(d_model, input_dim=path_input_dim)
        self.char_embedding = CharacterEmbedding(vocab_size, d_model, padding_idx=0)

        # Positional embeddings
        max_seq_len = 1 + max_path_len + 1 + max_char_len  # [CLS] + path + [SEP] + chars
        self.positional_embedding = PositionalEmbedding(max_seq_len, d_model)

        # Type embeddings
        self.type_embedding = TypeEmbedding(d_model)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        cls_token: torch.Tensor,
        sep_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create mixed sequence with embeddings.

        Args:
            path_coords: [batch, path_len, path_input_dim] path features
                         Default: [batch, path_len, 6] for (x, y, dx, dy, ds, log_dt)
            char_tokens: [batch, char_len] character token IDs
            cls_token: [batch, 1] CLS token IDs
            sep_token: [batch, 1] SEP token IDs

        Returns:
            [batch, total_seq_len, d_model] embeddings where
            total_seq_len = 1 + path_len + 1 + char_len
        """
        batch_size = path_coords.shape[0]
        path_len = path_coords.shape[1]
        char_len = char_tokens.shape[1]
        device = path_coords.device

        # Embed [CLS]
        cls_emb = self.char_embedding(cls_token)  # [batch, 1, d_model]

        # Embed path
        path_emb = self.path_embedding(path_coords)  # [batch, path_len, d_model]

        # Embed [SEP]
        sep_emb = self.char_embedding(sep_token)  # [batch, 1, d_model]

        # Embed characters
        char_emb = self.char_embedding(char_tokens)  # [batch, char_len, d_model]

        # Concatenate: [CLS] + PATH + [SEP] + CHARS
        sequence = torch.cat(
            [cls_emb, path_emb, sep_emb, char_emb], dim=1
        )  # [batch, seq_len, d_model]
        seq_len = sequence.shape[1]

        # Add positional embeddings
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )  # [batch, seq_len]
        pos_emb = self.positional_embedding(positions)

        # Add type embeddings
        # Type 0 for [CLS] + path + [SEP], Type 1 for chars
        type_ids = torch.cat(
            [
                torch.zeros(
                    batch_size, 1 + path_len + 1, dtype=torch.long, device=device
                ),  # [CLS], path, [SEP]
                torch.ones(batch_size, char_len, dtype=torch.long, device=device),  # chars
            ],
            dim=1,
        )  # [batch, seq_len]
        type_emb = self.type_embedding(type_ids)

        # Combine: content + position + type
        embeddings = sequence + pos_emb + type_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
