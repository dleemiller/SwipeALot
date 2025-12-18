"""Tests for CrossEncoderCollator."""

from unittest.mock import MagicMock

import pytest
import torch

from src.swipealot.data.collators import CrossEncoderCollator
from src.swipealot.data.tokenizer import CharacterTokenizer


class TestCrossEncoderCollator:
    """Test cross-encoder collator batching logic."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock(spec=CharacterTokenizer)
        tokenizer.pad_token_id = 0
        return tokenizer

    def test_initialization(self, mock_tokenizer):
        """Test collator initialization."""
        collator = CrossEncoderCollator(tokenizer=mock_tokenizer)
        assert collator.tokenizer == mock_tokenizer

    def test_single_item_shapes(self, mock_tokenizer):
        """Test batching with single item returns correct shapes."""
        collator = CrossEncoderCollator(tokenizer=mock_tokenizer)

        # Single item: 1 positive + 3 negatives
        batch = [
            {
                "path_coords": torch.randn(64, 3),
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (3, 48)),
                "negative_masks": torch.ones(3, 48, dtype=torch.long),
            }
        ]

        result = collator(batch)

        # Should have 1 * (1 + 3) = 4 pairs total
        assert result["path_coords"].shape[0] == 4
        assert result["char_tokens"].shape[0] == 4
        assert result["attention_mask"].shape[0] == 4

        # Labels should be [0] (positive at index 0)
        assert result["labels"].shape == (1,)
        assert result["labels"][0] == 0

        # Group sizes should be [4]
        assert result["group_sizes"].shape == (1,)
        assert result["group_sizes"][0] == 4

    def test_multiple_items_batch(self, mock_tokenizer):
        """Test batching with multiple items."""
        collator = CrossEncoderCollator(tokenizer=mock_tokenizer)

        # 2 items: each with 1 positive + 2 negatives
        batch = [
            {
                "path_coords": torch.randn(64, 3),
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (2, 48)),
                "negative_masks": torch.ones(2, 48, dtype=torch.long),
            },
            {
                "path_coords": torch.randn(64, 3),
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (2, 48)),
                "negative_masks": torch.ones(2, 48, dtype=torch.long),
            },
        ]

        result = collator(batch)

        # Should have 2 * (1 + 2) = 6 pairs total
        assert result["path_coords"].shape[0] == 6
        assert result["char_tokens"].shape[0] == 6
        assert result["attention_mask"].shape[0] == 6

        # Labels should be [0, 0]
        assert result["labels"].shape == (2,)
        assert torch.all(result["labels"] == 0)

        # Group sizes should be [3, 3]
        assert result["group_sizes"].shape == (2,)
        assert torch.all(result["group_sizes"] == 3)

    def test_path_repetition(self, mock_tokenizer):
        """Test that path is repeated for each positive/negative pair."""
        collator = CrossEncoderCollator(tokenizer=mock_tokenizer)

        # Create item with distinct path values
        path = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3)

        batch = [
            {
                "path_coords": path,
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (2, 48)),
                "negative_masks": torch.ones(2, 48, dtype=torch.long),
            }
        ]

        result = collator(batch)

        # Should have 1 + 2 = 3 pairs
        assert result["path_coords"].shape[0] == 3

        # All 3 pairs should have the same path
        assert torch.all(result["path_coords"][0] == path)
        assert torch.all(result["path_coords"][1] == path)
        assert torch.all(result["path_coords"][2] == path)

    def test_labels_always_zero(self, mock_tokenizer):
        """Test that labels are always 0 (positive at index 0)."""
        collator = CrossEncoderCollator(tokenizer=mock_tokenizer)

        batch = [
            {
                "path_coords": torch.randn(64, 3),
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (3, 48)),
                "negative_masks": torch.ones(3, 48, dtype=torch.long),
            }
            for _ in range(5)
        ]

        result = collator(batch)

        # All labels should be 0
        assert result["labels"].shape == (5,)
        assert torch.all(result["labels"] == 0)

    def test_variable_num_negatives(self, mock_tokenizer):
        """Test with different numbers of negatives."""
        collator = CrossEncoderCollator(tokenizer=mock_tokenizer)

        # Test with 1 negative
        batch_1 = [
            {
                "path_coords": torch.randn(64, 3),
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (1, 48)),
                "negative_masks": torch.ones(1, 48, dtype=torch.long),
            }
        ]
        result_1 = collator(batch_1)
        assert result_1["path_coords"].shape[0] == 2  # 1 + 1

        # Test with 5 negatives
        batch_5 = [
            {
                "path_coords": torch.randn(64, 3),
                "path_mask": torch.ones(64, dtype=torch.long),
                "positive_word": torch.randint(0, 100, (48,)),
                "positive_mask": torch.ones(48, dtype=torch.long),
                "negative_words": torch.randint(0, 100, (5, 48)),
                "negative_masks": torch.ones(5, 48, dtype=torch.long),
            }
        ]
        result_5 = collator(batch_5)
        assert result_5["path_coords"].shape[0] == 6  # 1 + 5
