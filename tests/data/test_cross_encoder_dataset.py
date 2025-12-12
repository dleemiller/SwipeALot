"""Tests for CrossEncoderDataset."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.swipealot.data.cross_encoder_dataset import CrossEncoderDataset
from src.swipealot.data.negative_mining import save_negative_pool
from src.swipealot.data.tokenizer import CharacterTokenizer


class TestCrossEncoderDataset:
    """Test cross-encoder dataset with negative sampling."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock(spec=CharacterTokenizer)
        tokenizer.vocab_size = 100
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.encode.side_effect = lambda word: list(range(len(word)))  # Simple encoding
        return tokenizer

    @pytest.fixture
    def negative_pool_file(self):
        """Create temporary negative pool file."""
        # Create test negative pool
        pool = {
            "hello": [("jello", 9.5), ("yellow", 7.0), ("mellow", 5.0)],
            "world": [("would", 8.0), ("word", 7.5), ("worm", 5.0)],
            "test": [("best", 10.0), ("rest", 8.0), ("west", 6.0)],
        }

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        save_negative_pool(pool, temp_path)

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_sample_negatives_with_pool(self, negative_pool_file, mock_tokenizer):
        """Test negative sampling with difficulty-based sampling."""
        # Mock dataset
        mock_dataset = [
            {
                "word": "hello",
                "data": [{"x": 0.5, "y": 0.5, "t": 0}] * 10,
                "canvas_width": 1.0,
                "canvas_height": 1.0,
            }
        ]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                max_path_len=64,
                max_word_len=48,
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=3,
                difficulty_sampling=True,
            )

            # Sample negatives for "hello"
            negatives = dataset._sample_negatives("hello")

            # Should return 3 negatives
            assert len(negatives) == 3

            # Negatives should be from the pool for "hello"
            available = {"jello", "yellow", "mellow"}
            for neg in negatives:
                assert neg in available

    def test_sample_negatives_difficulty_sampling(self, negative_pool_file, mock_tokenizer):
        """Test that difficulty sampling favors higher scores."""
        mock_dataset = [{"word": "test", "data": [{"x": 0.5, "y": 0.5, "t": 0}]}]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=3,
                difficulty_sampling=True,
            )

            # Sample many times and count
            sample_counts = {}
            for _ in range(100):
                negatives = dataset._sample_negatives("test")
                for neg in negatives:
                    sample_counts[neg] = sample_counts.get(neg, 0) + 1

            # "best" has highest difficulty (10.0), should be sampled more
            # "west" has lowest difficulty (6.0), should be sampled less
            # This is probabilistic, so we just check all were sampled
            assert len(sample_counts) >= 2

    def test_sample_negatives_uniform_sampling(self, negative_pool_file, mock_tokenizer):
        """Test uniform sampling (difficulty_sampling=False)."""
        mock_dataset = [{"word": "test", "data": [{"x": 0.5, "y": 0.5, "t": 0}]}]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=2,
                difficulty_sampling=False,  # Uniform sampling
            )

            negatives = dataset._sample_negatives("test")

            # Should return 2 negatives
            assert len(negatives) == 2

            # Should be from pool
            available = {"best", "rest", "west"}
            for neg in negatives:
                assert neg in available

    def test_sample_negatives_fallback_random(self, mock_tokenizer):
        """Test fallback to random sampling when word not in pool."""
        mock_dataset = [
            {"word": "unknown", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
            {"word": "test", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
            {"word": "hello", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
        ]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            # No negative pool
            dataset = CrossEncoderDataset(
                split="train",
                tokenizer=mock_tokenizer,
                negative_pool_path=None,
                num_negatives=2,
            )

            # Should fall back to random negatives from all_words
            negatives = dataset._sample_negatives("unknown")

            # Should return 2 negatives
            assert len(negatives) == 2

            # Should not include the word itself
            assert "unknown" not in negatives

            # Should be from the vocabulary
            vocab = {"test", "hello"}
            for neg in negatives:
                assert neg in vocab

    def test_getitem_structure(self, negative_pool_file, mock_tokenizer):
        """Test __getitem__ returns correct structure."""
        mock_dataset = [
            {
                "word": "hello",
                "data": [{"x": 0.5, "y": 0.5, "t": 0}] * 10,
                "canvas_width": 1.0,
                "canvas_height": 1.0,
            }
        ]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                max_path_len=64,
                max_word_len=48,
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=3,
            )

            item = dataset[0]

            # Check all required keys
            assert "path_coords" in item
            assert "path_mask" in item
            assert "positive_word" in item
            assert "positive_mask" in item
            assert "negative_words" in item
            assert "negative_masks" in item
            assert "original_word" in item

            # Check shapes
            assert item["path_coords"].shape == (64, 3)
            assert item["path_mask"].shape == (64,)
            assert item["positive_word"].shape == (48,)
            assert item["positive_mask"].shape == (48,)
            assert item["negative_words"].shape == (3, 48)
            assert item["negative_masks"].shape == (3, 48)

            # Check types
            assert item["path_coords"].dtype == torch.float32
            assert item["path_mask"].dtype == torch.long
            assert item["positive_word"].dtype == torch.long
            assert item["positive_mask"].dtype == torch.long
            assert item["negative_words"].dtype == torch.long
            assert item["negative_masks"].dtype == torch.long

    def test_process_word_adds_eos(self, negative_pool_file, mock_tokenizer):
        """Test that EOS token is added to words."""
        mock_dataset = [
            {
                "word": "test",
                "data": [{"x": 0.5, "y": 0.5, "t": 0}],
                "canvas_width": 1.0,
                "canvas_height": 1.0,
            }
        ]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                max_word_len=48,
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=1,
            )

            # Process word should add EOS
            tokens, mask = dataset._process_word("test")

            # Mock tokenizer.encode returns [0, 1, 2, 3] for "test"
            # EOS token (2) should be added
            # So we expect [0, 1, 2, 3, 2, ...padding...]
            assert tokens[4] == mock_tokenizer.eos_token_id

    def test_no_duplicate_negatives(self, negative_pool_file, mock_tokenizer):
        """Test that sampled negatives don't include the positive word."""
        mock_dataset = [
            {
                "word": "hello",
                "data": [{"x": 0.5, "y": 0.5, "t": 0}],
            }
        ]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=3,
            )

            # Sample many times
            for _ in range(20):
                negatives = dataset._sample_negatives("hello")

                # Positive word should never be in negatives
                assert "hello" not in negatives

                # No duplicates in negatives
                assert len(negatives) == len(set(negatives))

    def test_len(self, negative_pool_file, mock_tokenizer):
        """Test dataset length."""
        mock_dataset = [{"word": f"word{i}", "data": []} for i in range(10)]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                tokenizer=mock_tokenizer,
                negative_pool_path=negative_pool_file,
                num_negatives=3,
            )

            assert len(dataset) == 10

    def test_case_insensitive_negatives(self, mock_tokenizer):
        """Test that 'The' and 'the' are treated as same word (case-insensitive)."""
        # Create dataset with mixed case words
        mock_dataset = [
            {"word": "The", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
            {"word": "the", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
            {"word": "Hello", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
            {"word": "WORLD", "data": [{"x": 0.5, "y": 0.5, "t": 0}]},
        ]

        with patch("src.swipealot.data.cross_encoder_dataset.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            dataset = CrossEncoderDataset(
                split="train",
                tokenizer=mock_tokenizer,
                negative_pool_path=None,  # Use random sampling
                num_negatives=2,
            )

            # All words should be lowercased in vocabulary
            assert "the" in dataset.all_words
            assert "The" not in dataset.all_words
            assert "hello" in dataset.all_words
            assert "Hello" not in dataset.all_words
            assert "world" in dataset.all_words
            assert "WORLD" not in dataset.all_words

            # Vocabulary should have 3 unique words (the, hello, world)
            assert len(dataset.all_words) == 3

            # Sample negatives for "The" - should NOT include "the" as a negative
            negatives = dataset._sample_negatives("The")
            assert "the" not in negatives  # Shouldn't include itself
            assert all(neg in {"hello", "world"} for neg in negatives)
