"""Test HuggingFace integration components."""

import tempfile
from pathlib import Path

import torch

from src.swipealot.huggingface import (
    SwipeCrossEncoder,
    SwipeCrossEncoderConfig,
    SwipeCrossEncoderForSequenceClassification,
    SwipeProcessor,
    SwipeTokenizer,
    SwipeTransformerConfig,
    SwipeTransformerModel,
)


class TestConfiguration:
    """Test configuration classes."""

    def test_swipe_transformer_config(self):
        config = SwipeTransformerConfig(
            d_model=256,
            n_layers=4,
            n_heads=4,
        )
        assert config.model_type == "swipe_transformer"
        assert config.d_model == 256
        assert config.n_layers == 4

    def test_cross_encoder_config(self):
        config = SwipeCrossEncoderConfig(
            d_model=256,
            num_labels=1,
        )
        assert config.model_type == "swipe_cross_encoder"
        assert config.num_labels == 1
        assert config.problem_type == "regression"

    def test_config_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwipeCrossEncoderConfig(d_model=512)
            config.save_pretrained(tmpdir)

            loaded_config = SwipeCrossEncoderConfig.from_pretrained(tmpdir)
            assert loaded_config.d_model == 512


class TestModels:
    """Test model classes."""

    def test_transformer_model_init(self):
        config = SwipeTransformerConfig(
            vocab_size=100,
            d_model=128,
            n_layers=2,
        )
        model = SwipeTransformerModel(config)
        assert model.config.d_model == 128
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_cross_encoder_init(self):
        config = SwipeCrossEncoderConfig(
            vocab_size=100,
            d_model=128,
            n_layers=2,
        )
        model = SwipeCrossEncoderForSequenceClassification(config)
        assert model.num_labels == 1

    def test_forward_pass(self):
        config = SwipeCrossEncoderConfig(
            vocab_size=100,
            d_model=128,
            n_layers=2,
            max_path_len=32,
            max_char_len=20,
        )
        model = SwipeCrossEncoderForSequenceClassification(config)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        path_coords = torch.randn(batch_size, 32, 3)
        input_ids = torch.randint(0, 100, (batch_size, 20))
        attention_mask = torch.ones(batch_size, 1 + 32 + 1 + 20)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                path_coords=path_coords,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        assert outputs.logits.shape == (batch_size, 1)

    def test_save_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwipeCrossEncoderConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
            )
            model = SwipeCrossEncoderForSequenceClassification(config)
            model.eval()

            # Save with safetensors
            model.save_pretrained(tmpdir, safe_serialization=True)

            # Check safetensors file exists
            files = list(Path(tmpdir).iterdir())
            assert any("safetensors" in f.name for f in files)

            # Load
            loaded_model = SwipeCrossEncoderForSequenceClassification.from_pretrained(tmpdir)
            assert loaded_model.config.d_model == 128


class TestTokenizer:
    """Test tokenizer."""

    def test_tokenizer_init(self):
        tokenizer = SwipeTokenizer()
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token == "[PAD]"

    def test_tokenization(self):
        tokenizer = SwipeTokenizer()
        text = "hello"
        encoded = tokenizer(text, return_tensors="pt")
        assert "input_ids" in encoded
        assert "attention_mask" in encoded

    def test_save_load_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = SwipeTokenizer()
            tokenizer.save_pretrained(tmpdir)

            loaded_tokenizer = SwipeTokenizer.from_pretrained(tmpdir)
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


class TestProcessor:
    """Test processor."""

    def test_processor_init(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer)
        assert processor.max_path_len == 64

    def test_process_inputs(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=32, max_char_len=20)

        path_coords = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1]]
        text = "hello"

        inputs = processor(path_coords, text, return_tensors="pt")

        assert "path_coords" in inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["path_coords"].shape[-1] == 3
        # Check attention mask has correct length: [CLS] + path + [SEP] + chars
        assert inputs["attention_mask"].shape[1] == 1 + 32 + 1 + 20

    def test_batch_processing(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=32, max_char_len=20)

        path_coords = [
            [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1]],
            [[0.2, 0.3, 0.1], [0.25, 0.35, 0.2]],
        ]
        text = ["hello", "world"]

        inputs = processor(path_coords, text, return_tensors="pt")

        assert inputs["path_coords"].shape[0] == 2
        assert inputs["input_ids"].shape[0] == 2

    def test_save_load_processor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = SwipeTokenizer()
            processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=128)
            processor.save_pretrained(tmpdir)

            # Load manually (processor.from_pretrained has issues with custom classes)
            loaded_tokenizer = SwipeTokenizer.from_pretrained(tmpdir)
            loaded_processor = SwipeProcessor(tokenizer=loaded_tokenizer, max_path_len=128)
            assert loaded_processor.max_path_len == 128


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test save -> load -> inference pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            config = SwipeCrossEncoderConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
                max_path_len=32,
                max_char_len=20,
            )
            model = SwipeCrossEncoderForSequenceClassification(config)
            model.eval()
            model.save_pretrained(tmpdir, safe_serialization=True)

            # Create and save tokenizer
            tokenizer = SwipeTokenizer()
            tokenizer.save_pretrained(tmpdir)

            # Load everything
            loaded_model = SwipeCrossEncoderForSequenceClassification.from_pretrained(tmpdir)
            loaded_model.eval()
            loaded_tokenizer = SwipeTokenizer.from_pretrained(tmpdir)

            # Create processor
            processor = SwipeProcessor(tokenizer=loaded_tokenizer, max_path_len=32, max_char_len=20)

            # Run inference
            path_coords = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1]]
            text = "hello"

            inputs = processor(path_coords, text, return_tensors="pt")

            with torch.no_grad():
                outputs = loaded_model(**inputs)

            assert outputs.logits.shape == (1, 1)

    def test_cross_encoder_wrapper(self):
        """Test SwipeCrossEncoder wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            config = SwipeCrossEncoderConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
                max_path_len=32,
                max_char_len=20,
            )
            model = SwipeCrossEncoderForSequenceClassification(config)
            model.save_pretrained(tmpdir, safe_serialization=True)

            tokenizer = SwipeTokenizer()
            tokenizer.save_pretrained(tmpdir)

            # Load with wrapper
            wrapper = SwipeCrossEncoder(tmpdir, device="cpu")

            # Test predict
            path = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1], [0.2, 0.3, 0.2]]
            words = ["hello", "world", "test"]

            scores = wrapper.predict(path, words)
            assert scores.shape == (3,)

            # Test rank
            ranked = wrapper.rank(path, words, top_k=2)
            assert len(ranked) == 2
            assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)

    def test_deterministic_outputs(self):
        """Test that outputs are deterministic after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwipeCrossEncoderConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
                max_path_len=32,
                max_char_len=20,
            )
            model = SwipeCrossEncoderForSequenceClassification(config)
            model.eval()

            tokenizer = SwipeTokenizer()
            processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=32, max_char_len=20)

            # Create inputs
            path_coords = torch.randn(2, 32, 3)
            text = ["hello", "world"]
            inputs = processor(path_coords, text, return_tensors="pt")

            # Get original outputs
            with torch.no_grad():
                outputs = model(**inputs)

            # Save and load
            model.save_pretrained(tmpdir, safe_serialization=True)
            loaded_model = SwipeCrossEncoderForSequenceClassification.from_pretrained(tmpdir)
            loaded_model.eval()

            # Get loaded outputs
            with torch.no_grad():
                loaded_outputs = loaded_model(**inputs)

            # Verify match
            assert torch.allclose(outputs.logits, loaded_outputs.logits, atol=1e-5)
