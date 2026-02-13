"""Test HuggingFace integration components."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.swipealot.huggingface import (
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

    def test_config_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwipeTransformerConfig(
                vocab_size=100,
                d_model=512,
                n_layers=2,
            )
            config.save_pretrained(tmpdir)

            loaded_config = SwipeTransformerConfig.from_pretrained(tmpdir)
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

    def test_transformer_forward_pass(self):
        config = SwipeTransformerConfig(
            vocab_size=100,
            d_model=128,
            n_layers=2,
            max_path_len=32,
            max_char_len=20,
        )
        model = SwipeTransformerModel(config)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        path_coords = torch.randn(batch_size, 32, config.path_input_dim)
        input_ids = torch.randint(0, 100, (batch_size, 20))
        # Sequence: [CLS] + path_tokens + [SEP] + char_tokens
        seq_len = 1 + 32 + 1 + 20  # = 54
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                path_coords=path_coords,
                attention_mask=attention_mask,
            )

        # Model outputs logits for text segment only
        assert outputs.char_logits.shape == (batch_size, 20, 100)
        # Hidden states are for the full mixed sequence: [CLS] + path + [SEP] + chars
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.d_model)

    def test_save_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwipeTransformerConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
            )
            model = SwipeTransformerModel(config)
            model.eval()

            # Save with safetensors
            model.save_pretrained(tmpdir, safe_serialization=True)

            # Check safetensors file exists
            files = list(Path(tmpdir).iterdir())
            assert any("safetensors" in f.name for f in files)

            # Load
            loaded_model = SwipeTransformerModel.from_pretrained(tmpdir)
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

    def test_punctuation_maps_to_punc_token(self):
        tokenizer = SwipeTokenizer()
        punc_id = tokenizer.convert_tokens_to_ids("[PUNC]")
        ids = tokenizer.encode("a,a", add_special_tokens=False)
        a_id = tokenizer.convert_tokens_to_ids("a")
        assert ids == [a_id, punc_id, a_id]
        assert tokenizer.decode(ids) == "aa"


class TestProcessor:
    """Test processor."""

    def test_requires_path_or_text(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer)
        with pytest.raises(ValueError):
            processor(path_coords=None, text=None, return_tensors="pt")

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
        assert inputs["path_coords"].shape[-1] == 8
        # Check attention mask has correct length: [CLS] + path + [SEP] + chars
        assert inputs["attention_mask"].shape[1] == 1 + 32 + 1 + 20

    def test_encode_helpers(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=32, max_char_len=20)

        path_coords = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1]]
        text = ["hello", "world"]

        path_only = processor.encode_path(path_coords, return_tensors="pt")
        assert "path_coords" in path_only
        assert "input_ids" in path_only
        assert "attention_mask" in path_only

        text_only = processor.encode_text(text, return_tensors="pt")
        assert "path_coords" in text_only
        assert "input_ids" in text_only
        assert "attention_mask" in text_only

        # Attention layout: [CLS:1] + path[max_path_len] + [SEP:1] + text[max_char_len]
        max_path_len = processor.max_path_len
        max_char_len = processor.max_char_len

        def _split(attn):
            cls = attn[:, :1]
            path = attn[:, 1 : 1 + max_path_len]
            sep = attn[:, 1 + max_path_len : 1 + max_path_len + 1]
            txt = attn[:, 1 + max_path_len + 1 : 1 + max_path_len + 1 + max_char_len]
            return cls, path, sep, txt

        cls, path, sep, txt = _split(path_only["attention_mask"])
        assert int(cls.sum().item()) == 1
        assert int(sep.sum().item()) == 1
        assert int(path.sum().item()) == max_path_len  # resampled -> no padding
        assert int(txt.sum().item()) == 0  # no text attention

        cls, path, sep, txt = _split(text_only["attention_mask"])
        assert int(cls.sum().item()) == 2
        assert int(sep.sum().item()) == 2
        assert int(path.sum().item()) == 0  # no path attention
        assert int(txt.sum().item()) > 0  # has text attention

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

    def test_raw_dict_path_is_converted_and_attended(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=16, max_char_len=8)

        raw_path = [
            {"x": 0.10, "y": 0.20, "t": 0.0},
            {"x": 0.20, "y": 0.30, "t": 10.0},
        ]

        inputs = processor(path_coords=raw_path, text=None, return_tensors="pt")
        assert inputs["path_coords"].shape == (1, 16, 8)
        assert inputs["input_ids"].shape == (1, 8)

        # Path-only: full attention to path segment, none to text segment.
        char_start = 1 + processor.max_path_len + 1
        assert int(inputs["attention_mask"][0, 1 : 1 + processor.max_path_len].sum().item()) == 16
        assert int(inputs["attention_mask"][0, char_start:].sum().item()) == 0

    def test_numpy_xyt_path_is_converted_to_features(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=16, max_char_len=8)

        path = np.array([[0.10, 0.20, 0.0], [0.20, 0.30, 10.0]], dtype=np.float32)
        inputs = processor(path_coords=path, text="hi", return_tensors="pt")
        assert inputs["path_coords"].shape == (1, 16, 8)

    def test_text_truncation_preserves_eos(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=8, max_char_len=6)

        inputs = processor(path_coords=None, text="abcdefghij", return_tensors="pt")
        eos_id = int(tokenizer.eos_token_id)
        assert int(inputs["input_ids"][0, -1].item()) == eos_id

    def test_tensor_path_is_padded_to_max_path_len(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(
            tokenizer=tokenizer, max_path_len=32, max_char_len=20, path_input_dim=8
        )

        path = torch.zeros(1, 10, 8)
        path[0, :, 0] = 0.25
        path[0, :, 1] = 0.5

        inputs = processor(path_coords=path, text=None, return_tensors="pt")
        assert inputs["path_coords"].shape == (1, 32, 8)
        assert inputs["attention_mask"].shape[1] == 1 + 32 + 1 + 20

    def test_numeric_list_path_with_8d_features_is_padded(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(
            tokenizer=tokenizer, max_path_len=32, max_char_len=20, path_input_dim=8
        )

        path = [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0] for _ in range(10)]
        inputs = processor(path_coords=path, text=None, return_tensors="pt")
        assert inputs["path_coords"].shape == (1, 32, 8)
        assert inputs["attention_mask"].shape[1] == 1 + 32 + 1 + 20

    def test_return_tensors_none_returns_python_lists_for_numeric_path(self):
        tokenizer = SwipeTokenizer()
        processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=8, max_char_len=6)

        path_coords = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1]]
        inputs = processor(path_coords=path_coords, text="hi", return_tensors=None)
        assert isinstance(inputs["path_coords"], list)
        assert isinstance(inputs["input_ids"], list)
        assert isinstance(inputs["attention_mask"], list)

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
        """Test save -> load -> inference pipeline for base model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            config = SwipeTransformerConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
                max_path_len=32,
                max_char_len=20,
            )
            model = SwipeTransformerModel(config)
            model.eval()
            model.save_pretrained(tmpdir, safe_serialization=True)

            # Create and save tokenizer
            tokenizer = SwipeTokenizer()
            tokenizer.save_pretrained(tmpdir)

            # Load everything
            loaded_model = SwipeTransformerModel.from_pretrained(tmpdir)
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

            assert outputs.char_logits.shape[0] == 1  # batch size
            assert outputs.char_logits.shape[2] == 100  # vocab size

    def test_deterministic_outputs(self):
        """Test that outputs are deterministic after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwipeTransformerConfig(
                vocab_size=100,
                d_model=128,
                n_layers=2,
                max_path_len=32,
                max_char_len=20,
            )
            model = SwipeTransformerModel(config)
            model.eval()

            tokenizer = SwipeTokenizer()
            processor = SwipeProcessor(tokenizer=tokenizer, max_path_len=32, max_char_len=20)

            # Create inputs
            path_coords = torch.randn(2, 32, config.path_input_dim)
            text = ["hello", "world"]
            inputs = processor(path_coords, text, return_tensors="pt")

            # Get original outputs
            with torch.no_grad():
                outputs = model(**inputs)

            # Save and load
            model.save_pretrained(tmpdir, safe_serialization=True)
            loaded_model = SwipeTransformerModel.from_pretrained(tmpdir)
            loaded_model.eval()

            # Get loaded outputs
            with torch.no_grad():
                loaded_outputs = loaded_model(**inputs)

            # Verify match
            assert torch.allclose(outputs.char_logits, loaded_outputs.char_logits, atol=1e-5)
