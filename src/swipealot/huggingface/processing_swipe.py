"""Processor for handling multimodal swipe inputs (path + text)."""

import numpy as np
import torch
from transformers import ProcessorMixin

from ..data.preprocessing import (
    normalize_and_compute_features,
    sample_path_points_with_features,
)


class SwipeProcessor(ProcessorMixin):
    """
    Processor for handling multimodal swipe inputs (path coordinates + text).

    This processor combines path coordinate preprocessing with text tokenization,
    creating the inputs needed for SwipeTransformer models.

    Args:
        tokenizer: SwipeTokenizer instance
        max_path_len (int): Maximum path length. Defaults to 64.
        max_char_len (int): Maximum character length. Defaults to 38.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"  # Will use auto_map from tokenizer_config.json

    def __init__(
        self,
        tokenizer=None,
        max_path_len: int = 64,
        max_char_len: int = 38,
        path_input_dim: int = 6,
    ):
        self.tokenizer = tokenizer
        self.max_path_len = max_path_len
        self.max_char_len = max_char_len
        self.path_input_dim = path_input_dim
        # Attributes expected by newer transformers (not used for swipe models)
        self.chat_template = None
        self.audio_tokenizer = None
        self.feature_extractor = None
        self.image_processor = None

    def __call__(
        self,
        path_coords: list[list[list[float]]] | torch.Tensor | np.ndarray | None = None,
        text: str | list[str] | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ):
        """
        Process path coordinates and text into model inputs.

        Args:
            path_coords: List of paths (raw {"x", "y", "t"} dicts) or tensor [batch, path_len, path_input_dim].
                        Raw paths will be preprocessed to compute motion features.
                        Can be None if only processing text.
            text: String or list of strings to encode. Can be None if only processing paths.
            padding: Whether to pad sequences. Can be True/False or "max_length"
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length for text (overrides max_char_len)
            return_tensors: "pt" for PyTorch, "np" for NumPy, None for lists
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with:
                - path_coords: [batch, max_path_len, path_input_dim] (if path_coords provided)
                  Default: [batch, max_path_len, 6] for (x, y, dx, dy, ds, log_dt)
                - input_ids: [batch, max_char_len] (if text provided)
                - attention_mask: [batch, total_seq_len]
        """
        if path_coords is None and text is None:
            raise ValueError("Must provide either path_coords or text (or both)")

        # Determine batch size
        if path_coords is not None:
            # Handle path coordinates
            if isinstance(path_coords, (list, tuple)):
                # Check if it's a batch or single path
                if len(path_coords) > 0 and isinstance(path_coords[0][0], (list, tuple)):
                    # Batch of paths [[path1], [path2], ...]
                    path_coords = torch.tensor(path_coords, dtype=torch.float32)
                else:
                    # Single path [[x,y,t], [x,y,t], ...]
                    path_coords = torch.tensor([path_coords], dtype=torch.float32)
            elif isinstance(path_coords, np.ndarray):
                path_coords = torch.from_numpy(path_coords).float()
                if path_coords.dim() == 2:
                    # Single path, add batch dimension
                    path_coords = path_coords.unsqueeze(0)
            elif isinstance(path_coords, torch.Tensor):
                if path_coords.dim() == 2:
                    # Single path, add batch dimension
                    path_coords = path_coords.unsqueeze(0)

            batch_size = path_coords.shape[0]
        elif text is not None:
            if isinstance(text, str):
                batch_size = 1
                text = [text]
            else:
                batch_size = len(text)
        else:
            batch_size = 1

        result = {}

        # Process path coordinates
        if path_coords is not None:
            # Check if path_coords is raw data (list of dicts) or already a tensor
            if isinstance(path_coords, (list, tuple)) and len(path_coords) > 0:
                # Check if it's raw path data (list of {"x", "y", "t"} dicts)
                first_elem = path_coords[0]
                if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                    first_point = first_elem[0]
                    if isinstance(first_point, dict) and "x" in first_point:
                        # Raw path data - need to preprocess
                        processed_paths = []
                        path_masks = []
                        for path in path_coords:
                            # Compute motion features
                            features = normalize_and_compute_features(path)
                            # Spatially uniform resampling to max_path_len
                            path_feats, mask = sample_path_points_with_features(
                                features, self.max_path_len
                            )
                            processed_paths.append(path_feats)
                            path_masks.append(mask)

                        path_coords = np.stack(processed_paths)  # [batch, max_path_len, 6]
                        _path_mask = np.stack(path_masks)  # [batch, max_path_len]

                        # Convert to tensor if needed
                        if return_tensors == "pt":
                            path_coords = torch.from_numpy(path_coords).float()
                            _path_mask = torch.from_numpy(_path_mask).long()
                    else:
                        # Already numeric data, process as before
                        path_coords = torch.tensor(path_coords, dtype=torch.float32)
                        if path_coords.dim() == 2:
                            path_coords = path_coords.unsqueeze(0)

                        current_path_len = path_coords.shape[1]
                        if truncation and current_path_len > self.max_path_len:
                            path_coords = path_coords[:, : self.max_path_len, :]
                        if padding and current_path_len < self.max_path_len:
                            pad_len = self.max_path_len - current_path_len
                            pad_shape = (batch_size, pad_len, self.path_input_dim)
                            path_coords = torch.cat([path_coords, torch.zeros(pad_shape)], dim=1)

                        # Create mask
                        _path_mask = torch.ones(batch_size, self.max_path_len, dtype=torch.long)
                        is_padding = (path_coords == 0).all(dim=-1)
                        _path_mask[is_padding] = 0
                else:
                    # Single path, wrap in batch
                    path_coords = torch.tensor([path_coords], dtype=torch.float32)
                    _path_mask = torch.ones(1, self.max_path_len, dtype=torch.long)
            elif isinstance(path_coords, np.ndarray):
                path_coords = torch.from_numpy(path_coords).float()
                if path_coords.dim() == 2:
                    path_coords = path_coords.unsqueeze(0)
                # If user provided raw (x,y,t) triples but model expects engineered features,
                # convert to motion features and resample.
                if path_coords.shape[-1] == 3 and self.path_input_dim == 6:
                    processed_paths = []
                    path_masks = []
                    for path in path_coords.cpu().numpy():
                        raw = [{"x": float(p[0]), "y": float(p[1]), "t": float(p[2])} for p in path]
                        features = normalize_and_compute_features(raw)
                        path_feats, mask = sample_path_points_with_features(
                            features, self.max_path_len
                        )
                        processed_paths.append(path_feats)
                        path_masks.append(mask)

                    path_coords = torch.from_numpy(np.stack(processed_paths)).float()
                    _path_mask = torch.from_numpy(np.stack(path_masks)).long()
                else:
                    _path_mask = torch.ones(
                        path_coords.shape[0], self.max_path_len, dtype=torch.long
                    )
            elif isinstance(path_coords, torch.Tensor):
                if path_coords.dim() == 2:
                    path_coords = path_coords.unsqueeze(0)
                # If user provided raw (x,y,t) triples but model expects engineered features,
                # convert to motion features and resample.
                if path_coords.shape[-1] == 3 and self.path_input_dim == 6:
                    processed_paths = []
                    path_masks = []
                    for path in path_coords.detach().cpu().numpy():
                        raw = [{"x": float(p[0]), "y": float(p[1]), "t": float(p[2])} for p in path]
                        features = normalize_and_compute_features(raw)
                        path_feats, mask = sample_path_points_with_features(
                            features, self.max_path_len
                        )
                        processed_paths.append(path_feats)
                        path_masks.append(mask)

                    path_coords = torch.from_numpy(np.stack(processed_paths)).float()
                    _path_mask = torch.from_numpy(np.stack(path_masks)).long()
                else:
                    _path_mask = torch.ones(
                        path_coords.shape[0], self.max_path_len, dtype=torch.long
                    )

            result["path_coords"] = path_coords
        else:
            # No path coords provided, create empty/zero tensors
            path_coords = torch.zeros(batch_size, self.max_path_len, self.path_input_dim)
            _path_mask = torch.zeros(batch_size, self.max_path_len, dtype=torch.long)
            result["path_coords"] = path_coords

        # Process text
        if text is not None:
            # Ensure text is a list
            if isinstance(text, str):
                text = [text]

            # Tokenize text
            text_max_length = max_length if max_length is not None else self.max_char_len

            # First tokenize without padding/truncation to add EOS
            encoded_raw = self.tokenizer(
                text,
                padding=False,
                truncation=False,
                return_tensors=None,  # Get lists first
                **kwargs,
            )

            # Add EOS token after each word (matching training dataset behavior)
            eos_id = self.tokenizer.eos_token_id
            for i in range(len(encoded_raw["input_ids"])):
                # Add EOS if not already present
                if encoded_raw["input_ids"][i][-1] != eos_id:
                    encoded_raw["input_ids"][i].append(eos_id)

            # Now apply padding and truncation
            max_len_needed = max(len(ids) for ids in encoded_raw["input_ids"])
            if truncation and max_len_needed > text_max_length:
                # Truncate but preserve EOS at the end
                for i in range(len(encoded_raw["input_ids"])):
                    if len(encoded_raw["input_ids"][i]) > text_max_length:
                        encoded_raw["input_ids"][i] = encoded_raw["input_ids"][i][
                            : text_max_length - 1
                        ] + [eos_id]

            # Pad sequences
            if padding:
                pad_id = self.tokenizer.pad_token_id
                for i in range(len(encoded_raw["input_ids"])):
                    seq_len = len(encoded_raw["input_ids"][i])
                    if seq_len < text_max_length:
                        encoded_raw["input_ids"][i].extend([pad_id] * (text_max_length - seq_len))

            # Create attention mask (1 for real tokens + EOS, 0 for padding)
            _char_mask = []
            for ids in encoded_raw["input_ids"]:
                mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in ids]
                _char_mask.append(mask)

            # Convert to tensors if requested
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor(encoded_raw["input_ids"], dtype=torch.long)
                _char_mask = torch.tensor(_char_mask, dtype=torch.long)
            elif return_tensors == "np":
                result["input_ids"] = np.array(encoded_raw["input_ids"], dtype=np.int64)
                _char_mask = np.array(_char_mask, dtype=np.int64)
            else:
                result["input_ids"] = encoded_raw["input_ids"]
        else:
            # No text provided, create padding tokens
            if return_tensors == "pt":
                char_tokens = torch.full(
                    (batch_size, self.max_char_len), self.tokenizer.pad_token_id, dtype=torch.long
                )
                _char_mask = torch.zeros(batch_size, self.max_char_len, dtype=torch.long)
            elif return_tensors == "np":
                char_tokens = np.full(
                    (batch_size, self.max_char_len), self.tokenizer.pad_token_id, dtype=np.int64
                )
                _char_mask = np.zeros((batch_size, self.max_char_len), dtype=np.int64)
            else:
                char_tokens = [
                    [self.tokenizer.pad_token_id] * self.max_char_len for _ in range(batch_size)
                ]
                _char_mask = [[0] * self.max_char_len for _ in range(batch_size)]

            result["input_ids"] = char_tokens

        # Create combined attention mask: [CLS] + path + [SEP] + chars
        # Sequence structure: [CLS:1] + _path_mask + [SEP:1] + _char_mask
        if return_tensors == "pt":
            cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
            sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
            attention_mask = torch.cat([cls_mask, _path_mask, sep_mask, _char_mask], dim=1)
        elif return_tensors == "np":
            cls_mask = np.ones((batch_size, 1), dtype=np.int64)
            sep_mask = np.ones((batch_size, 1), dtype=np.int64)
            attention_mask = np.concatenate([cls_mask, _path_mask, sep_mask, _char_mask], axis=1)
        else:
            cls_mask = [[1] for _ in range(batch_size)]
            sep_mask = [[1] for _ in range(batch_size)]
            attention_mask = [
                cls + path.tolist() + sep + char
                for cls, path, sep, char in zip(
                    cls_mask, _path_mask, sep_mask, _char_mask, strict=False
                )
            ]

        result["attention_mask"] = attention_mask

        # Convert to requested format
        if return_tensors == "np":
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].numpy()
        elif return_tensors is None:
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].tolist()

        return result

    def batch_decode(self, token_ids, **kwargs):
        """
        Decode token IDs to strings.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(token_ids, **kwargs)

    def decode(self, token_ids, **kwargs):
        """
        Decode single sequence of token IDs to string.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    # Preprocessing methods are now imported from shared preprocessing module
    # See src/swipealot/data/preprocessing.py for the implementation

    def save_pretrained(
        self,
        save_directory,
        push_to_hub=False,
        **kwargs,
    ):
        """
        Save the processor to a directory, ensuring auto_map is included.
        """
        # Call parent save_pretrained
        result = super().save_pretrained(
            save_directory,
            push_to_hub=push_to_hub,
            **kwargs,
        )

        # Add auto_map to processor_config.json for AutoProcessor compatibility
        import json
        from pathlib import Path

        # Try both possible config file names
        for config_name in ["preprocessor_config.json", "processor_config.json"]:
            processor_config_path = Path(save_directory) / config_name
            if processor_config_path.exists():
                with open(processor_config_path) as f:
                    config = json.load(f)

                config["auto_map"] = {"AutoProcessor": "processing_swipe.SwipeProcessor"}

                with open(processor_config_path, "w") as f:
                    json.dump(config, f, indent=2)
                break

        return result
