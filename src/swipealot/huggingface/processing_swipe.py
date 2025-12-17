"""Processor for handling multimodal swipe inputs (path + text)."""

import numpy as np
import torch
from transformers import ProcessorMixin


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

    def __init__(self, tokenizer=None, max_path_len: int = 64, max_char_len: int = 38):
        self.tokenizer = tokenizer
        self.max_path_len = max_path_len
        self.max_char_len = max_char_len
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
            path_coords: List of paths or tensor [batch, path_len, 3]
                        Each point is (x, y, time). Can be None if only processing text.
            text: String or list of strings to encode. Can be None if only processing paths.
            padding: Whether to pad sequences. Can be True/False or "max_length"
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length for text (overrides max_char_len)
            return_tensors: "pt" for PyTorch, "np" for NumPy, None for lists
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with:
                - path_coords: [batch, max_path_len, 3] (if path_coords provided)
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
            current_path_len = path_coords.shape[1]

            # Truncate if needed
            if truncation and current_path_len > self.max_path_len:
                path_coords = path_coords[:, : self.max_path_len, :]
                current_path_len = self.max_path_len

            # Pad if needed
            if padding and current_path_len < self.max_path_len:
                pad_len = self.max_path_len - current_path_len
                path_coords = torch.cat([path_coords, torch.zeros(batch_size, pad_len, 3)], dim=1)

            # Create path mask (1 = real data, 0 = padding)
            # Detect padding by checking for all-zero coordinates
            path_mask = torch.ones(batch_size, self.max_path_len, dtype=torch.long)
            # A point is padding if all its coordinates (x, y, t) are zero
            is_padding = (path_coords == 0).all(dim=-1)  # [batch, path_len]
            path_mask[is_padding] = 0

            result["path_coords"] = path_coords
            # Store path_mask internally for attention_mask construction
            _path_mask = path_mask
        else:
            # No path coords provided, create empty/zero tensors
            path_coords = torch.zeros(batch_size, self.max_path_len, 3)
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

    def normalize_coordinates(
        self, data_points: list[dict], canvas_width: float = None, canvas_height: float = None
    ) -> list[dict]:
        """
        Normalize swipe coordinates and timestamps.

        Args:
            data_points: List of dicts with 'x', 'y', 't' keys
            canvas_width: Canvas width (not used - kept for compatibility)
            canvas_height: Canvas height (not used - kept for compatibility)

        Returns:
            List of normalized coordinate dicts with x, y in [0,1] and t in [0,1]

        Note:
            For futo-org/swipe.futo.org dataset, x and y are already normalized to [0,1].
            This function clamps them to ensure they stay in bounds and normalizes timestamps.
        """
        if not data_points:
            return []

        # Extract timestamps for normalization
        timestamps = [p["t"] for p in data_points]
        t_min = min(timestamps)
        t_max = max(timestamps)
        t_range = t_max - t_min if t_max > t_min else 1.0

        normalized = []
        for point in data_points:
            # x and y are already normalized to [0,1] in the dataset
            # But sometimes they go slightly outside bounds, so clamp them
            x_norm = max(0.0, min(1.0, point["x"]))
            y_norm = max(0.0, min(1.0, point["y"]))

            # Normalize timestamp to [0, 1]
            t_norm = (point["t"] - t_min) / t_range

            normalized.append({"x": x_norm, "y": y_norm, "t": t_norm})

        return normalized

    def sample_path_points(self, data_points: list[dict], max_len: int = None) -> tuple:
        """
        Sample or pad path points to fixed length using linear interpolation.

        Args:
            data_points: List of coordinate dicts with 'x', 'y', 't' keys
            max_len: Target length (defaults to self.max_path_len if not specified)

        Returns:
            Tuple of (sampled_points, mask) where:
            - sampled_points: numpy array of shape [max_len, 3] with (x, y, t) coordinates
            - mask: numpy array of shape [max_len] indicating valid (1) vs padding (0) points

        Note:
            - If path has fewer points than max_len, it's zero-padded
            - If path has more points than max_len, it's downsampled using linear interpolation
            - If path has exactly max_len points, it's returned as-is
        """
        if max_len is None:
            max_len = self.max_path_len

        num_points = len(data_points)

        if num_points == max_len:
            points = data_points
            mask = [1] * max_len
        elif num_points < max_len:
            # Pad with zeros
            points = data_points + [{"x": 0.0, "y": 0.0, "t": 0.0}] * (max_len - num_points)
            mask = [1] * num_points + [0] * (max_len - num_points)
        else:
            # Downsample using linear interpolation
            # Extract coordinates as arrays
            x_coords = np.array([p["x"] for p in data_points])
            y_coords = np.array([p["y"] for p in data_points])
            t_coords = np.array([p["t"] for p in data_points])

            # Original indices (parameter for interpolation)
            original_indices = np.arange(num_points)

            # Target indices for interpolation (evenly spaced)
            target_indices = np.linspace(0, num_points - 1, max_len)

            # Interpolate each coordinate independently
            x_interp = np.interp(target_indices, original_indices, x_coords)
            y_interp = np.interp(target_indices, original_indices, y_coords)
            t_interp = np.interp(target_indices, original_indices, t_coords)

            # Reconstruct points
            points = [
                {"x": float(x), "y": float(y), "t": float(t)}
                for x, y, t in zip(x_interp, y_interp, t_interp, strict=True)
            ]
            mask = [1] * max_len

        # Convert to numpy arrays
        coords = np.array([[p["x"], p["y"], p["t"]] for p in points], dtype=np.float32)
        mask = np.array(mask, dtype=np.int64)

        return coords, mask

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
                with open(processor_config_path, "r") as f:
                    config = json.load(f)

                config["auto_map"] = {
                    "AutoProcessor": "processing_swipe.SwipeProcessor"
                }

                with open(processor_config_path, "w") as f:
                    json.dump(config, f, indent=2)
                break

        return result
