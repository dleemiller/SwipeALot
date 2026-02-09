"""Processor for handling multimodal swipe inputs (path + text)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import ProcessorMixin

from ..data.preprocessing import preprocess_raw_path_to_features


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
        path_resample_mode: str = "time",
    ):
        self.tokenizer = tokenizer
        self.max_path_len = max_path_len
        self.max_char_len = max_char_len
        self.path_input_dim = path_input_dim
        self.path_resample_mode = path_resample_mode
        self.chat_template = None

    def __call__(
        self,
        path_coords: (
            list[dict[str, float]]
            | list[list[dict[str, float]]]
            | list[list[list[float]]]
            | torch.Tensor
            | np.ndarray
            | None
        ) = None,
        text: str | list[str] | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ):
        """
        Process path coordinates and text into model inputs.

        Args:
            path_coords:
                Swipe paths in one of the supported formats:
                - Raw path (single example): list of dicts like `{"x": ..., "y": ..., "t": ...}`
                - Raw batch: list of raw paths
                - Numeric arrays/tensors: `[batch, path_len, D]` or `[path_len, D]`
                If `D==3` and `path_input_dim==6`, raw `(x,y,t)` triples are converted to engineered
                `(x, y, dx, dy, ds, log_dt)` features and resampled to `max_path_len`.
                If omitted, the processor emits a zero path with a zero path attention mask.
            text:
                String or list of strings to encode.
                If omitted, the processor emits padded text tokens with a zero text attention mask.
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
                - attention_mask: [batch, total_seq_len] (covers `[CLS] + path + [SEP] + text`)
        """
        if path_coords is None and text is None:
            raise ValueError("Must provide either path_coords or text (or both)")

        batch_size, path_coords, text = self._infer_batch_size(path_coords, text)

        result: dict[str, Any] = {}

        path_coords_out, path_mask = self._process_path_coords(
            path_coords=path_coords,
            batch_size=batch_size,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
        result["path_coords"] = path_coords_out

        input_ids, char_mask = self._process_text(
            text=text,
            batch_size=batch_size,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )
        result["input_ids"] = input_ids

        result["attention_mask"] = self._build_attention_mask(
            path_mask=path_mask,
            char_mask=char_mask,
            batch_size=batch_size,
            return_tensors=return_tensors,
        )

        self._convert_result_in_place(result, return_tensors=return_tensors)
        return result

    def _infer_batch_size(
        self,
        path_coords: (
            list[dict[str, float]]
            | list[list[dict[str, float]]]
            | list[list[list[float]]]
            | torch.Tensor
            | np.ndarray
            | None
        ),
        text: str | list[str] | None,
    ) -> tuple[int, Any, str | list[str] | None]:
        if path_coords is not None:
            if isinstance(path_coords, (list, tuple)):
                if len(path_coords) == 0:
                    batch_size = 1
                else:
                    first = path_coords[0]
                    if isinstance(first, dict):
                        batch_size = 1
                    elif (
                        isinstance(first, (list, tuple))
                        and len(first) > 0
                        and isinstance(first[0], dict)
                    ):
                        batch_size = len(path_coords)
                    elif (
                        isinstance(first, (list, tuple))
                        and len(first) > 0
                        and isinstance(first[0], (list, tuple))
                    ):
                        path_coords = torch.tensor(path_coords, dtype=torch.float32)
                        batch_size = int(path_coords.shape[0])
                    else:
                        path_coords = torch.tensor([path_coords], dtype=torch.float32)
                        batch_size = int(path_coords.shape[0])
            elif isinstance(path_coords, np.ndarray):
                path_coords = torch.from_numpy(path_coords).float()
                if path_coords.dim() == 2:
                    path_coords = path_coords.unsqueeze(0)
                batch_size = int(path_coords.shape[0])
            elif isinstance(path_coords, torch.Tensor):
                if path_coords.dim() == 2:
                    path_coords = path_coords.unsqueeze(0)
                batch_size = int(path_coords.shape[0])
            else:
                batch_size = 1
        elif text is not None:
            if isinstance(text, str):
                batch_size = 1
                text = [text]
            else:
                batch_size = len(text)
        else:
            batch_size = 1

        return batch_size, path_coords, text

    def _process_path_coords(
        self,
        *,
        path_coords,
        batch_size: int,
        truncation: bool,
        padding: bool | str,
        return_tensors: str | None,
    ) -> tuple[Any, Any]:
        if path_coords is None:
            path_coords_out = torch.zeros(batch_size, self.max_path_len, self.path_input_dim)
            path_mask = torch.zeros(batch_size, self.max_path_len, dtype=torch.long)
            return path_coords_out, path_mask

        if isinstance(path_coords, (list, tuple)) and len(path_coords) > 0:
            first_elem = path_coords[0]

            if isinstance(first_elem, dict) and "x" in first_elem:
                path_feats, mask = preprocess_raw_path_to_features(
                    path_coords,
                    self.max_path_len,
                    resample_mode=self.path_resample_mode,
                )
                if return_tensors == "pt":
                    return (
                        torch.from_numpy(path_feats).float().unsqueeze(0),
                        torch.from_numpy(mask).long().unsqueeze(0),
                    )
                return (np.expand_dims(path_feats, axis=0), np.expand_dims(mask, axis=0))

            if (
                isinstance(first_elem, (list, tuple))
                and len(first_elem) > 0
                and isinstance(first_elem[0], dict)
                and "x" in first_elem[0]
            ):
                processed_paths = []
                path_masks = []
                for path in path_coords:
                    path_feats, mask = preprocess_raw_path_to_features(
                        path,
                        self.max_path_len,
                        resample_mode=self.path_resample_mode,
                    )
                    processed_paths.append(path_feats)
                    path_masks.append(mask)

                path_coords_np = np.stack(processed_paths)
                path_mask_np = np.stack(path_masks)
                if return_tensors == "pt":
                    return torch.from_numpy(path_coords_np).float(), torch.from_numpy(
                        path_mask_np
                    ).long()
                return path_coords_np, path_mask_np

            # Numeric list input
            path_tensor = torch.tensor(path_coords, dtype=torch.float32)
            if path_tensor.dim() == 2:
                path_tensor = path_tensor.unsqueeze(0)

            current_path_len = int(path_tensor.shape[1])
            if truncation and current_path_len > self.max_path_len:
                path_tensor = path_tensor[:, : self.max_path_len, :]
            if padding and current_path_len < self.max_path_len:
                pad_len = self.max_path_len - current_path_len
                pad_shape = (batch_size, pad_len, self.path_input_dim)
                path_tensor = torch.cat([path_tensor, torch.zeros(pad_shape)], dim=1)

            path_mask = torch.ones(batch_size, self.max_path_len, dtype=torch.long)
            is_padding = (path_tensor == 0).all(dim=-1)
            path_mask[is_padding] = 0
            return path_tensor, path_mask

        if isinstance(path_coords, np.ndarray):
            path_coords = torch.from_numpy(path_coords).float()

        if isinstance(path_coords, torch.Tensor):
            if path_coords.dim() == 2:
                path_coords = path_coords.unsqueeze(0)
            if path_coords.shape[-1] == 3 and self.path_input_dim == 6:
                processed_paths = []
                path_masks = []
                for path in path_coords.detach().cpu().numpy():
                    raw = [{"x": float(p[0]), "y": float(p[1]), "t": float(p[2])} for p in path]
                    path_feats, mask = preprocess_raw_path_to_features(
                        raw,
                        self.max_path_len,
                        resample_mode=self.path_resample_mode,
                    )
                    processed_paths.append(path_feats)
                    path_masks.append(mask)
                return torch.from_numpy(np.stack(processed_paths)).float(), torch.from_numpy(
                    np.stack(path_masks)
                ).long()

            if int(path_coords.shape[-1]) != int(self.path_input_dim):
                raise ValueError(
                    f"Expected path_coords.shape[-1] == path_input_dim ({self.path_input_dim}), "
                    f"got {int(path_coords.shape[-1])}. If your path is (x,y,t), pass D=3."
                )

            path_tensor = path_coords
            current_path_len = int(path_tensor.shape[1])
            if truncation and current_path_len > self.max_path_len:
                path_tensor = path_tensor[:, : self.max_path_len, :]
            if padding and current_path_len < self.max_path_len:
                pad_len = self.max_path_len - current_path_len
                pad_shape = (int(path_tensor.shape[0]), pad_len, int(path_tensor.shape[-1]))
                pad = torch.zeros(pad_shape, dtype=path_tensor.dtype, device=path_tensor.device)
                path_tensor = torch.cat([path_tensor, pad], dim=1)

            path_mask = torch.ones(
                int(path_tensor.shape[0]),
                int(path_tensor.shape[1]),
                dtype=torch.long,
                device=path_tensor.device,
            )
            is_padding = (path_tensor == 0).all(dim=-1)
            path_mask[is_padding] = 0
            return path_tensor, path_mask

        # Fallback: treat unknown input as empty path.
        path_coords_out = torch.zeros(batch_size, self.max_path_len, self.path_input_dim)
        path_mask = torch.zeros(batch_size, self.max_path_len, dtype=torch.long)
        return path_coords_out, path_mask

    def _process_text(
        self,
        *,
        text: str | list[str] | None,
        batch_size: int,
        padding: bool | str,
        truncation: bool,
        max_length: int | None,
        return_tensors: str | None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if text is None:
            if return_tensors == "pt":
                char_tokens = torch.full(
                    (batch_size, self.max_char_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                char_mask = torch.zeros(batch_size, self.max_char_len, dtype=torch.long)
            elif return_tensors == "np":
                char_tokens = np.full(
                    (batch_size, self.max_char_len),
                    self.tokenizer.pad_token_id,
                    dtype=np.int64,
                )
                char_mask = np.zeros((batch_size, self.max_char_len), dtype=np.int64)
            else:
                char_tokens = [
                    [self.tokenizer.pad_token_id] * self.max_char_len for _ in range(batch_size)
                ]
                char_mask = [[0] * self.max_char_len for _ in range(batch_size)]
            return char_tokens, char_mask

        if isinstance(text, str):
            text = [text]

        text_max_length = max_length if max_length is not None else self.max_char_len

        encoded_raw = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors=None,
            **kwargs,
        )

        eos_id = self.tokenizer.eos_token_id
        for i in range(len(encoded_raw["input_ids"])):
            if encoded_raw["input_ids"][i][-1] != eos_id:
                encoded_raw["input_ids"][i].append(eos_id)

        max_len_needed = max(len(ids) for ids in encoded_raw["input_ids"])
        if truncation and max_len_needed > text_max_length:
            for i in range(len(encoded_raw["input_ids"])):
                if len(encoded_raw["input_ids"][i]) > text_max_length:
                    encoded_raw["input_ids"][i] = encoded_raw["input_ids"][i][
                        : text_max_length - 1
                    ] + [eos_id]

        if padding:
            pad_id = self.tokenizer.pad_token_id
            for i in range(len(encoded_raw["input_ids"])):
                seq_len = len(encoded_raw["input_ids"][i])
                if seq_len < text_max_length:
                    encoded_raw["input_ids"][i].extend([pad_id] * (text_max_length - seq_len))

        char_mask_list = [
            [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in ids]
            for ids in encoded_raw["input_ids"]
        ]

        if return_tensors == "pt":
            return (
                torch.tensor(encoded_raw["input_ids"], dtype=torch.long),
                torch.tensor(char_mask_list, dtype=torch.long),
            )
        if return_tensors == "np":
            return (
                np.array(encoded_raw["input_ids"], dtype=np.int64),
                np.array(char_mask_list, dtype=np.int64),
            )
        return encoded_raw["input_ids"], char_mask_list

    def _build_attention_mask(
        self,
        *,
        path_mask,
        char_mask,
        batch_size: int,
        return_tensors: str | None,
    ):
        if return_tensors == "pt":
            cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
            sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
            return torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=1)
        if return_tensors == "np":
            cls_mask = np.ones((batch_size, 1), dtype=np.int64)
            sep_mask = np.ones((batch_size, 1), dtype=np.int64)
            return np.concatenate([cls_mask, path_mask, sep_mask, char_mask], axis=1)

        cls_mask = [[1] for _ in range(batch_size)]
        sep_mask = [[1] for _ in range(batch_size)]
        return [
            cls + path.tolist() + sep + char
            for cls, path, sep, char in zip(cls_mask, path_mask, sep_mask, char_mask, strict=False)
        ]

    def _convert_result_in_place(
        self, result: dict[str, Any], *, return_tensors: str | None
    ) -> None:
        if return_tensors == "np":
            for key, value in list(result.items()):
                if isinstance(value, torch.Tensor):
                    result[key] = value.numpy()
        elif return_tensors is None:
            for key, value in list(result.items()):
                if isinstance(value, torch.Tensor):
                    result[key] = value.tolist()

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

    def encode_path(self, path_coords, *, return_tensors: str | None = "pt", **kwargs: Any):
        """Create model inputs from a swipe path only (no text)."""
        return self(path_coords=path_coords, text=None, return_tensors=return_tensors, **kwargs)

    def encode_text(self, text, *, return_tensors: str | None = "pt", **kwargs: Any):
        """Create model inputs from text only (no path)."""
        return self(path_coords=None, text=text, return_tensors=return_tensors, **kwargs)

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
