from __future__ import annotations

import random
from typing import Any

import torch

from swipealot.text_utils import swipable_length

from ..tokenizer import CharacterTokenizer
from .utils import mask_contiguous_blocks_1d


class PairwiseMaskedCollator:
    """
    Creates asymmetric contrastive pairs with query (gradients) and key (detached).

    Two modes mixed during training:

    1. Inverted mode (default 80%):
       - Query: Heavy augmentation (0.5-0.7 masking) -> gets gradients
       - Key: Light augmentation (0.1-0.2 masking) -> detached

    2. Modality mode (default 20%):
       - Query: Text masked, path visible -> gets gradients (teaches path->representation)
       - Key: Path masked, text visible -> detached (provides target)

    Masks all tokens including EOS (only PAD and SEP are protected).
    Uses SEP token embeddings for contrastive loss.
    """

    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        mask_path: bool = True,
        modality_prob: float = 0.2,
        zero_attention_prob: float = 0.5,
        path_mask_block_max_len: int = 32,
        inverted_char_prob_heavy: float | tuple[float, float] = (0.5, 0.7),
        inverted_path_prob_heavy: float | tuple[float, float] = (0.5, 0.7),
        inverted_char_prob_light: float | tuple[float, float] = (0.1, 0.2),
        inverted_path_prob_light: float | tuple[float, float] = (0.1, 0.2),
    ):
        self.tokenizer = tokenizer
        self.mask_path = mask_path
        self.modality_prob = modality_prob
        self.zero_attention_prob = zero_attention_prob
        self.path_mask_block_max_len = path_mask_block_max_len
        self.max_char_len = None  # derived per-sample
        self.pairwise_inverted_char_prob_heavy = inverted_char_prob_heavy
        self.pairwise_inverted_path_prob_heavy = inverted_path_prob_heavy
        self.pairwise_inverted_char_prob_light = inverted_char_prob_light
        self.pairwise_inverted_path_prob_light = inverted_path_prob_light

    def _create_inverted_masks(
        self, path_coords, path_mask, char_tokens, char_mask, heavy_aug: bool
    ):
        path_len = path_coords.shape[0]
        char_len = char_tokens.shape[0]

        def _prob_from_cfg(val):
            if isinstance(val, (tuple, list)):
                return random.uniform(val[0], val[1])
            return float(val)

        if heavy_aug:
            path_mask_prob = _prob_from_cfg(self.pairwise_inverted_path_prob_heavy)
            text_mask_prob = _prob_from_cfg(self.pairwise_inverted_char_prob_heavy)
        else:
            path_mask_prob = _prob_from_cfg(self.pairwise_inverted_path_prob_light)
            text_mask_prob = _prob_from_cfg(self.pairwise_inverted_char_prob_light)

        path_mask_indices = torch.zeros(path_len, dtype=torch.long)
        if self.mask_path and path_mask_prob > 0.0:
            n_valid = int(path_mask.sum().item())
            n_to_mask = int(round(float(path_mask_prob) * n_valid))
            if n_to_mask > 0:
                path_mask_indices = mask_contiguous_blocks_1d(
                    path_mask,
                    n_to_mask,
                    max_block_len=self.path_mask_block_max_len,
                    rng=random,
                )

        char_mask_indices = torch.zeros(char_len, dtype=torch.long)
        for i in range(char_len):
            if char_mask[i] == 0:
                continue
            if random.random() < text_mask_prob:
                char_mask_indices[i] = 1

        return path_mask_indices, char_mask_indices

    def _create_modality_masks(
        self, path_coords, path_mask, char_tokens, char_mask, mask_path_modality: bool
    ):
        path_len = path_coords.shape[0]
        char_len = char_tokens.shape[0]

        if mask_path_modality:
            path_mask_indices = (
                path_mask.clone() if self.mask_path else torch.zeros(path_len, dtype=torch.long)
            )
            char_mask_indices = torch.zeros(char_len, dtype=torch.long)
        else:
            path_mask_indices = torch.zeros(path_len, dtype=torch.long)
            char_mask_indices = torch.zeros(char_len, dtype=torch.long)
            for i in range(char_len):
                if char_mask[i] == 0:
                    continue
                char_mask_indices[i] = 1

        return path_mask_indices, char_mask_indices

    def _apply_path_mask(self, path_coords, path_mask_indices):
        masked_coords = path_coords.clone()
        for i in range(len(path_mask_indices)):
            if path_mask_indices[i] == 1:
                masked_coords[i] = 0.0
        return masked_coords

    def _apply_char_mask(self, char_tokens, char_mask_indices):
        masked_tokens = char_tokens.clone()
        labels = torch.full_like(char_tokens, -100)

        for i in range(len(char_mask_indices)):
            if char_mask_indices[i] == 1:
                labels[i] = char_tokens[i]
                masked_tokens[i] = self.tokenizer.mask_token_id

        return masked_tokens, labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        views_paths = []
        views_tokens = []
        views_labels = []
        views_attention = []
        views_char_mask = []
        views_path_mask = []
        views_path_labels = []
        views_path_mask_indices = []
        pair_ids = []
        gradient_mask = []
        length_targets = []
        length_supervise_mask = []

        for pair_id, item in enumerate(batch):
            path_coords = item["path_coords"]
            path_mask = item["path_mask"]
            char_tokens = item["char_tokens"]
            char_mask = item["char_mask"]

            use_modality_mode = random.random() < self.modality_prob

            cls_mask = torch.ones(1, dtype=torch.long)
            sep_mask = torch.ones(1, dtype=torch.long)
            attn_base = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=0)

            if use_modality_mode:
                path_mask_a, char_mask_a = self._create_modality_masks(
                    path_coords, path_mask, char_tokens, char_mask, mask_path_modality=False
                )
                path_mask_b, char_mask_b = self._create_modality_masks(
                    path_coords, path_mask, char_tokens, char_mask, mask_path_modality=True
                )
                gradient_a = 1
                gradient_b = 0
            else:
                path_mask_a, char_mask_a = self._create_inverted_masks(
                    path_coords, path_mask, char_tokens, char_mask, heavy_aug=True
                )
                path_mask_b, char_mask_b = self._create_inverted_masks(
                    path_coords, path_mask, char_tokens, char_mask, heavy_aug=False
                )
                gradient_a = 1
                gradient_b = 0

            masked_path_a = self._apply_path_mask(path_coords, path_mask_a)
            masked_char_a, labels_a = self._apply_char_mask(char_tokens, char_mask_a)

            use_zero_attn = (
                use_modality_mode
                and self.zero_attention_prob > 0.0
                and random.random() < self.zero_attention_prob
            )
            if use_zero_attn:
                attn_mask_a = torch.cat(
                    [cls_mask, path_mask, sep_mask, torch.zeros_like(char_mask)], dim=0
                )
                masked_char_a = torch.full_like(char_tokens, self.tokenizer.pad_token_id)
                labels_a = torch.full_like(char_tokens, -100)
                char_mask_view_a = torch.zeros_like(char_mask)
                length_supervise = 1
            else:
                attn_mask_a = attn_base
                char_mask_view_a = char_mask
                length_supervise = 0

            views_paths.append(masked_path_a)
            views_tokens.append(masked_char_a)
            views_labels.append(labels_a)
            views_attention.append(attn_mask_a)
            views_char_mask.append(char_mask_view_a)
            views_path_mask.append(path_mask)
            views_path_labels.append(path_coords)
            views_path_mask_indices.append(path_mask_a)
            pair_ids.append(pair_id)
            gradient_mask.append(gradient_a)
            length_targets.append(swipable_length(item["word"], max_len=char_tokens.shape[0]))
            length_supervise_mask.append(length_supervise)

            masked_path_b = self._apply_path_mask(path_coords, path_mask_b)
            masked_char_b, labels_b = self._apply_char_mask(char_tokens, char_mask_b)

            if use_zero_attn and use_modality_mode:
                attn_mask_b = torch.cat(
                    [cls_mask, torch.zeros_like(path_mask), sep_mask, char_mask], dim=0
                )
                path_mask_view_b = torch.zeros_like(path_mask)
                path_mask_indices_b = torch.zeros_like(path_mask_b)
            else:
                attn_mask_b = attn_base
                path_mask_view_b = path_mask
                path_mask_indices_b = path_mask_b

            views_paths.append(masked_path_b)
            views_tokens.append(masked_char_b)
            views_labels.append(labels_b)
            views_attention.append(attn_mask_b)
            views_char_mask.append(char_mask)
            views_path_mask.append(path_mask_view_b)
            views_path_labels.append(path_coords)
            views_path_mask_indices.append(path_mask_indices_b)
            pair_ids.append(pair_id)
            gradient_mask.append(gradient_b)
            length_targets.append(swipable_length(item["word"], max_len=char_tokens.shape[0]))
            length_supervise_mask.append(0)

        result = {
            "path_coords": torch.stack(views_paths),
            "input_ids": torch.stack(views_tokens),
            "char_labels": torch.stack(views_labels),
            "attention_mask": torch.stack(views_attention),
            "char_mask": torch.stack(views_char_mask),
            "path_mask": torch.stack(views_path_mask),
            "pair_ids": torch.tensor(pair_ids, dtype=torch.long),
            "gradient_mask": torch.tensor(gradient_mask, dtype=torch.long),
            "words": [item["word"] for item in batch for _ in range(2)],
        }

        if self.mask_path:
            result["path_labels"] = torch.stack(views_path_labels)
            result["path_mask_indices"] = torch.stack(views_path_mask_indices)

        result["length_target"] = torch.tensor(length_targets, dtype=torch.long)
        result["length_supervise_mask"] = torch.tensor(length_supervise_mask, dtype=torch.long)
        return result
