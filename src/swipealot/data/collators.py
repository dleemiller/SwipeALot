"""Data collators for masked language modeling and contrastive learning."""

import random
from typing import Any

import torch

from .tokenizer import CharacterTokenizer


class MaskedCollator:
    """
    Collator that creates masked versions of characters and paths for MLM-style training.
    """

    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        char_mask_prob: float = 0.15,
        path_mask_prob: float = 0.15,
        mask_path: bool = True,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Character tokenizer for masking
            char_mask_prob: Probability of masking each character
            path_mask_prob: Probability of masking each path point
            mask_path: Whether to mask path points
        """
        self.tokenizer = tokenizer
        self.char_mask_prob = char_mask_prob
        self.path_mask_prob = path_mask_prob
        self.mask_path = mask_path

    def mask_characters(self, char_tokens: torch.Tensor, char_mask: torch.Tensor) -> tuple:
        """
        Mask character tokens following BERT strategy (vectorized).

        Args:
            char_tokens: [batch, seq_len] character token IDs
            char_mask: [batch, seq_len] mask indicating valid tokens (1) vs padding (0)

        Returns:
            Tuple of (masked_tokens, labels)
            - masked_tokens: [batch, seq_len] with some tokens masked
            - labels: [batch, seq_len] with -100 for non-masked positions
        """
        batch_size, seq_len = char_tokens.shape
        masked_tokens = char_tokens.clone()
        labels = torch.full_like(char_tokens, -100)  # -100 is ignored by CrossEntropyLoss

        # Vectorized masking: create random decisions for all tokens
        mask_decisions = torch.rand(batch_size, seq_len) < self.char_mask_prob
        mask_decisions = mask_decisions & (char_mask == 1)  # Only mask valid tokens

        # For tokens we decided to mask, store original as label
        labels[mask_decisions] = char_tokens[mask_decisions]

        # BERT masking strategy (vectorized)
        # For each token to mask, decide: 80% MASK, 10% random, 10% keep
        bert_probs = torch.rand(batch_size, seq_len)

        # 80%: Replace with [MASK]
        use_mask_token = mask_decisions & (bert_probs < 0.8)
        masked_tokens[use_mask_token] = self.tokenizer.mask_token_id

        # 10%: Replace with random token
        use_random_token = mask_decisions & (bert_probs >= 0.8) & (bert_probs < 0.9)
        num_random = use_random_token.sum().item()
        if num_random > 0:
            random_tokens = torch.randint(
                len(self.tokenizer.special_tokens),
                self.tokenizer.vocab_size,
                (num_random,),
                dtype=char_tokens.dtype,
            )
            masked_tokens[use_random_token] = random_tokens

        # 10%: Keep original (no action needed)

        return masked_tokens, labels

    def mask_path_points(self, path_coords: torch.Tensor, path_mask: torch.Tensor) -> tuple:
        """
        Mask path coordinates by replacing with zeros (vectorized).

        Args:
            path_coords: [batch, seq_len, 3] path coordinates
            path_mask: [batch, seq_len] mask indicating valid points (1) vs padding (0)

        Returns:
            Tuple of (masked_coords, labels, mask_indices)
            - masked_coords: [batch, seq_len, 3] with some points zeroed
            - labels: [batch, seq_len, 3] original coordinates (only for masked points)
            - mask_indices: [batch, seq_len] binary mask (1 = was masked, 0 = not masked)
        """
        batch_size, seq_len, _ = path_coords.shape
        masked_coords = path_coords.clone()
        labels = path_coords.clone()

        # Vectorized masking: create random decisions for all points
        mask_decisions = torch.rand(batch_size, seq_len) < self.path_mask_prob
        mask_decisions = mask_decisions & (path_mask == 1)  # Only mask valid points

        # Zero out masked coordinates
        mask_indices = mask_decisions.long()
        masked_coords[mask_decisions] = 0.0

        return masked_coords, labels, mask_indices

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate batch and apply masking.

        Args:
            batch: List of samples from dataset

        Returns:
            Dictionary with batched and masked tensors
        """
        # Stack tensors
        path_coords = torch.stack([item["path_coords"] for item in batch])  # [B, path_len, 3]
        char_tokens = torch.stack([item["char_tokens"] for item in batch])  # [B, char_len]
        path_mask = torch.stack([item["path_mask"] for item in batch])  # [B, path_len]
        char_mask = torch.stack([item["char_mask"] for item in batch])  # [B, char_len]

        # Apply character masking
        masked_char_tokens, char_labels = self.mask_characters(char_tokens, char_mask)

        # Apply path masking (if enabled)
        if self.mask_path:
            masked_path_coords, path_labels, path_mask_indices = self.mask_path_points(
                path_coords, path_mask
            )
        else:
            masked_path_coords = path_coords
            path_labels = None
            path_mask_indices = None

        # Create combined attention mask for full sequence: [CLS] + path + [SEP] + chars
        batch_size = path_coords.shape[0]
        cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
        sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
        attention_mask = torch.cat(
            [cls_mask, path_mask, sep_mask, char_mask], dim=1
        )  # [B, 1+path_len+1+char_len]

        result = {
            "path_coords": masked_path_coords,
            "char_tokens": masked_char_tokens,
            "char_labels": char_labels,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "words": [item["word"] for item in batch],  # Original words for evaluation
        }

        if self.mask_path:
            result["path_labels"] = path_labels
            result["path_mask_indices"] = path_mask_indices

        return result


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
        self, tokenizer: CharacterTokenizer, mask_path: bool = True, modality_prob: float = 0.2
    ):
        """
        Args:
            tokenizer: Character tokenizer
            mask_path: Whether to mask path coordinates
            modality_prob: Probability of using modality-based masking (vs inverted)
        """
        self.tokenizer = tokenizer
        self.mask_path = mask_path
        self.modality_prob = modality_prob

    def _create_inverted_masks(
        self, path_coords, path_mask, char_tokens, char_mask, heavy_aug: bool
    ):
        """
        Create random masks for path and text tokens (inverted mode).

        Args:
            heavy_aug: If True, use heavy augmentation (0.5-0.7). If False, use light (0.1-0.2)

        Returns: (path_mask_indices, char_mask_indices) where 1 = masked, 0 = not masked
        """
        path_len = path_coords.shape[0]
        char_len = char_tokens.shape[0]

        # Random masking probabilities based on augmentation strength
        if heavy_aug:
            path_mask_prob = random.uniform(0.5, 0.7)
            text_mask_prob = random.uniform(0.5, 0.7)
        else:
            path_mask_prob = random.uniform(0.1, 0.2)
            text_mask_prob = random.uniform(0.1, 0.2)

        # Create path mask
        path_mask_indices = torch.zeros(path_len, dtype=torch.long)
        if self.mask_path:
            for i in range(path_len):
                if path_mask[i] == 1 and random.random() < path_mask_prob:
                    path_mask_indices[i] = 1

        # Create character mask (exclude PAD only, allow EOS to be masked)
        char_mask_indices = torch.zeros(char_len, dtype=torch.long)
        for i in range(char_len):
            # Skip PAD tokens only
            if char_mask[i] == 0:  # padding
                continue
            # Apply random masking (including EOS)
            if random.random() < text_mask_prob:
                char_mask_indices[i] = 1

        return path_mask_indices, char_mask_indices

    def _create_modality_masks(
        self, path_coords, path_mask, char_tokens, char_mask, mask_path_modality: bool
    ):
        """
        Create modality-based masks (fully mask one modality).

        Args:
            mask_path_modality: If True, mask all path tokens. If False, mask all text tokens.

        Returns: (path_mask_indices, char_mask_indices) where 1 = masked, 0 = not masked
        """
        path_len = path_coords.shape[0]
        char_len = char_tokens.shape[0]

        if mask_path_modality:
            # Mask all valid path tokens
            path_mask_indices = (
                path_mask.clone() if self.mask_path else torch.zeros(path_len, dtype=torch.long)
            )
            # Don't mask text tokens
            char_mask_indices = torch.zeros(char_len, dtype=torch.long)
        else:
            # Don't mask path tokens
            path_mask_indices = torch.zeros(path_len, dtype=torch.long)
            # Mask all valid text tokens (including EOS, exclude PAD only)
            char_mask_indices = torch.zeros(char_len, dtype=torch.long)
            for i in range(char_len):
                # Skip PAD tokens only
                if char_mask[i] == 0:
                    continue
                char_mask_indices[i] = 1

        return path_mask_indices, char_mask_indices

    def _apply_path_mask(self, path_coords, path_mask_indices):
        """Apply masking to path coordinates by zeroing them out."""
        masked_coords = path_coords.clone()
        for i in range(len(path_mask_indices)):
            if path_mask_indices[i] == 1:
                masked_coords[i] = 0.0
        return masked_coords

    def _apply_char_mask(self, char_tokens, char_mask_indices):
        """Apply masking to character tokens using MASK token."""
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
        gradient_mask = []  # 1 = gets gradients (query), 0 = detached (key)

        for pair_id, item in enumerate(batch):
            path_coords = item["path_coords"]
            path_mask = item["path_mask"]
            char_tokens = item["char_tokens"]
            char_mask = item["char_mask"]

            # Randomly choose between inverted mode and modality mode
            use_modality_mode = random.random() < self.modality_prob

            # attention mask components (SEP is always unmasked)
            cls_mask = torch.ones(1, dtype=torch.long)
            sep_mask = torch.ones(1, dtype=torch.long)
            attn_base = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=0)

            if use_modality_mode:
                # Modality mode:
                # View A (query): Text masked, path visible -> gets gradients
                # View B (key): Path masked, text visible -> detached
                path_mask_a, char_mask_a = self._create_modality_masks(
                    path_coords, path_mask, char_tokens, char_mask, mask_path_modality=False
                )
                path_mask_b, char_mask_b = self._create_modality_masks(
                    path_coords, path_mask, char_tokens, char_mask, mask_path_modality=True
                )
                gradient_a = 1  # Query gets gradients
                gradient_b = 0  # Key is detached
            else:
                # Inverted mode:
                # View A (query): Heavy augmentation (0.5-0.7) -> gets gradients
                # View B (key): Light augmentation (0.1-0.2) -> detached
                path_mask_a, char_mask_a = self._create_inverted_masks(
                    path_coords, path_mask, char_tokens, char_mask, heavy_aug=True
                )
                path_mask_b, char_mask_b = self._create_inverted_masks(
                    path_coords, path_mask, char_tokens, char_mask, heavy_aug=False
                )
                gradient_a = 1  # Query gets gradients
                gradient_b = 0  # Key is detached

            # Create View A (query)
            masked_path_a = self._apply_path_mask(path_coords, path_mask_a)
            masked_char_a, labels_a = self._apply_char_mask(char_tokens, char_mask_a)

            views_paths.append(masked_path_a)
            views_tokens.append(masked_char_a)
            views_labels.append(labels_a)
            views_attention.append(attn_base)
            views_char_mask.append(char_mask)
            views_path_mask.append(path_mask)
            views_path_labels.append(path_coords)
            views_path_mask_indices.append(path_mask_a)
            pair_ids.append(pair_id)
            gradient_mask.append(gradient_a)

            # Create View B (key)
            masked_path_b = self._apply_path_mask(path_coords, path_mask_b)
            masked_char_b, labels_b = self._apply_char_mask(char_tokens, char_mask_b)

            views_paths.append(masked_path_b)
            views_tokens.append(masked_char_b)
            views_labels.append(labels_b)
            views_attention.append(attn_base)
            views_char_mask.append(char_mask)
            views_path_mask.append(path_mask)
            views_path_labels.append(path_coords)
            views_path_mask_indices.append(path_mask_b)
            pair_ids.append(pair_id)
            gradient_mask.append(gradient_b)

        result = {
            "path_coords": torch.stack(views_paths),
            "char_tokens": torch.stack(views_tokens),
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

        return result


class ValidationCollator:
    """
    Collator for validation that doesn't apply masking.

    Evaluates the model's ability to predict all character positions
    from the full unmasked input, giving true reconstruction accuracy.
    """

    def __init__(self, tokenizer: CharacterTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Stack tensors without any masking
        path_coords = torch.stack([item["path_coords"] for item in batch])
        char_tokens = torch.stack([item["char_tokens"] for item in batch])
        path_mask = torch.stack([item["path_mask"] for item in batch])
        char_mask = torch.stack([item["char_mask"] for item in batch])

        # Create labels for all valid (non-padding) positions (vectorized)
        # We want to evaluate prediction on all real tokens (excluding PAD)
        batch_size = char_tokens.shape[0]
        char_labels = char_tokens.clone()

        # Set padding positions to -100 (ignore in loss)
        char_labels[char_mask == 0] = -100

        # Create attention mask: [CLS] + path + [SEP] + chars
        cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
        sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
        attention_mask = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=1)

        return {
            "path_coords": path_coords,
            "char_tokens": char_tokens,
            "char_labels": char_labels,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "words": [item["word"] for item in batch],
        }


class CrossEncoderCollator:
    """
    Collator for cross-encoder training with Multiple Negatives Ranking Loss.

    Batches (path, positive, negative_1, ..., negative_n) tuples.
    Each query has 1 positive + N negatives.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Collate batch of cross-encoder samples.

        Input: List of dicts with:
            - path_coords: [path_len, 3]
            - positive_word: [word_len]
            - negative_words: [num_negatives, word_len]

        Output: Dict with:
            - path_coords: [batch * (1+N), path_len, 3]  # Repeated for each word
            - char_tokens: [batch * (1+N), word_len]     # Positive + negatives
            - attention_mask: [batch * (1+N), seq_len]
            - labels: [batch] with value 0 (positive is always first in each group)
            - group_sizes: [batch] with value (1+N) for each query
        """
        batch_size = len(batch)
        num_negatives = batch[0]["negative_words"].shape[0]
        docs_per_query = 1 + num_negatives  # 1 positive + N negatives

        # Lists to accumulate all pairs
        all_path_coords = []
        all_path_masks = []
        all_char_tokens = []
        all_char_masks = []

        # Process each query
        for item in batch:
            path_coords = item["path_coords"]  # [path_len, 3]
            path_mask = item["path_mask"]  # [path_len]

            # Positive pair
            all_path_coords.append(path_coords)
            all_path_masks.append(path_mask)
            all_char_tokens.append(item["positive_word"])
            all_char_masks.append(item["positive_mask"])

            # Negative pairs
            for i in range(num_negatives):
                all_path_coords.append(path_coords)  # Same path repeated
                all_path_masks.append(path_mask)
                all_char_tokens.append(item["negative_words"][i])
                all_char_masks.append(item["negative_masks"][i])

        # Stack all pairs
        path_coords = torch.stack(all_path_coords)  # [batch*(1+N), path_len, 3]
        path_mask = torch.stack(all_path_masks)  # [batch*(1+N), path_len]
        char_tokens = torch.stack(all_char_tokens)  # [batch*(1+N), word_len]
        char_mask = torch.stack(all_char_masks)  # [batch*(1+N), word_len]

        # Create attention mask: [CLS] + path + [SEP] + chars
        batch_pairs = path_coords.shape[0]
        cls_mask = torch.ones(batch_pairs, 1, dtype=torch.long)
        sep_mask = torch.ones(batch_pairs, 1, dtype=torch.long)
        attention_mask = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=1)

        # Labels: positive is always at index 0 within each group
        labels = torch.zeros(batch_size, dtype=torch.long)

        # Group sizes: how many docs per query
        group_sizes = torch.full((batch_size,), docs_per_query, dtype=torch.long)

        return {
            "path_coords": path_coords,
            "char_tokens": char_tokens,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "labels": labels,  # Positive is always index 0
            "group_sizes": group_sizes,  # For reshaping scores
            "batch_size": batch_size,  # Original batch size
        }
