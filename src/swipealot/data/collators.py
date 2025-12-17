"""Data collators for masked language modeling and contrastive learning."""

import random
from typing import Any

import torch

from .tokenizer import CharacterTokenizer


def _swipable_length(word: str, max_len: int) -> int:
    length = sum(1 for c in word.lower() if c.isalpha() or c.isdigit())
    return min(length, max_len)


def _mask_contiguous_blocks_1d(
    valid_mask: torch.Tensor,
    n_to_mask: int,
    max_block_len: int,
    rng: random.Random,
) -> torch.Tensor:
    """
    Create a 1D binary mask (1=masked) consisting of contiguous blocks within valid segments.

    Args:
        valid_mask: [seq_len] tensor with 1 for valid points, 0 for padding/invalid.
        n_to_mask: Target number of positions to mask (clipped to number of valid points).
        max_block_len: Absolute maximum length for any single contiguous masked block.
        rng: Random generator to sample block sizes and locations.

    Returns:
        mask_indices: [seq_len] long tensor with 1 at masked positions.
    """
    seq_len = int(valid_mask.shape[0])
    valid_mask = valid_mask.to(dtype=torch.long)
    n_valid = int(valid_mask.sum().item())
    n_to_mask = max(0, min(int(n_to_mask), n_valid))

    mask_indices = torch.zeros(seq_len, dtype=torch.long)
    if n_to_mask == 0 or n_valid == 0:
        return mask_indices
    max_block_len = max(1, int(max_block_len))

    # Find contiguous valid segments (runs of 1s).
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, v in enumerate(valid_mask.tolist()):
        if v == 1 and not in_seg:
            seg_start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((seg_start, i))  # end exclusive
            in_seg = False
    if in_seg:
        segments.append((seg_start, seq_len))

    masked = torch.zeros(seq_len, dtype=torch.bool)
    remaining = n_to_mask

    def _segment_available(s: int, e: int) -> int:
        return int((~masked[s:e]).sum().item())

    def _left_run(start: int) -> int:
        run = 0
        i = start - 1
        while i >= 0 and valid_mask[i].item() == 1 and masked[i].item():
            run += 1
            i -= 1
        return run

    def _right_run(end: int) -> int:
        run = 0
        i = end
        while i < seq_len and valid_mask[i].item() == 1 and masked[i].item():
            run += 1
            i += 1
        return run

    # Sample blocks until we reach the target number of masked positions.
    # Prefer larger blocks by sampling block_len uniformly from [min_len, max_len].
    attempts = 0
    max_attempts = 50 * (n_to_mask + 1)
    while remaining > 0 and attempts < max_attempts:
        # Choose a segment weighted by available unmasked capacity.
        candidates: list[tuple[int, int, int]] = []
        total_avail = 0
        for s, e in segments:
            avail = _segment_available(s, e)
            if avail > 0:
                candidates.append((s, e, avail))
                total_avail += avail

        if total_avail == 0:
            break

        pick = rng.randrange(total_avail)
        chosen_s = chosen_e = 0
        chosen_avail = 0
        acc = 0
        for s, e, avail in candidates:
            acc += avail
            if pick < acc:
                chosen_s, chosen_e, chosen_avail = s, e, avail
                break

        max_len = min(remaining, chosen_avail, max_block_len)
        if max_len <= 0:
            attempts += 1
            continue

        min_len = 2 if max_len >= 2 else 1
        block_len = rng.randint(min_len, max_len)

        placed = False
        for _ in range(6):
            # Find all valid start positions for a contiguous unmasked block of length block_len
            # that would not create a run longer than max_block_len.
            start_candidates: list[int] = []
            for start in range(chosen_s, chosen_e - block_len + 1):
                end = start + block_len
                if masked[start:end].any():
                    continue
                run_len = _left_run(start) + block_len + _right_run(end)
                if run_len <= max_block_len:
                    start_candidates.append(start)

            if start_candidates:
                start = rng.choice(start_candidates)
                masked[start : start + block_len] = True
                remaining -= block_len
                placed = True
                break

            # Try a smaller block length.
            if block_len <= min_len:
                break
            block_len = max(min_len, block_len // 2)

        if not placed:
            # Fallback: mask a single random available position in the chosen segment,
            # but still respect the max_block_len cap.
            avail_positions = (~masked[chosen_s:chosen_e]).nonzero(as_tuple=True)[0]
            if avail_positions.numel() > 0:
                perm = avail_positions[torch.randperm(avail_positions.numel())]
                for rel in perm.tolist():
                    pos = chosen_s + int(rel)
                    if _left_run(pos) + 1 + _right_run(pos + 1) <= max_block_len:
                        masked[pos] = True
                        remaining -= 1
                        break

        attempts += 1

    # If we couldn't place enough contiguous blocks, fill remaining masks uniformly over valid positions.
    if remaining > 0:
        avail_positions = ((valid_mask == 1) & (~masked)).nonzero(as_tuple=True)[0]
        if avail_positions.numel() > 0:
            perm = avail_positions[torch.randperm(avail_positions.numel())]

            def _max_run_with(pos: int) -> int:
                # Compute max contiguous run length if we set masked[pos]=True.
                tmp = masked.clone()
                tmp[pos] = True
                # Evaluate only within valid_mask
                m = (tmp & (valid_mask == 1)).to(torch.int64)
                max_run = 0
                cur = 0
                for v in m.tolist():
                    if v == 1:
                        cur += 1
                        if cur > max_run:
                            max_run = cur
                    else:
                        cur = 0
                return max_run

            for p in perm.tolist():
                if remaining <= 0:
                    break
                if _max_run_with(int(p)) <= max_block_len:
                    masked[int(p)] = True
                    remaining -= 1

    mask_indices[masked] = 1
    return mask_indices


class MaskedCollator:
    """
    Collator that creates masked versions of characters and paths for MLM-style training.
    """

    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        char_mask_prob: float | tuple[float, float] = 0.15,
        path_mask_prob: float = 0.15,
        mask_path: bool = True,
        mask_vocab_only: bool = False,
        path_mask_block_max_len: int = 32,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Character tokenizer for masking
            char_mask_prob: Probability of masking each character.
                          Can be a float (fixed probability) or tuple (min, max)
                          to randomly sample probability per batch from range.
            path_mask_prob: Probability of masking each path point
            mask_path: Whether to mask path points
            mask_vocab_only: If True, only mask vocabulary tokens (a-z, 0-9),
                           never mask special tokens ([EOS], [PUNC], [UNK])
        """
        self.tokenizer = tokenizer
        self.char_mask_prob = char_mask_prob
        self.path_mask_prob = path_mask_prob
        self.mask_path = mask_path
        self.mask_vocab_only = mask_vocab_only
        self.path_mask_block_max_len = path_mask_block_max_len

        # Check if char_mask_prob is a range
        if isinstance(char_mask_prob, (tuple, list)):
            if len(char_mask_prob) != 2:
                raise ValueError("char_mask_prob range must have exactly 2 values (min, max)")
            self.char_mask_prob_min, self.char_mask_prob_max = char_mask_prob
            self.use_random_mask_prob = True
        else:
            self.use_random_mask_prob = False

        # Precompute vocabulary token range for efficiency
        if self.mask_vocab_only:
            self.vocab_start_id = len(tokenizer.special_tokens)  # 7
            self.vocab_end_id = tokenizer.vocab_size  # 43

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

        # Sample masking probability for this batch if using random range
        if self.use_random_mask_prob:
            import random

            char_mask_prob = random.uniform(self.char_mask_prob_min, self.char_mask_prob_max)
        else:
            char_mask_prob = self.char_mask_prob

        # Vectorized masking: create random decisions for all tokens
        mask_decisions = torch.rand(batch_size, seq_len) < char_mask_prob
        mask_decisions = mask_decisions & (char_mask == 1)  # Only mask valid tokens

        # Filter to vocabulary tokens only if mask_vocab_only=True
        if self.mask_vocab_only:
            is_vocab_token = (char_tokens >= self.vocab_start_id) & (
                char_tokens < self.vocab_end_id
            )
            mask_decisions = mask_decisions & is_vocab_token

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
        Mask path coordinates by replacing with zeros.

        Uses contiguous block masking (not independent per-point masking) so the model
        sees larger missing segments rather than many tiny holes.

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

        mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for b in range(batch_size):
            n_valid = int(path_mask[b].sum().item())
            if n_valid == 0 or self.path_mask_prob <= 0.0:
                continue
            n_to_mask = int(round(float(self.path_mask_prob) * n_valid))
            if n_to_mask <= 0:
                continue
            mask_indices[b] = _mask_contiguous_blocks_1d(
                path_mask[b],
                n_to_mask,
                max_block_len=self.path_mask_block_max_len,
                rng=random,
            )

        masked_coords[mask_indices.bool()] = 0.0

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
            "input_ids": masked_char_tokens,  # Renamed from char_tokens
            "char_labels": char_labels,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "words": [item["word"] for item in batch],  # Original words for evaluation
        }

        if self.mask_path:
            result["path_labels"] = path_labels
            result["path_mask_indices"] = path_mask_indices

        # Length targets are always available from the word string (used for metrics and optional loss).
        max_len = char_tokens.shape[1]
        result["length_target"] = torch.tensor(
            [_swipable_length(item["word"], max_len) for item in batch], dtype=torch.long
        )
        result["length_supervise_mask"] = torch.ones(len(batch), dtype=torch.long)

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
        """
        Args:
            tokenizer: Character tokenizer
            mask_path: Whether to mask path coordinates
            modality_prob: Probability of using modality-based masking (vs inverted)
            zero_attention_prob: Probability of fully zeroing attention in modality mode.
                When triggered, it zeros text attention for the text-masked view and path
                attention for the path-masked view to drop supervision symmetrically.
            inverted_*_prob_*: Masking probabilities for inverted mode; can be floats or
                (min, max) ranges sampled per call.
        """
        self.tokenizer = tokenizer
        self.mask_path = mask_path
        self.modality_prob = modality_prob
        self.zero_attention_prob = zero_attention_prob
        self.path_mask_block_max_len = path_mask_block_max_len
        self.max_char_len = None  # derived per-sample
        # Configurable inverted-mode probabilities (can be floats or (min,max) ranges)
        self.pairwise_inverted_char_prob_heavy = inverted_char_prob_heavy
        self.pairwise_inverted_path_prob_heavy = inverted_path_prob_heavy
        self.pairwise_inverted_char_prob_light = inverted_char_prob_light
        self.pairwise_inverted_path_prob_light = inverted_path_prob_light

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

        def _prob_from_cfg(val):
            if isinstance(val, (tuple, list)):
                return random.uniform(val[0], val[1])
            return float(val)

        # Random masking probabilities based on augmentation strength
        if heavy_aug:
            path_mask_prob = _prob_from_cfg(self.pairwise_inverted_path_prob_heavy)
            text_mask_prob = _prob_from_cfg(self.pairwise_inverted_char_prob_heavy)
        else:
            path_mask_prob = _prob_from_cfg(self.pairwise_inverted_path_prob_light)
            text_mask_prob = _prob_from_cfg(self.pairwise_inverted_char_prob_light)

        # Create path mask (contiguous block masking)
        path_mask_indices = torch.zeros(path_len, dtype=torch.long)
        if self.mask_path and path_mask_prob > 0.0:
            n_valid = int(path_mask.sum().item())
            n_to_mask = int(round(float(path_mask_prob) * n_valid))
            if n_to_mask > 0:
                path_mask_indices = _mask_contiguous_blocks_1d(
                    path_mask,
                    n_to_mask,
                    max_block_len=self.path_mask_block_max_len,
                    rng=random,
                )

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
        length_targets = []
        length_supervise_mask = []

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

            # Optionally zero out attention in modality mode to drop supervision symmetrically.
            use_zero_attn = (
                use_modality_mode
                and self.zero_attention_prob > 0.0
                and random.random() < self.zero_attention_prob
            )
            if use_zero_attn:
                # No attention to text positions; also drop char loss for this view.
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
            length_targets.append(_swipable_length(item["word"], char_tokens.shape[0]))
            length_supervise_mask.append(length_supervise)

            # Create View B (key)
            masked_path_b = self._apply_path_mask(path_coords, path_mask_b)
            masked_char_b, labels_b = self._apply_char_mask(char_tokens, char_mask_b)

            if use_zero_attn and use_modality_mode:
                # Symmetrically zero path attention on the key view and drop path supervision.
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
            length_targets.append(_swipable_length(item["word"], char_tokens.shape[0]))
            length_supervise_mask.append(0)  # never supervise on key

        result = {
            "path_coords": torch.stack(views_paths),
            "input_ids": torch.stack(views_tokens),  # Renamed from char_tokens
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


class ValidationCollator:
    """
    Collator for validation that doesn't apply masking.

    Evaluates the model's ability to predict all character positions
    from the full unmasked input, giving true reconstruction accuracy.
    """

    def __init__(self, tokenizer: CharacterTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Stack tensors
        path_coords = torch.stack([item["path_coords"] for item in batch])
        char_tokens = torch.stack([item["char_tokens"] for item in batch])
        path_mask = torch.stack([item["path_mask"] for item in batch])
        char_mask = torch.stack([item["char_mask"] for item in batch])

        batch_size = char_tokens.shape[0]

        # Save true labels before masking
        char_labels = char_tokens.clone()
        char_labels[char_mask == 0] = -100  # Ignore padding in loss

        # Mask all character tokens (BERT-style: mask input but allow attention)
        masked_char_tokens = char_tokens.clone()
        masked_char_tokens[char_mask == 1] = self.tokenizer.mask_token_id

        # Create attention mask: [CLS] + path + [SEP] + chars (WITH attention)
        # This allows model to use positional info while predicting masked tokens
        cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
        sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
        attention_mask = torch.cat([cls_mask, path_mask, sep_mask, char_mask], dim=1)

        words = [item["word"] for item in batch]
        max_len = char_tokens.shape[1]
        length_target = torch.tensor(
            [_swipable_length(w, max_len) for w in words], dtype=torch.long
        )

        return {
            "path_coords": path_coords,
            "input_ids": masked_char_tokens,  # Renamed from char_tokens
            "char_labels": char_labels,
            "path_mask": path_mask,
            "char_mask": char_mask,
            "attention_mask": attention_mask,
            "words": words,
            "length_target": length_target,
            "length_supervise_mask": torch.ones(batch_size, dtype=torch.long),
        }
