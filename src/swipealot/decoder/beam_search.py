"""Trie-constrained beam search decoder for CTC outputs (Cython-backed)."""

from __future__ import annotations

import numpy as np
import torch

from ._beam_search import beam_search_decode as _cy_beam_search
from ._beam_search import beam_search_decode_with_scores as _cy_beam_search_scores
from .trie import Trie


class TrieBeamSearch:
    """Trie-constrained CTC beam search decoder.

    Thin wrapper around Cython ``beam_search_decode`` that accepts
    torch tensors and matches the previous pure-Python API.
    """

    def __init__(
        self,
        trie: Trie,
        beam_width: int = 10,
        blank_idx: int = 26,
        use_frequency_weighting: bool = True,
        frequency_bonus: float = 0.4,
        length_bonus: float = 0.8,
    ):
        self.trie = trie
        self.beam_width = beam_width
        self.blank_idx = blank_idx
        self.use_frequency_weighting = use_frequency_weighting
        self.frequency_bonus = frequency_bonus
        self.length_bonus = length_bonus
        self.idx_to_char = {i: chr(ord("a") + i) for i in range(26)}

    @staticmethod
    def _to_numpy(log_probs: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(log_probs, torch.Tensor):
            return log_probs.detach().cpu().float().numpy()
        return np.asarray(log_probs, dtype=np.float32)

    def decode(
        self,
        log_probs: torch.Tensor | np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Decode CTC log probabilities to word candidates.

        Args:
            log_probs: [T, 27] log probabilities (time-first).
            top_k: Number of candidates to return.

        Returns:
            List of (word, score) tuples, sorted by score descending.
        """
        arr = self._to_numpy(log_probs)
        return _cy_beam_search(
            arr,
            self.trie._cy,
            beam_width=self.beam_width,
            freq_bonus=self.frequency_bonus if self.use_frequency_weighting else 0.0,
            length_bonus=self.length_bonus,
            top_k=top_k,
        )

    def decode_batch(
        self,
        log_probs: torch.Tensor | np.ndarray,
        top_k: int = 3,
    ) -> list[list[tuple[str, float]]]:
        """Decode batch of CTC log probabilities.

        Args:
            log_probs: [T, B, 27] log probabilities (time-first).
            top_k: Number of candidates per example.

        Returns:
            List of candidate lists, one per batch element.
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu()
        n_batch = log_probs.shape[1]
        return [self.decode(log_probs[:, b, :], top_k=top_k) for b in range(n_batch)]

    def decode_greedy(self, log_probs: torch.Tensor | np.ndarray) -> str:
        """Greedy decoding (no beam search, no trie constraint).

        Useful for debugging and comparison.

        Args:
            log_probs: [T, 27] log probabilities.

        Returns:
            Decoded string (may not be a valid word).
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu()

        best_tokens = (
            log_probs.argmax(dim=-1).tolist()
            if isinstance(log_probs, torch.Tensor)
            else np.argmax(log_probs, axis=-1).tolist()
        )

        result = []
        prev_token = None
        for token in best_tokens:
            if token == self.blank_idx:
                prev_token = None
                continue
            if token == prev_token:
                continue
            result.append(self.idx_to_char.get(token, "?"))
            prev_token = token

        return "".join(result)

    def decode_with_scores(
        self,
        log_probs: torch.Tensor | np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """Decode with detailed score breakdown.

        Args:
            log_probs: [T, 27] log probabilities.
            top_k: Number of candidates.

        Returns:
            List of dicts with word, ctc_score, freq_score, length_score.
        """
        arr = self._to_numpy(log_probs)
        return _cy_beam_search_scores(
            arr,
            self.trie._cy,
            beam_width=self.beam_width,
            top_k=top_k,
        )


def prefix_beam_search(
    log_probs: torch.Tensor,
    trie: Trie,
    beam_width: int = 10,
    top_k: int = 3,
    blank_idx: int = 26,
) -> list[tuple[str, float]]:
    """Functional interface for trie-constrained beam search."""
    decoder = TrieBeamSearch(
        trie=trie,
        beam_width=beam_width,
        blank_idx=blank_idx,
    )

    if log_probs.dim() == 2:
        return decoder.decode(log_probs, top_k=top_k)
    else:
        return decoder.decode_batch(log_probs, top_k=top_k)
