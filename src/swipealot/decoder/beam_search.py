"""Trie-constrained beam search decoder for CTC outputs.

Decodes CTC logits into word candidates using vocabulary-constrained
beam search. Only produces valid vocabulary words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .trie import Trie, TrieNode


@dataclass
class BeamHypothesis:
    """A hypothesis in the beam search.

    Attributes:
        prefix: Current character sequence (may include blanks)
        text: Actual text without blanks
        score: Log probability score
        trie_node: Current position in vocabulary trie
        blank_ended: Whether last token was blank
    """

    prefix: list[int] = field(default_factory=list)
    text: str = ""
    score: float = 0.0
    trie_node: TrieNode | None = None
    blank_ended: bool = True  # Start as if blank-ended


class TrieBeamSearch:
    """Trie-constrained CTC beam search decoder.

    Decodes CTC log probabilities into word candidates, constrained
    to only produce words that exist in the vocabulary trie.

    Features:
    - Vocabulary-constrained: Only produces valid words
    - Handles repeated characters via CTC blank tokens
    - Optional frequency weighting to boost common words
    - Efficient pruning via trie structure

    Example:
        trie = Trie.from_file("vocabulary.txt")
        decoder = TrieBeamSearch(trie, beam_width=10)

        # Get log probs from model
        log_probs = model.get_log_probs(path_features)  # [T, B, 27]

        # Decode single example
        candidates = decoder.decode(log_probs[:, 0, :])  # [(word, score), ...]
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
        """Initialize beam search decoder.

        Args:
            trie: Vocabulary trie for constraint
            beam_width: Maximum number of hypotheses to keep
            blank_idx: Index of CTC blank token (default 26)
            use_frequency_weighting: Whether to boost common words
            frequency_bonus: Weight for log-frequency in scoring
            length_bonus: Bonus per character to offset CTC's short-word bias
        """
        self.trie = trie
        self.beam_width = beam_width
        self.blank_idx = blank_idx
        self.use_frequency_weighting = use_frequency_weighting
        self.frequency_bonus = frequency_bonus
        self.length_bonus = length_bonus

        # Character index to character mapping (a=0, b=1, ...)
        self.idx_to_char = {i: chr(ord("a") + i) for i in range(26)}

    def _length_score(self, word_len: int) -> float:
        """Compute static length score for a candidate word.

        Counteracts CTC's short-word bias by giving a per-character bonus.

        Args:
            word_len: Length of the candidate word

        Returns:
            Length score contribution
        """
        return self.length_bonus * word_len

    def decode(
        self,
        log_probs: torch.Tensor,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Decode CTC log probabilities to word candidates.

        Args:
            log_probs: [T, 27] log probabilities (time-first)
            top_k: Number of candidates to return

        Returns:
            List of (word, score) tuples, sorted by score descending
        """
        n_steps = log_probs.shape[0]

        # Convert to numpy for efficiency
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()

        # Initialize beam with empty hypothesis
        beam: list[BeamHypothesis] = [BeamHypothesis(trie_node=self.trie.root, blank_ended=True)]

        # Process each time step
        for t in range(n_steps):
            new_beam: dict[str, BeamHypothesis] = {}

            for hyp in beam:
                # Get log probs for this timestep
                probs_t = log_probs[t]  # [27]

                # Option 1: Emit blank (stay in same state)
                blank_score = hyp.score + probs_t[self.blank_idx]
                key = hyp.text + "|blank"
                if key not in new_beam or new_beam[key].score < blank_score:
                    new_beam[key] = BeamHypothesis(
                        prefix=hyp.prefix + [self.blank_idx],
                        text=hyp.text,
                        score=blank_score,
                        trie_node=hyp.trie_node,
                        blank_ended=True,
                    )

                # Option 2: Emit character (if valid in trie)
                if hyp.trie_node is not None:
                    for char, child_node in hyp.trie_node.children.items():
                        char_idx = ord(char) - ord("a")
                        char_score = hyp.score + probs_t[char_idx]

                        # Can only emit same char if blank-ended
                        if hyp.text and hyp.text[-1] == char and not hyp.blank_ended:
                            continue

                        new_text = hyp.text + char
                        key = new_text

                        if key not in new_beam or new_beam[key].score < char_score:
                            new_beam[key] = BeamHypothesis(
                                prefix=hyp.prefix + [char_idx],
                                text=new_text,
                                score=char_score,
                                trie_node=child_node,
                                blank_ended=False,
                            )

            # Prune beam to top-k
            beam = sorted(new_beam.values(), key=lambda h: h.score, reverse=True)
            beam = beam[: self.beam_width]

            # Early termination if beam is empty
            if not beam:
                break

        # Filter to complete words and apply frequency weighting + length score
        candidates: list[tuple[str, float]] = []
        for hyp in beam:
            if hyp.trie_node is not None and hyp.trie_node.is_word:
                score = hyp.score
                if self.use_frequency_weighting:
                    score += self.frequency_bonus * hyp.trie_node.log_frequency
                score += self._length_score(len(hyp.text))
                candidates.append((hyp.text, score))

        # Sort by final score and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        top_k: int = 3,
    ) -> list[list[tuple[str, float]]]:
        """Decode batch of CTC log probabilities.

        Args:
            log_probs: [T, B, 27] log probabilities (time-first)
            top_k: Number of candidates per example

        Returns:
            List of candidate lists, one per batch element
        """
        n_batch = log_probs.shape[1]
        results = []

        for b in range(n_batch):
            candidates = self.decode(log_probs[:, b, :], top_k=top_k)
            results.append(candidates)

        return results

    def decode_greedy(self, log_probs: torch.Tensor) -> str:
        """Greedy decoding (no beam search, no trie constraint).

        Useful for debugging and comparison.

        Args:
            log_probs: [T, 27] log probabilities

        Returns:
            Decoded string (may not be a valid word)
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu()

        # Get most likely token at each timestep
        best_tokens = log_probs.argmax(dim=-1).tolist()  # [T]

        # Collapse repeats and remove blanks
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
        log_probs: torch.Tensor,
        top_k: int = 3,
    ) -> list[dict]:
        """Decode with detailed score breakdown.

        Args:
            log_probs: [T, 27] log probabilities
            top_k: Number of candidates

        Returns:
            List of dicts with word, ctc_score, freq_score, length_score, final_score
        """
        n_steps = log_probs.shape[0]

        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()

        # Run beam search (same as decode)
        beam: list[BeamHypothesis] = [BeamHypothesis(trie_node=self.trie.root, blank_ended=True)]

        for t in range(n_steps):
            new_beam: dict[str, BeamHypothesis] = {}

            for hyp in beam:
                probs_t = log_probs[t]

                # Blank
                blank_score = hyp.score + probs_t[self.blank_idx]
                key = hyp.text + "|blank"
                if key not in new_beam or new_beam[key].score < blank_score:
                    new_beam[key] = BeamHypothesis(
                        prefix=hyp.prefix + [self.blank_idx],
                        text=hyp.text,
                        score=blank_score,
                        trie_node=hyp.trie_node,
                        blank_ended=True,
                    )

                # Characters
                if hyp.trie_node is not None:
                    for char, child_node in hyp.trie_node.children.items():
                        char_idx = ord(char) - ord("a")
                        char_score = hyp.score + probs_t[char_idx]

                        if hyp.text and hyp.text[-1] == char and not hyp.blank_ended:
                            continue

                        new_text = hyp.text + char
                        key = new_text

                        if key not in new_beam or new_beam[key].score < char_score:
                            new_beam[key] = BeamHypothesis(
                                prefix=hyp.prefix + [char_idx],
                                text=new_text,
                                score=char_score,
                                trie_node=child_node,
                                blank_ended=False,
                            )

            beam = sorted(new_beam.values(), key=lambda h: h.score, reverse=True)
            beam = beam[: self.beam_width]

            if not beam:
                break

        # Build detailed results
        results: list[dict] = []
        for hyp in beam:
            if hyp.trie_node is not None and hyp.trie_node.is_word:
                freq_score = (
                    self.frequency_bonus * hyp.trie_node.log_frequency
                    if self.use_frequency_weighting
                    else 0.0
                )
                length_score = self._length_score(len(hyp.text))
                results.append(
                    {
                        "word": hyp.text,
                        "ctc_score": hyp.score,
                        "freq_score": freq_score,
                        "length_score": length_score,
                        "frequency": hyp.trie_node.frequency,
                        "final_score": hyp.score + freq_score + length_score,
                    }
                )

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]


def prefix_beam_search(
    log_probs: torch.Tensor,
    trie: Trie,
    beam_width: int = 10,
    top_k: int = 3,
    blank_idx: int = 26,
) -> list[tuple[str, float]]:
    """Functional interface for trie-constrained beam search.

    Convenience function that creates a decoder and runs decode.

    Args:
        log_probs: [T, 27] or [T, B, 27] log probabilities
        trie: Vocabulary trie
        beam_width: Beam width
        top_k: Number of results
        blank_idx: Blank token index

    Returns:
        List of (word, score) tuples
    """
    decoder = TrieBeamSearch(
        trie=trie,
        beam_width=beam_width,
        blank_idx=blank_idx,
    )

    if log_probs.dim() == 2:
        return decoder.decode(log_probs, top_k=top_k)
    else:
        return decoder.decode_batch(log_probs, top_k=top_k)
