"""Vocabulary trie for constrained beam search (Cython-backed)."""

from __future__ import annotations

import math
from pathlib import Path

from ._beam_search import Trie as _CyTrie

_LETTERS = list("abcdefghijklmnopqrstuvwxyz")


class Trie:
    """Wrapper around Cython flat-array trie, preserving the public API.

    The underlying Cython trie uses a flat C array with per-node child
    pointers, giving O(1) child lookup and cache-friendly traversal.
    """

    def __init__(self, num_chars: int = 26, letters: list[str] | None = None):
        self._cy = _CyTrie(num_chars, letters or _LETTERS)

    def insert(self, word: str, frequency: float = 1.0) -> None:
        """Insert a word into the trie.

        Args:
            word: Word to insert (lowercase letters only).
            frequency: Word frequency (raw, will be log-transformed internally).
        """
        word = word.lower().strip()
        if not word or not word.isalpha():
            return
        self._cy.insert(word, math.log(frequency + 1e-10))

    def search(self, word: str) -> bool:
        """Check if exact word exists in trie."""
        return self._cy.search(word)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        max_words: int | None = None,
        has_frequency: bool = True,
    ) -> Trie:
        """Load trie from vocabulary file.

        File format (tab-separated):
            word<TAB>frequency
            or just:
            word

        Args:
            path: Path to vocabulary file.
            max_words: Maximum words to load (None = all).
            has_frequency: Whether file includes frequency column.

        Returns:
            Populated Trie instance.
        """
        trie = cls()
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_words is not None and i >= max_words:
                    break

                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                word = parts[0].strip()

                if has_frequency and len(parts) > 1:
                    try:
                        frequency = float(parts[1])
                    except ValueError:
                        frequency = 1.0
                else:
                    frequency = 1.0 / (i + 1)

                trie.insert(word, frequency)

        return trie

    def __len__(self) -> int:
        return len(self._cy)

    def __contains__(self, word: str) -> bool:
        return self.search(word)

    @property
    def word_count(self) -> int:
        return len(self._cy)
