"""Vocabulary trie for constrained beam search.

The trie structure enables efficient vocabulary-constrained decoding
by quickly determining valid next characters at each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrieNode:
    """A node in the vocabulary trie."""

    children: dict[str, TrieNode] = field(default_factory=dict)
    is_word: bool = False
    word: str | None = None  # Full word if this is end node
    frequency: float = 0.0  # Word frequency (higher = more common)
    log_frequency: float = float("-inf")  # Log of frequency for scoring


class Trie:
    """Vocabulary trie for efficient prefix lookup and constrained decoding.

    Supports:
    - Word insertion with optional frequency
    - Prefix search (starts_with)
    - Valid next characters lookup
    - Word frequency for beam search scoring

    Example:
        trie = Trie()
        trie.insert("hello", frequency=0.001)
        trie.insert("help", frequency=0.0005)

        # Check valid continuations
        node = trie.get_prefix_node("hel")
        valid_chars = trie.valid_next_chars(node)  # ['l', 'p']

        # Check if word exists
        trie.search("hello")  # True
    """

    def __init__(self):
        """Initialize empty trie."""
        self.root = TrieNode()
        self.word_count = 0
        self._min_frequency = float("inf")
        self._max_frequency = 0.0

    def insert(self, word: str, frequency: float = 1.0) -> None:
        """Insert a word into the trie.

        Args:
            word: Word to insert (lowercase letters only)
            frequency: Word frequency/probability (higher = more common)
        """
        word = word.lower().strip()
        if not word or not word.isalpha():
            return

        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_word:
            self.word_count += 1

        node.is_word = True
        node.word = word
        node.frequency = frequency
        node.log_frequency = math.log(frequency + 1e-10)

        self._min_frequency = min(self._min_frequency, frequency)
        self._max_frequency = max(self._max_frequency, frequency)

    def search(self, word: str) -> bool:
        """Check if exact word exists in trie.

        Args:
            word: Word to search for

        Returns:
            True if word exists in vocabulary
        """
        node = self.get_prefix_node(word)
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix.

        Args:
            prefix: Prefix to check

        Returns:
            True if any word has this prefix
        """
        return self.get_prefix_node(prefix) is not None

    def get_prefix_node(self, prefix: str) -> TrieNode | None:
        """Get the trie node for a given prefix.

        Args:
            prefix: Character sequence to traverse

        Returns:
            TrieNode at end of prefix, or None if prefix doesn't exist
        """
        prefix = prefix.lower()
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def valid_next_chars(self, node: TrieNode | None = None) -> list[str]:
        """Get valid next characters from a trie node.

        Args:
            node: Current trie node (uses root if None)

        Returns:
            List of valid next characters (sorted)
        """
        if node is None:
            node = self.root
        return sorted(node.children.keys())

    def get_word_info(self, word: str) -> tuple[bool, float, float] | None:
        """Get word existence and frequency info.

        Args:
            word: Word to look up

        Returns:
            Tuple of (is_word, frequency, log_frequency) or None if not found
        """
        node = self.get_prefix_node(word)
        if node is None or not node.is_word:
            return None
        return (True, node.frequency, node.log_frequency)

    def get_words_with_prefix(self, prefix: str, max_words: int = 100) -> list[tuple[str, float]]:
        """Get all words starting with prefix.

        Args:
            prefix: Prefix to search
            max_words: Maximum number of words to return

        Returns:
            List of (word, frequency) tuples, sorted by frequency descending
        """
        results: list[tuple[str, float]] = []

        node = self.get_prefix_node(prefix)
        if node is None:
            return results

        # DFS to collect all words
        stack: list[TrieNode] = [node]
        while stack and len(results) < max_words * 2:  # Collect extra for sorting
            current = stack.pop()
            if current.is_word and current.word is not None:
                results.append((current.word, current.frequency))
            for child in current.children.values():
                stack.append(child)

        # Sort by frequency descending and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_words]

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
            path: Path to vocabulary file
            max_words: Maximum words to load (None = all)
            has_frequency: Whether file includes frequency column

        Returns:
            Populated Trie instance
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
                    # Assign decreasing frequency based on line order
                    # (assumes file is sorted by frequency)
                    frequency = 1.0 / (i + 1)

                trie.insert(word, frequency)

        return trie

    def to_file(self, path: str | Path) -> None:
        """Save trie to vocabulary file.

        Args:
            path: Output path for vocabulary file
        """
        path = Path(path)
        words: list[tuple[str, float]] = []

        # Collect all words via DFS
        stack: list[TrieNode] = [self.root]
        while stack:
            node = stack.pop()
            if node.is_word and node.word is not None:
                words.append((node.word, node.frequency))
            for child in node.children.values():
                stack.append(child)

        # Sort by frequency descending
        words.sort(key=lambda x: x[1], reverse=True)

        with open(path, "w", encoding="utf-8") as f:
            for word, freq in words:
                f.write(f"{word}\t{freq}\n")

    def __len__(self) -> int:
        """Return number of words in trie."""
        return self.word_count

    def __contains__(self, word: str) -> bool:
        """Check if word in trie."""
        return self.search(word)

    def save(self, path: str | Path) -> None:
        """Save trie to binary file for fast loading.

        Args:
            path: Output path (recommended: .trie extension)
        """
        import pickle

        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Trie:
        """Load trie from binary file.

        Args:
            path: Path to .trie file

        Returns:
            Loaded Trie instance
        """
        import pickle

        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)
