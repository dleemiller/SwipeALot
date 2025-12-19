"""Text helpers shared across training/eval.

SwipeALot training/evaluation treats a word's "swipable" content as lowercase
alphanumeric characters only (letters + digits). Punctuation is ignored.
"""

from __future__ import annotations


def swipable_text(text: str) -> str:
    """Lowercase `text` and keep only letters/digits."""
    return "".join(c for c in text.lower() if c.isalpha() or c.isdigit())


def swipable_length(text: str, *, max_len: int | None = None) -> int:
    """Length of `swipable_text(text)`, optionally clipped to `max_len`."""
    length = sum(1 for c in text.lower() if c.isalpha() or c.isdigit())
    if max_len is None:
        return length
    return min(int(max_len), length)
