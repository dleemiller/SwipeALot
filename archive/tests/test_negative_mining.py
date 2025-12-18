"""Tests for negative mining with bigrams and Jaccard similarity."""

import tempfile
from pathlib import Path

from src.swipealot.data.negative_mining import (
    build_negative_pool,
    compute_difficulty_score,
    get_bigrams,
    jaccard_similarity,
    load_negative_pool,
    normalize_word,
    save_negative_pool,
)


class TestBigrams:
    """Test bigram extraction."""

    def test_get_bigrams_normal(self):
        """Test bigram extraction for normal words."""
        assert get_bigrams("hello") == {"he", "el", "ll", "lo"}
        assert get_bigrams("test") == {"te", "es", "st"}

    def test_get_bigrams_single_char(self):
        """Test bigram extraction for single character."""
        assert get_bigrams("a") == set()

    def test_get_bigrams_two_chars(self):
        """Test bigram extraction for two characters."""
        assert get_bigrams("ab") == {"ab"}

    def test_get_bigrams_empty(self):
        """Test bigram extraction for empty string."""
        assert get_bigrams("") == set()

    def test_get_bigrams_duplicates(self):
        """Test bigram extraction with duplicate bigrams."""
        # "lolol" has overlapping "lo" and "ol" bigrams
        bigrams = get_bigrams("lolol")
        assert bigrams == {"lo", "ol"}  # Set should deduplicate


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""

    def test_identical_sets(self):
        """Test Jaccard for identical sets."""
        set1 = {"he", "el", "ll", "lo"}
        set2 = {"he", "el", "ll", "lo"}
        assert jaccard_similarity(set1, set2) == 1.0

    def test_disjoint_sets(self):
        """Test Jaccard for completely different sets."""
        set1 = {"ab", "cd"}
        set2 = {"ef", "gh"}
        assert jaccard_similarity(set1, set2) == 0.0

    def test_partial_overlap(self):
        """Test Jaccard for partial overlap."""
        # "hello" vs "jello": 3 shared bigrams out of 4 total
        set1 = {"he", "el", "ll", "lo"}
        set2 = {"je", "el", "ll", "lo"}
        # Intersection: {el, ll, lo} = 3
        # Union: {he, je, el, ll, lo} = 5
        # Jaccard = 3/5 = 0.6
        assert jaccard_similarity(set1, set2) == 0.6

    def test_empty_sets(self):
        """Test Jaccard for empty sets."""
        assert jaccard_similarity(set(), set()) == 0.0
        assert jaccard_similarity({"ab"}, set()) == 0.0
        assert jaccard_similarity(set(), {"ab"}) == 0.0


class TestNormalization:
    """Test word normalization."""

    def test_normalize_no_punctuation(self):
        """Test normalization of word without punctuation."""
        assert normalize_word("grip") == "grip"
        assert normalize_word("hello") == "hello"

    def test_normalize_with_punctuation(self):
        """Test normalization removes punctuation."""
        assert normalize_word("grip.") == "grip"
        assert normalize_word("grip,") == "grip"
        assert normalize_word("grip!") == "grip"
        assert normalize_word("hello?") == "hello"

    def test_normalize_case_insensitive(self):
        """Test normalization lowercases."""
        assert normalize_word("GRIP") == "grip"
        assert normalize_word("Grip") == "grip"
        assert normalize_word("GrIp") == "grip"

    def test_normalize_combined(self):
        """Test normalization with both punctuation and case."""
        assert normalize_word("GRIP.") == "grip"
        assert normalize_word("Hello,") == "hello"
        assert normalize_word("WORLD!") == "world"

    def test_normalize_preserves_different_words(self):
        """Test normalization keeps different words different."""
        assert normalize_word("grip") != normalize_word("grips")
        assert normalize_word("hello") != normalize_word("world")


class TestDifficultyScore:
    """Test difficulty score computation."""

    def test_high_jaccard_same_length(self):
        """High Jaccard (0.6-0.8) with same length should give highest score."""
        # Jaccard >= 0.6 gives base score 10.0
        # Length diff 0 gives penalty 0
        assert compute_difficulty_score(jaccard=0.7, length_diff=0) == 10.0

    def test_medium_jaccard_same_length(self):
        """Medium Jaccard (0.4-0.6) with same length."""
        # Jaccard >= 0.4 gives base score 8.0
        assert compute_difficulty_score(jaccard=0.5, length_diff=0) == 8.0

    def test_low_jaccard_same_length(self):
        """Low Jaccard (0.2-0.4) with same length."""
        # Jaccard >= 0.2 gives base score 5.0
        assert compute_difficulty_score(jaccard=0.3, length_diff=0) == 5.0

    def test_very_low_jaccard(self):
        """Very low Jaccard (<0.2)."""
        # Jaccard < 0.2 gives base score 2.0
        assert compute_difficulty_score(jaccard=0.1, length_diff=0) == 2.0

    def test_length_penalty(self):
        """Length difference should reduce difficulty."""
        # Base score 10.0, length_diff 2 -> penalty 1.0 -> final 9.0
        assert compute_difficulty_score(jaccard=0.7, length_diff=2) == 9.0

    def test_min_score_floor(self):
        """Score should not go below 0.1."""
        # Large length penalty should still floor at 0.1
        assert compute_difficulty_score(jaccard=0.1, length_diff=10) == 0.1


class TestBuildNegativePool:
    """Test negative pool building."""

    def test_simple_pool(self):
        """Test building pool with simple word list."""
        words = ["hello", "jello", "world", "would"]
        pool = build_negative_pool(
            words=words,
            num_negatives=10,
            min_jaccard=0.2,
            max_jaccard=0.9,
            max_length_diff=2,
        )

        # All words should be in pool
        assert len(pool) == 4

        # "hello" and "jello" should be negatives for each other (high Jaccard)
        hello_negatives = [neg for neg, _ in pool["hello"]]
        assert "jello" in hello_negatives

        jello_negatives = [neg for neg, _ in pool["jello"]]
        assert "hello" in jello_negatives

    def test_length_filter(self):
        """Test that length difference filter works."""
        words = ["hi", "hello", "world"]
        pool = build_negative_pool(
            words=words,
            num_negatives=10,
            min_jaccard=0.0,
            max_jaccard=1.0,
            max_length_diff=1,  # Max diff of 1
        )

        # "hi" (len=2) should not have "hello" (len=5) as negative (diff=3)
        if "hi" in pool and pool["hi"]:
            hi_negatives = [neg for neg, _ in pool["hi"]]
            assert "hello" not in hi_negatives

    def test_jaccard_filter(self):
        """Test that Jaccard similarity filter works."""
        words = ["abc", "xyz", "test"]
        pool = build_negative_pool(
            words=words,
            num_negatives=10,
            min_jaccard=0.5,  # Require at least 50% overlap
            max_jaccard=0.9,
            max_length_diff=2,
        )

        # "abc" and "xyz" have no bigram overlap, should not be negatives
        if "abc" in pool and pool["abc"]:
            abc_negatives = [neg for neg, _ in pool["abc"]]
            assert "xyz" not in abc_negatives

    def test_punctuation_filter(self):
        """Test that trivial punctuation variations are filtered out."""
        words = ["grip", "grip.", "grip,", "grips", "grim", "trip"]
        pool = build_negative_pool(
            words=words,
            num_negatives=10,
            min_jaccard=0.2,
            max_jaccard=0.9,
            max_length_diff=2,
        )

        # "grip" should NOT have "grip." or "grip," as negatives (same word)
        grip_negatives = [neg for neg, _ in pool["grip"]]
        assert "grip." not in grip_negatives, "grip. should be filtered (same as grip)"
        assert "grip," not in grip_negatives, "grip, should be filtered (same as grip)"

        # "grip" SHOULD have genuinely different words
        # Note: "grips" is kept (different word), "grim" and "trip" should be there too
        assert any(neg in ["grim", "trip"] for neg in grip_negatives), (
            "Should have genuinely different words as negatives"
        )

    def test_difficulty_ordering(self):
        """Test that negatives are sorted by difficulty (descending)."""
        words = ["hello", "jello", "world", "would", "test"]
        pool = build_negative_pool(
            words=words,
            num_negatives=10,
            min_jaccard=0.1,
            max_jaccard=0.9,
            max_length_diff=2,
        )

        # Check that scores are in descending order
        for word, negatives in pool.items():
            scores = [score for _, score in negatives]
            assert scores == sorted(scores, reverse=True), f"Scores not sorted for {word}"


class TestSaveLoadNegativePool:
    """Test saving and loading negative pools."""

    def test_save_load_parquet(self):
        """Test save and load roundtrip with Parquet."""
        # Create test pool
        pool = {
            "hello": [("jello", 9.5), ("yellow", 7.0)],
            "world": [("would", 8.0), ("word", 7.5)],
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            save_negative_pool(pool, temp_path)

            # Load back
            loaded_pool = load_negative_pool(temp_path)

            # Check content matches
            assert len(loaded_pool) == len(pool)
            assert set(loaded_pool.keys()) == set(pool.keys())

            for word in pool:
                assert word in loaded_pool
                # Check negatives match (allowing for float precision)
                original = pool[word]
                loaded = loaded_pool[word]
                assert len(original) == len(loaded)
                for (orig_neg, orig_score), (load_neg, load_score) in zip(
                    original, loaded, strict=True
                ):
                    assert orig_neg == load_neg
                    assert abs(orig_score - load_score) < 1e-5

        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)

    def test_load_maintains_order(self):
        """Test that loading maintains difficulty score ordering."""
        pool = {
            "test": [("best", 10.0), ("rest", 8.0), ("west", 5.0)],
        }

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            save_negative_pool(pool, temp_path)
            loaded_pool = load_negative_pool(temp_path)

            # Check ordering is preserved (descending by score)
            loaded_negatives = loaded_pool["test"]
            scores = [score for _, score in loaded_negatives]
            assert scores == sorted(scores, reverse=True)

        finally:
            Path(temp_path).unlink(missing_ok=True)
