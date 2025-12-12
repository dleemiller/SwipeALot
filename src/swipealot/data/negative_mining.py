"""Hard negative mining using character bigrams and Jaccard similarity for cross-encoder training."""

from collections import defaultdict

import pandas as pd


def get_bigrams(word: str) -> set[str]:
    """
    Extract character bigrams from a word.

    Args:
        word: Input word

    Returns:
        Set of character bigrams

    Examples:
        "hello" → {'he', 'el', 'll', 'lo'}
        "jello" → {'je', 'el', 'll', 'lo'}
    """
    if len(word) < 2:
        return set()
    return set(word[i : i + 2] for i in range(len(word) - 1))


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity in [0, 1]
    """
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_difficulty_score(jaccard: float, length_diff: int) -> float:
    """
    Compute difficulty score for negative sampling.

    Higher score = harder negative = higher sampling probability.

    Args:
        jaccard: Jaccard similarity between word bigram sets
        length_diff: Absolute difference in word lengths

    Returns:
        Difficulty score (higher = harder)
    """
    # Jaccard 0.6-0.8: hardest (very confusable, high character overlap)
    # Jaccard 0.4-0.6: hard (moderate overlap)
    # Jaccard 0.2-0.4: medium (some overlap)
    # Jaccard <0.2: easier (little overlap)

    if jaccard >= 0.6:
        base_score = 10.0
    elif jaccard >= 0.4:
        base_score = 8.0
    elif jaccard >= 0.2:
        base_score = 5.0
    else:
        base_score = 2.0

    # Penalize length differences (same length is harder)
    length_penalty = length_diff * 0.5

    return max(base_score - length_penalty, 0.1)


def build_negative_pool(
    words: set[str] | list[str],
    num_negatives: int = 50,
    min_jaccard: float = 0.2,
    max_jaccard: float = 0.9,
    max_length_diff: int = 2,
) -> dict[str, list[tuple[str, float]]]:
    """
    Build hard negative pool using character bigrams and Jaccard similarity.

    For each word, finds candidate negatives with Jaccard similarity in [min, max]
    and length difference <= max_length_diff.

    Args:
        words: Set or list of unique words
        num_negatives: Maximum negatives per word
        min_jaccard: Minimum Jaccard similarity (0.2 recommended for some overlap)
        max_jaccard: Maximum Jaccard similarity (0.9 recommended to exclude near-duplicates)
        max_length_diff: Maximum length difference (2 recommended)

    Returns:
        Dictionary mapping word -> [(negative, difficulty_score), ...]
        Sorted by difficulty score (descending)
    """
    words = list(set(words))  # Ensure unique
    print(f"Building negative pool for {len(words)} unique words...")

    # Build bigram sets for all words
    print("Extracting bigrams...")
    word_bigrams = {}
    for word in words:
        word_bigrams[word] = get_bigrams(word)

    # Build inverted index: bigram -> words containing it
    print("Building inverted index...")
    bigram_index = defaultdict(set)
    for word, bigrams in word_bigrams.items():
        for bigram in bigrams:
            bigram_index[bigram].add(word)

    print(f"Found {len(bigram_index)} unique bigrams")

    # Find negatives for each word
    negative_pool = {}
    total_pairs = 0

    for i, word in enumerate(words):
        if i % 1000 == 0:
            print(f"Processing word {i}/{len(words)}...")

        word_bg = word_bigrams[word]
        candidates = []

        # Get candidate words that share at least one bigram
        candidate_words = set()
        for bigram in word_bg:
            candidate_words.update(bigram_index[bigram])
        candidate_words.discard(word)  # Remove self

        # Score candidates
        for candidate in candidate_words:
            # Filter by length difference
            length_diff = abs(len(word) - len(candidate))
            if length_diff > max_length_diff:
                continue

            # Compute Jaccard similarity
            candidate_bg = word_bigrams[candidate]
            jaccard = jaccard_similarity(word_bg, candidate_bg)

            # Filter by Jaccard range
            if min_jaccard <= jaccard <= max_jaccard:
                difficulty = compute_difficulty_score(jaccard, length_diff)
                candidates.append((candidate, difficulty))

        # Sort by difficulty (hardest first) and keep top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        negative_pool[word] = candidates[:num_negatives]
        total_pairs += len(negative_pool[word])

    print("\nNegative pool built:")
    print(f"  Total words: {len(negative_pool)}")
    print(f"  Total negative pairs: {total_pairs}")
    print(f"  Avg negatives per word: {total_pairs / len(negative_pool):.2f}")

    # Statistics
    words_with_negatives = sum(1 for negs in negative_pool.values() if negs)
    words_without_negatives = len(negative_pool) - words_with_negatives
    print(f"  Words with negatives: {words_with_negatives}")
    print(f"  Words without negatives: {words_without_negatives}")

    if negative_pool:
        neg_counts = [len(negs) for negs in negative_pool.values()]
        print(f"  Min negatives: {min(neg_counts)}")
        print(f"  Max negatives: {max(neg_counts)}")
        print(f"  Median negatives: {sorted(neg_counts)[len(neg_counts) // 2]}")

    return negative_pool


def save_negative_pool(negative_pool: dict, output_path: str):
    """
    Save negative pool to Parquet file.

    Args:
        negative_pool: Dict[word, List[(negative, score)]]
        output_path: Path to output Parquet file
    """
    # Convert nested dict to flat table format
    rows = []
    for word, negatives in negative_pool.items():
        for negative, score in negatives:
            rows.append({"word": word, "negative": negative, "difficulty_score": score})

    # Create DataFrame and save as Parquet
    df = pd.DataFrame(rows)

    # Ensure output path has .parquet extension
    if not output_path.endswith(".parquet"):
        output_path = output_path.replace(".json", ".parquet")

    df.to_parquet(output_path, compression="zstd", index=False)

    print(f"\n✓ Saved negative pool to: {output_path}")
    print("  Format: Parquet (zstd compression)")
    print(f"  Total pairs: {len(df):,}")


def load_negative_pool(input_path: str) -> dict[str, list[tuple[str, float]]]:
    """
    Load negative pool from Parquet file.

    Args:
        input_path: Path to Parquet file

    Returns:
        Dictionary mapping word -> [(negative, difficulty_score), ...]
    """
    # Read Parquet file
    df = pd.read_parquet(input_path)

    # Convert flat table back to nested dict
    negative_pool = {}
    for word, group in df.groupby("word"):
        negatives = list(zip(group["negative"], group["difficulty_score"], strict=True))
        # Sort by difficulty score (descending)
        negatives.sort(key=lambda x: x[1], reverse=True)
        negative_pool[word] = negatives

    print(f"Loaded negative pool with {len(negative_pool)} words from Parquet")
    return negative_pool
