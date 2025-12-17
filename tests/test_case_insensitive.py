"""Tokenizer case-insensitivity checks."""

from swipealot.data import CharacterTokenizer


def test_tokenizer_is_case_insensitive():
    tokenizer = CharacterTokenizer()

    variants = ["hello", "HELLO", "Hello", "HeLLo"]
    encoded = [tokenizer.encode(v) for v in variants]

    # All encodings should match (case-insensitive)
    assert all(seq == encoded[0] for seq in encoded)

    # Decoding should lowercase and drop specials
    assert tokenizer.decode(encoded[0]) == "hello"

    # Uppercase letters should not appear in the vocabulary keys
    uppercase_in_vocab = any(
        ch.isupper() and len(ch) == 1 for ch in tokenizer.char_to_id.keys() if ch not in tokenizer.special_tokens
    )
    assert uppercase_in_vocab is False
