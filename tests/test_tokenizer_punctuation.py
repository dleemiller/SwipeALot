from src.swipealot.data.tokenizer import CharacterTokenizer


def test_punctuation_maps_to_punc_token_and_is_dropped_in_decode():
    tokenizer = CharacterTokenizer()
    punc_id = tokenizer.punc_token_id

    text = "Hello, World! 123"
    tokens = tokenizer.encode(text)

    # All letters should map to lowercase ids; punctuation/symbols to [PUNC]
    assert tokens.count(punc_id) >= 2  # at least comma and exclamation

    decoded = tokenizer.decode(tokens)
    # Decoding removes specials (including [PUNC]) and lowercases everything
    assert decoded == "helloworld123"


def test_uppercase_and_lowercase_share_same_token_ids():
    tokenizer = CharacterTokenizer()
    assert tokenizer.encode("HELLO") == tokenizer.encode("hello")
