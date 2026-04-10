from src.swipealot.data.tokenizer import CharacterTokenizer


def test_non_alpha_chars_are_dropped():
    tokenizer = CharacterTokenizer()

    text = "Hello, World! 123"
    tokens = tokenizer.encode(text)

    # Non-alpha chars (punctuation, digits, spaces) are dropped during encode
    decoded = tokenizer.decode(tokens)
    assert decoded == "helloworld"

    # Only alpha chars produce tokens
    assert len(tokens) == 10  # len("helloworld")


def test_uppercase_and_lowercase_share_same_token_ids():
    tokenizer = CharacterTokenizer()
    assert tokenizer.encode("HELLO") == tokenizer.encode("hello")
