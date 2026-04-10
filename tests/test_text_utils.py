from src.swipealot.text_utils import swipable_length, swipable_text


def test_swipable_text_filters_and_lowercases():
    assert swipable_text("I-beams!") == "ibeams"
    assert swipable_text("Panama,") == "panama"
    assert swipable_text("A1-B2") == "ab"


def test_swipable_length_matches_text_length():
    s = "five-o"
    assert swipable_length(s) == len(swipable_text(s))


def test_swipable_length_max_len_clips():
    assert swipable_length("abcdefgh", max_len=3) == 3
