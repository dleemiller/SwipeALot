import torch

from src.swipealot.data.collators import PairwiseMaskedCollator
from src.swipealot.data.tokenizer import CharacterTokenizer


def test_text_attention_zeroed_in_modality_view_when_prob_one():
    """
    With modality mode + zero_text_attention_prob=1.0, the text-masked view should
    zero out text attention and drop text supervision, preventing leakage of text length.
    """
    tokenizer = CharacterTokenizer()

    path_len = 3
    char_len = 4

    sample = {
        "word": "test",
        "path_coords": torch.ones(path_len, 3),  # dummy path
        "path_mask": torch.ones(path_len, dtype=torch.long),
        "char_tokens": torch.tensor([tokenizer.cls_token_id + i + 1 for i in range(char_len)]),
        "char_mask": torch.ones(char_len, dtype=torch.long),
    }

    collator = PairwiseMaskedCollator(
        tokenizer=tokenizer,
        mask_path=True,
        modality_prob=1.0,  # force modality mode
        zero_text_attention_prob=1.0,  # always zero text attention on the text-masked view
    )

    batch = collator([sample])

    # View A (index 0) should have text attention zeroed and no text supervision
    attn_a = batch["attention_mask"][0]
    char_tokens_a = batch["char_tokens"][0]
    char_labels_a = batch["char_labels"][0]
    char_mask_a = batch["char_mask"][0]

    # Attention: CLS + path + SEP are 1, text segment is all zeros
    assert attn_a.shape[0] == 1 + path_len + 1 + char_len
    assert attn_a[: 1 + path_len + 1].sum().item() == path_len + 2  # CLS + path + SEP
    assert attn_a[-char_len:].sum().item() == 0  # no attention on text positions

    # Text is replaced with PADs and labels are ignored
    assert torch.all(char_tokens_a == tokenizer.pad_token_id)
    assert torch.all(char_labels_a == -100)
    assert torch.all(char_mask_a == 0)

    # View B (index 1) keeps normal attention/masks
    attn_b = batch["attention_mask"][1]
    char_mask_b = batch["char_mask"][1]
    assert attn_b.sum().item() == 1 + path_len + 1 + char_len  # all attended
    assert torch.all(char_mask_b == 1)

    # Length supervision: only the zero-text-attention view should be supervised
    assert batch["length_supervise_mask"].tolist() == [1, 0]
    # Swipable length counts only a-z0-9 (no punctuation)
    assert batch["length_target"].tolist() == [len("test"), len("test")]


def test_length_target_ignores_punctuation_and_only_supervises_zero_text_view():
    tokenizer = CharacterTokenizer()
    path_len = 2
    char_len = 6  # includes punctuation + EOS + padding

    sample = {
        "word": "a,b!",
        "path_coords": torch.ones(path_len, 3),
        "path_mask": torch.ones(path_len, dtype=torch.long),
        "char_tokens": torch.tensor([tokenizer.cls_token_id + i + 1 for i in range(char_len)]),
        "char_mask": torch.ones(char_len, dtype=torch.long),
    }

    collator = PairwiseMaskedCollator(
        tokenizer=tokenizer,
        mask_path=True,
        modality_prob=1.0,
        zero_text_attention_prob=1.0,
    )

    batch = collator([sample])

    # Swipable characters: "a" and "b" -> length 2, punctuation ignored
    assert batch["length_target"].tolist() == [2, 2]
    assert batch["length_supervise_mask"].tolist() == [1, 0]
