import numpy as np

from swipealot.data import CharacterTokenizer
from swipealot.training.trainer import create_compute_metrics_fn


class _EvalPred:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


def test_compute_metrics_includes_length_metrics_when_available():
    tokenizer = CharacterTokenizer()
    compute_metrics = create_compute_metrics_fn(tokenizer)

    batch = 4
    char_len = 6
    vocab = tokenizer.vocab_size

    # Perfect char predictions on supervised positions (non -100).
    char_logits = np.zeros((batch, char_len, vocab), dtype=np.float32)
    char_labels = np.full((batch, char_len), -100, dtype=np.int64)
    char_labels[:, 0] = 7
    char_logits[:, 0, 7] = 10.0

    # Length predictions/targets
    length_logits = np.array([3.0, 4.2, 5.0, 5.9], dtype=np.float32)
    length_target = np.array([3, 4, 6, 6], dtype=np.int64)
    length_mask = np.array([1, 1, 1, 1], dtype=np.int64)

    metrics = compute_metrics(
        _EvalPred(
            predictions=(char_logits, length_logits),
            label_ids=(char_labels, length_target, length_mask),
        )
    )

    assert "char_accuracy" in metrics
    assert "length_mae" in metrics
    assert "length_rmse" in metrics
    assert "length_acc_within_1" in metrics
