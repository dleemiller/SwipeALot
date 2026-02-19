"""Evaluation metrics for attention shaping.

Ported from encodercnn AttentionMetrics — measures how well the predicted
attention matches the teacher's char->path attention patterns.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score


def compute_attention_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    char_mask: np.ndarray,
) -> dict[str, float]:
    """Compute all attention quality metrics.

    Args:
        pred: [N, 26, 128] predicted attention
        target: [N, 26, 128] target attention
        char_mask: [N, 26] valid character mask (bool)

    Returns:
        Dictionary of metric name -> value
    """
    results = {}

    # --- KL divergence (renormalized per-char distributions) ---
    eps = 1e-8
    pred_norm = pred / (pred.sum(axis=-1, keepdims=True) + eps)
    target_norm = target / (target.sum(axis=-1, keepdims=True) + eps)

    kl = (target_norm * (np.log(target_norm + eps) - np.log(pred_norm + eps))).sum(
        axis=-1
    )  # [N, 26]
    kl_masked = kl * char_mask
    num_valid = char_mask.sum()
    results["kl_mean"] = float(kl_masked.sum() / (num_valid + eps))

    # --- Entropy ---
    pred_ent = -(pred_norm * np.log(pred_norm + eps)).sum(axis=-1)  # [N, 26]
    target_ent = -(target_norm * np.log(target_norm + eps)).sum(axis=-1)  # [N, 26]
    results["pred_entropy"] = float((pred_ent * char_mask).sum() / (num_valid + eps))
    results["target_entropy"] = float((target_ent * char_mask).sum() / (num_valid + eps))

    # --- Top-k IoU (top 10% overlap) ---
    k = max(1, int(128 * 0.1))
    ious = []
    for b in range(pred.shape[0]):
        for c in range(26):
            if not char_mask[b, c]:
                continue
            pred_topk = set(np.argpartition(pred[b, c], -k)[-k:])
            target_topk = set(np.argpartition(target[b, c], -k)[-k:])
            intersection = len(pred_topk & target_topk)
            union = len(pred_topk | target_topk)
            ious.append(intersection / union if union > 0 else 0.0)
    results["top_k_iou"] = float(np.mean(ious)) if ious else 0.0

    # --- Ordering tau (Kendall's tau on dominant peak positions) ---
    pred_peaks = pred.argmax(axis=-1)  # [N, 26]
    target_peaks = target.argmax(axis=-1)  # [N, 26]
    taus = []
    for b in range(pred.shape[0]):
        valid_chars = np.where(char_mask[b])[0]
        if len(valid_chars) < 2:
            continue
        tau, _ = kendalltau(target_peaks[b, valid_chars], pred_peaks[b, valid_chars])
        if not np.isnan(tau):
            taus.append(tau)
    results["ordering_tau"] = float(np.mean(taus)) if taus else 0.0

    # --- Peak MAE ---
    all_mae = []
    all_count_errors = []
    threshold = 0.3
    for b in range(pred.shape[0]):
        for c in range(26):
            if not char_mask[b, c]:
                continue
            pred_dist = pred[b, c]
            target_dist = target[b, c]
            pred_max = pred_dist.max()
            target_max = target_dist.max()
            if pred_max < 1e-6 or target_max < 1e-6:
                continue
            pp, _ = find_peaks(pred_dist, height=pred_max * threshold)
            tp, _ = find_peaks(target_dist, height=target_max * threshold)
            if len(pp) == 0:
                pp = np.array([pred_dist.argmax()])
            if len(tp) == 0:
                tp = np.array([target_dist.argmax()])
            all_count_errors.append(abs(len(pp) - len(tp)))
            for p in pp:
                all_mae.append(np.min(np.abs(tp - p)))
    results["peak_mae_mean"] = float(np.mean(all_mae)) if all_mae else 0.0
    results["peak_mae_max"] = float(np.max(all_mae)) if all_mae else 0.0
    results["peak_count_error"] = float(np.mean(all_count_errors)) if all_count_errors else 0.0

    # --- Character detection (valid vs invalid separation) ---
    char_attention = pred.sum(axis=-1)  # [N, 26]
    scores = char_attention.flatten()
    labels = char_mask.flatten().astype(int)

    invalid_attn = char_attention[~char_mask.astype(bool)]
    valid_attn = char_attention[char_mask.astype(bool)]

    results["valid_mean"] = float(valid_attn.mean()) if len(valid_attn) > 0 else 0.0
    results["invalid_mean"] = float(invalid_attn.mean()) if len(invalid_attn) > 0 else 0.0

    attention_ratio = results["valid_mean"] / (results["invalid_mean"] + eps)
    results["snr_db"] = float(20 * np.log10(attention_ratio + eps))

    try:
        if labels.sum() > 0 and labels.sum() < len(labels):
            results["char_auc"] = float(roc_auc_score(labels, scores))
        else:
            results["char_auc"] = 1.0
    except ValueError:
        results["char_auc"] = 0.5

    return results
