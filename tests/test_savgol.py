"""Test SavgolPreprocessor (PyTorch) matches NumPy preprocessing."""

import numpy as np
import torch

from swipealot.data.preprocessing import preprocess_raw_path_to_sg_features
from swipealot.models.savgol import SavgolPreprocessor


def _make_raw_path(n: int = 30) -> list[dict]:
    """Create a simple curved trajectory for testing."""
    t_vals = np.linspace(0, 100, n)
    path = []
    for i, t in enumerate(t_vals):
        frac = i / max(n - 1, 1)
        path.append(
            {
                "x": float(np.clip(0.1 + 0.6 * frac + 0.05 * np.sin(4 * np.pi * frac), 0, 1)),
                "y": float(np.clip(0.2 + 0.3 * frac + 0.03 * np.cos(4 * np.pi * frac), 0, 1)),
                "t": float(t),
            }
        )
    return path


def test_savgol_pytorch_matches_numpy():
    """SavgolPreprocessor output matches NumPy preprocess_raw_path_to_sg_features."""
    max_len = 32
    raw_path = _make_raw_path(n=50)

    # --- NumPy path ---
    np_features, np_mask = preprocess_raw_path_to_sg_features(
        raw_path, max_len, resample_mode="time"
    )
    assert np_features.shape == (max_len, 8)
    assert np_mask.shape == (max_len,)

    # --- PyTorch path ---
    # SavgolPreprocessor expects already-resampled (x, y) in [B, 2, T].
    # Extract x, y from the NumPy features (channels 0, 1) to feed the same
    # resampled positions.
    x_resampled = np_features[:, 0]  # [T]
    y_resampled = np_features[:, 1]  # [T]
    xy = torch.tensor(
        np.stack([x_resampled, y_resampled], axis=0),
        dtype=torch.float32,
    ).unsqueeze(0)  # [1, 2, T]

    sg = SavgolPreprocessor()
    sg.eval()
    with torch.no_grad():
        pt_features = sg(xy)  # [1, T, 8]

    pt_features_np = pt_features.squeeze(0).numpy()  # [T, 8]

    # Compare each channel
    channel_names = ["x", "y", "dx", "dy", "d2x", "d2y", "speed", "curvature"]
    for i, name in enumerate(channel_names):
        np_ch = np_features[:, i]
        pt_ch = pt_features_np[:, i]
        max_diff = float(np.max(np.abs(np_ch - pt_ch)))
        assert max_diff < 1e-4, (
            f"Channel {name} (idx {i}) max diff = {max_diff:.6f}, exceeds tolerance"
        )


def test_savgol_output_shape():
    """SavgolPreprocessor produces [B, T, 8] output."""
    sg = SavgolPreprocessor()
    xy = torch.randn(3, 2, 64)
    out = sg(xy)
    assert out.shape == (3, 64, 8)


def test_savgol_empty_path_numpy():
    """NumPy preprocessing returns zeros for empty paths."""
    features, mask = preprocess_raw_path_to_sg_features([], 16)
    assert features.shape == (16, 8)
    assert mask.shape == (16,)
    assert np.all(features == 0)
    assert np.all(mask == 0)
