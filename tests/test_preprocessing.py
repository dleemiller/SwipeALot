"""Tests for preprocessing utilities in swipealot.data.dataset."""

import numpy as np

from swipealot.data.dataset import normalize_coordinates, sample_path_points


def test_normalize_coordinates_clamps_and_scales():
    points = [
        {"x": 1.2, "y": -0.1, "t": 5.0},
        {"x": 0.5, "y": 0.5, "t": 10.0},
        {"x": 0.0, "y": 1.5, "t": 15.0},
    ]
    normalized = normalize_coordinates(points, canvas_width=1.0, canvas_height=1.0)

    xs = [p["x"] for p in normalized]
    ys = [p["y"] for p in normalized]
    ts = [p["t"] for p in normalized]

    assert all(0.0 <= x <= 1.0 for x in xs)
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert min(ts) == 0.0 and max(ts) == 1.0


def test_sample_path_points_padding_and_masking():
    # Fewer points than max_len -> padding
    coords, mask = sample_path_points(
        [{"x": 0.1, "y": 0.2, "t": 0.0}, {"x": 0.2, "y": 0.3, "t": 0.5}], max_len=4
    )
    assert coords.shape == (4, 3)
    assert mask.tolist() == [1, 1, 0, 0]

    # More points than max_len -> downsample
    many_points = [{"x": float(i), "y": float(i + 1), "t": float(i)} for i in range(6)]
    coords_down, mask_down = sample_path_points(many_points, max_len=3)
    assert coords_down.shape == (3, 3)
    assert mask_down.tolist() == [1, 1, 1]
    # Interpolated values should be within the original range
    assert np.all(coords_down[:, 0] >= min(p["x"] for p in many_points))
    assert np.all(coords_down[:, 0] <= max(p["x"] for p in many_points))

