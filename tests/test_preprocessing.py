"""Tests for preprocessing utilities in `swipealot.data.preprocessing`."""

import numpy as np
import pytest

from swipealot.data.preprocessing import (
    normalize_and_compute_features,
    sample_path_points_with_features,
)

# Tests for motion-aware features


def test_normalize_and_compute_features_basic():
    """Test basic feature computation: dx, dy, dt, log_dt."""
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},
        {"x": 0.5, "y": 0.5, "t": 10.0},
        {"x": 1.0, "y": 1.0, "t": 20.0},
    ]
    features = normalize_and_compute_features(points)

    # First point should have dx=dy=dt=0
    assert features[0]["dx"] == 0.0
    assert features[0]["dy"] == 0.0
    assert features[0]["dt"] == 0.0
    assert features[0]["log_dt"] == np.log1p(0.0)

    # Second point deltas
    assert features[1]["dx"] == pytest.approx(0.5)
    assert features[1]["dy"] == pytest.approx(0.5)
    assert features[1]["dt"] == pytest.approx(10.0)
    assert features[1]["log_dt"] == pytest.approx(np.log1p(10.0))

    # Third point deltas
    assert features[2]["dx"] == pytest.approx(0.5)
    assert features[2]["dy"] == pytest.approx(0.5)
    assert features[2]["dt"] == pytest.approx(10.0)


def test_normalize_and_compute_features_dt_clamping():
    """Test that dt is clamped to reasonable range."""
    points = [
        {"x": 0.0, "y": 0.0, "t": 0.0},
        {"x": 0.1, "y": 0.1, "t": 1.0},  # dt=1.0 < dt_clamp_min_ms=5.0
        {"x": 0.2, "y": 0.2, "t": 1001.0},  # dt=1000.0 > dt_clamp_max_ms=66.0
    ]
    features = normalize_and_compute_features(
        points,
        dt_clamp_min_ms=5.0,
        dt_clamp_max_ms=66.0,
    )

    # Second point: dt should be clamped to min=5.0
    assert features[1]["dt"] == pytest.approx(5.0)

    # Third point: dt should be clamped to max=66.0
    assert features[2]["dt"] == pytest.approx(66.0)


def test_normalize_and_compute_features_coordinate_clamping():
    """Test that x, y coordinates are clamped to [0, 1]."""
    points = [
        {"x": -0.5, "y": 1.5, "t": 0.0},
        {"x": 0.5, "y": 0.5, "t": 10.0},
    ]
    features = normalize_and_compute_features(points)

    # First point coordinates should be clamped
    assert features[0]["x"] == 0.0  # Clamped from -0.5
    assert features[0]["y"] == 1.0  # Clamped from 1.5

    # Second point normal
    assert features[1]["x"] == 0.5
    assert features[1]["y"] == 0.5


def test_sample_path_points_with_features_shapes_and_endpoints():
    points = [
        {"x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0, "ds": 0.0, "dt": 0.0, "log_dt": 0.0},
        {
            "x": 0.5,
            "y": 0.5,
            "dx": 0.5,
            "dy": 0.5,
            "ds": np.sqrt(0.5**2 + 0.5**2),
            "dt": 10.0,
            "log_dt": np.log1p(10.0),
        },
        {
            "x": 1.0,
            "y": 1.0,
            "dx": 0.5,
            "dy": 0.5,
            "ds": np.sqrt(0.5**2 + 0.5**2),
            "dt": 10.0,
            "log_dt": np.log1p(10.0),
        },
    ]
    features, mask = sample_path_points_with_features(points, max_len=3)

    assert features.shape == (3, 6)
    assert mask.shape == (3,)
    assert np.all(mask == 1)

    # Preserve endpoints under resampling
    assert features[0, 0] == pytest.approx(0.0)
    assert features[0, 1] == pytest.approx(0.0)
    assert features[-1, 0] == pytest.approx(1.0)
    assert features[-1, 1] == pytest.approx(1.0)


def test_sample_path_points_with_features_upsampling():
    """Test upsampling short paths via interpolation."""
    points = [
        {"x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0, "ds": 0.0, "dt": 0.0, "log_dt": 0.0},
        {
            "x": 1.0,
            "y": 1.0,
            "dx": 1.0,
            "dy": 1.0,
            "ds": np.sqrt(2.0),
            "dt": 10.0,
            "log_dt": np.log1p(10.0),
        },
    ]
    features, mask = sample_path_points_with_features(points, max_len=4)

    assert features.shape == (4, 6)
    assert mask.shape == (4,)
    assert np.all(mask == 1)  # All valid (no padding)

    # Interpolation should create smooth transitions
    # First point should be unchanged
    assert features[0, 0] == pytest.approx(0.0)  # x
    assert features[0, 1] == pytest.approx(0.0)  # y

    # Last point should be unchanged
    assert features[3, 0] == pytest.approx(1.0)  # x
    assert features[3, 1] == pytest.approx(1.0)  # y

    # Middle points should be interpolated
    assert 0.0 < features[1, 0] < 1.0
    assert 0.0 < features[2, 0] < 1.0


def test_sample_path_points_with_features_downsampling():
    """Test downsampling long paths via interpolation."""
    # Create a longer path
    dt = 10.0
    log_dt = np.log1p(dt)
    points = []
    prev_x = 0.0
    prev_y = 0.0
    for i in range(10):
        x = float(i) / 9
        y = float(i) / 9
        dx = 0.0 if i == 0 else x - prev_x
        dy = 0.0 if i == 0 else y - prev_y
        ds = np.sqrt(dx**2 + dy**2)
        points.append(
            {
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                "ds": ds,
                "dt": 0.0 if i == 0 else dt,
                "log_dt": 0.0 if i == 0 else log_dt,
            }
        )
        prev_x = x
        prev_y = y

    features, mask = sample_path_points_with_features(points, max_len=5)

    assert features.shape == (5, 6)
    assert mask.shape == (5,)
    assert np.all(mask == 1)  # All valid

    # Should preserve start and end points
    assert features[0, 0] == pytest.approx(0.0)
    assert features[4, 0] == pytest.approx(1.0)


def test_sample_path_points_with_features_empty_path():
    """Test handling of empty paths."""
    features, mask = sample_path_points_with_features([], max_len=3)

    assert features.shape == (3, 6)
    assert mask.shape == (3,)
    assert np.all(features == 0.0)
    assert np.all(mask == 0)  # All masked


def test_feature_interpolation_preserves_structure():
    """Test that interpolation preserves feature relationships."""
    # Create a simple linear path
    points = [
        {"x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0, "ds": 0.0, "dt": 0.0, "log_dt": 0.0},
        {
            "x": 0.5,
            "y": 0.5,
            "dx": 0.5,
            "dy": 0.5,
            "ds": np.sqrt(0.5**2 + 0.5**2),
            "dt": 10.0,
            "log_dt": np.log1p(10.0),
        },
        {
            "x": 1.0,
            "y": 1.0,
            "dx": 0.5,
            "dy": 0.5,
            "ds": np.sqrt(0.5**2 + 0.5**2),
            "dt": 10.0,
            "log_dt": np.log1p(10.0),
        },
    ]
    features, mask = sample_path_points_with_features(points, max_len=5)

    # All features should be interpolated independently
    assert features.shape == (5, 6)

    # x should be monotonically increasing for this linear path
    x_values = features[:, 0]
    assert np.all(np.diff(x_values) >= 0)

    # y should also be monotonically increasing
    y_values = features[:, 1]
    assert np.all(np.diff(y_values) >= 0)


def test_spatial_resampling_is_approximately_uniform_in_distance():
    points = [
        {"x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0, "ds": 0.0, "dt": 0.0, "log_dt": 0.0},
        {"x": 0.1, "y": 0.0, "dx": 0.1, "dy": 0.0, "ds": 0.1, "dt": 10.0, "log_dt": np.log1p(10.0)},
        {"x": 0.9, "y": 0.0, "dx": 0.8, "dy": 0.0, "ds": 0.8, "dt": 10.0, "log_dt": np.log1p(10.0)},
        {"x": 1.0, "y": 0.0, "dx": 0.1, "dy": 0.0, "ds": 0.1, "dt": 10.0, "log_dt": np.log1p(10.0)},
    ]
    features, _ = sample_path_points_with_features(points, max_len=5)

    dx = features[:, 2]
    dy = features[:, 3]
    ds = features[:, 4]
    # ds is computed from dx/dy in resampling
    assert np.allclose(ds, np.sqrt(dx**2 + dy**2), atol=1e-5)

    # For a straight line, distances should be ~uniform except the first step (0)
    steps = ds[1:]
    assert np.max(steps) - np.min(steps) < 1e-2
