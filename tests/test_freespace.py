"""
ORCA EYE — Unit Tests: FreeSpaceEstimator
Verifies geometric prior, obstacle masking, and uncertainty reporting.
"""

import numpy as np
import pytest
from perception.freespace import FreeSpaceEstimator, LABEL_FREE, LABEL_OBSTACLE, LABEL_UNKNOWN
from perception.depth import DepthResult
from perception.detector import ObjectState


def make_cfg():
    return {
        "walkable_bottom_fraction": 0.55,
        "obstacle_inflation": 1.05,
        "depth_gradient_threshold": 0.08,
        "near_depth_threshold": 0.30,
        "free_probability_threshold": 0.55,
    }


def make_depth(H=480, W=640, value=0.8, valid=True):
    dm = np.full((H, W), value, dtype=np.float32)
    return DepthResult(
        depth_map=dm,
        raw_depth=dm,
        inference_ms=0.0,
        statistics={"min": value, "max": value, "mean": value, "std": 0.0, "uncertainty": 0.0},
        is_valid=valid,
    )


def make_detection(x1, y1, x2, y2, class_name="person"):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return ObjectState(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=[x1, y1, x2, y2],
        center=[cx, cy],
        area=float((x2 - x1) * (y2 - y1)),
    )


def test_output_shape():
    estimator = FreeSpaceEstimator(make_cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_depth()
    result = estimator.estimate(frame, depth, [])
    assert result.label_map.shape == (480, 640)
    assert result.free_prob_map.shape == (480, 640)
    assert 0.0 <= result.uncertainty <= 1.0


def test_labels_are_valid():
    estimator = FreeSpaceEstimator(make_cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_depth()
    result = estimator.estimate(frame, depth, [])
    unique_labels = set(np.unique(result.label_map).tolist())
    assert unique_labels.issubset({0, 1, 2}), f"Invalid labels: {unique_labels}"


def test_obstacle_box_is_marked():
    estimator = FreeSpaceEstimator(make_cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_depth(value=0.8)  # far depth — no near-depth obstacle masking
    # Detection covering a clear region
    det = make_detection(100, 100, 300, 400)
    result = estimator.estimate(frame, depth, [det])
    # The center of the bbox should be marked as OBSTACLE
    cy = (100 + 400) // 2
    cx = (100 + 300) // 2
    assert result.label_map[cy, cx] == LABEL_OBSTACLE, \
        f"Expected OBSTACLE at ({cy},{cx}), got {result.label_map[cy, cx]}"


def test_bottom_is_more_free_than_top():
    """Geometric prior: bottom of image should have higher free probability."""
    estimator = FreeSpaceEstimator(make_cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_depth(value=0.8)  # far = walkable
    result = estimator.estimate(frame, depth, [])
    bottom_mean = result.free_prob_map[400:, :].mean()
    top_mean = result.free_prob_map[:100, :].mean()
    assert bottom_mean > top_mean, \
        f"Bottom ({bottom_mean:.3f}) should be > top ({top_mean:.3f})"


def test_statistics_keys():
    estimator = FreeSpaceEstimator(make_cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_depth()
    result = estimator.estimate(frame, depth, [])
    for key in ("free_fraction", "obstacle_fraction", "unknown_fraction", "uncertainty"):
        assert key in result.statistics, f"Missing key: {key}"


def test_column_freespace_length():
    estimator = FreeSpaceEstimator(make_cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_depth()
    result = estimator.estimate(frame, depth, [])
    col_fs = estimator.get_column_freespace(result.label_map, n_cols=20)
    assert len(col_fs) == 20
    assert all(0.0 <= v <= 1.0 for v in col_fs)
