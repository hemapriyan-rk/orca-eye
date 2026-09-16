"""
ORCA EYE — Unit Tests: SpatialMap
Verifies grid construction, update, and accessor methods.
"""

import numpy as np
import pytest
from perception.freespace import FreeSpaceResult, LABEL_FREE, LABEL_OBSTACLE, LABEL_UNKNOWN
from perception.depth import DepthResult


def make_cfg():
    return {
        "grid_rows": 4,
        "grid_cols": 6,
        "occupancy_decay": 0.85,
        "uncertainty_growth": 0.05,
    }


def make_freespace(H=480, W=640, label=LABEL_FREE):
    label_map = np.full((H, W), label, dtype=np.uint8)
    free_prob_map = np.ones((H, W), dtype=np.float32) * (0.8 if label == LABEL_FREE else 0.2)
    return FreeSpaceResult(
        label_map=label_map,
        free_prob_map=free_prob_map,
        uncertainty=0.0 if label == LABEL_FREE else 0.5,
        statistics={
            "free_fraction": 1.0 if label == LABEL_FREE else 0.0,
            "obstacle_fraction": 1.0 if label == LABEL_OBSTACLE else 0.0,
            "unknown_fraction": 0.0,
            "uncertainty": 0.0,
        },
    )


def make_depth(H=480, W=640, value=0.8):
    dm = np.full((H, W), value, dtype=np.float32)
    return DepthResult(
        depth_map=dm,
        raw_depth=dm,
        inference_ms=0.0,
        statistics={"min": value, "max": value, "mean": value, "std": 0.0, "uncertainty": 0.0},
        is_valid=True,
    )


def test_grid_created():
    from navigation.spatial_map import SpatialMap
    sm = SpatialMap(make_cfg(), frame_width=640, frame_height=480)
    assert sm.rows == 4
    assert sm.cols == 6
    # All cells exist
    for r in range(4):
        for c in range(6):
            cell = sm.get_cell(r, c)
            assert cell is not None


def test_out_of_bounds_returns_none():
    from navigation.spatial_map import SpatialMap
    sm = SpatialMap(make_cfg(), frame_width=640, frame_height=480)
    assert sm.get_cell(100, 0) is None
    assert sm.get_cell(0, 100) is None


def test_update_free_scene():
    from navigation.spatial_map import SpatialMap
    sm = SpatialMap(make_cfg(), frame_width=640, frame_height=480)
    fs = make_freespace(label=LABEL_FREE)
    depth = make_depth(value=0.8)
    sm.update(fs, depth, [])
    # Most cells should have low occupancy after free scene
    occ = sm.to_occupancy_array()
    assert occ.mean() < 0.3, f"Expected low occupancy, got {occ.mean()}"


def test_update_obstacle_scene():
    from navigation.spatial_map import SpatialMap
    sm = SpatialMap(make_cfg(), frame_width=640, frame_height=480)
    fs = make_freespace(label=LABEL_OBSTACLE)
    depth = make_depth(value=0.1)
    sm.update(fs, depth, [])
    occ = sm.to_occupancy_array()
    assert occ.mean() > 0.1, f"Expected elevated occupancy, got {occ.mean()}"


def test_to_dict():
    from navigation.spatial_map import SpatialMap
    sm = SpatialMap(make_cfg())
    d = sm.to_dict()
    assert "grid_rows" in d
    assert "grid_cols" in d
    assert "mean_occupancy" in d
