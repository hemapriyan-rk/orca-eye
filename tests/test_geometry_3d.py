"""
Unit Tests for 3D Egocentric Geometry & Free-Space Corridor Engine
===================================================================
Tests:
  1. Optical ray backprojection and egocentric 3D coordinate mapping.
  2. Ground/floor plane estimation on horizontal terrain.
  3. Protruding 3D obstacle segmentation (distinguishing floor vs obstacles).
  4. 3D corridor clearance evaluation and directional recommendation.
  5. Frontal collision detection for near obstacles vs open hallway floor.
"""

import math
import numpy as np
import pytest

from perception.calibration import CameraCalibration
from perception.geometry_3d import Geometry3D, Corridor3DProfile, Geometry3DResult


@pytest.fixture
def calibrated_cam():
    return CameraCalibration.from_fov(frame_width=360, frame_height=640, hfov_deg=65.0)


@pytest.fixture
def geom_engine(calibrated_cam):
    return Geometry3D(calibrated_cam, corridor_width_m=0.80, stop_distance_m=1.00)


def test_geometry_3d_initialization_and_ray_grid(geom_engine, calibrated_cam):
    """Verify 3D geometry engine initializes with correct optics and candidate directions."""
    assert geom_engine.calib.frame_width == 360
    assert geom_engine.calib.frame_height == 640
    assert "STRAIGHT" in geom_engine.candidate_directions
    assert "LEFT" in geom_engine.candidate_directions
    assert "RIGHT" in geom_engine.candidate_directions

    # Check ray grid creation
    geom_engine._init_ray_grid(640, 360)
    assert geom_engine._ray_x is not None
    assert geom_engine._ray_z is not None
    assert geom_engine._ray_x.shape[0] == 640 // geom_engine.step
    assert geom_engine._ray_x.shape[1] == 360 // geom_engine.step


def test_open_flat_floor_no_obstacles(geom_engine):
    """
    Synthetic depth map simulating an open hallway floor.
    Bottom of image is near ground (depth=0.1), top of corridor is far (depth=0.9).
    Verify that an open flat floor produces NO frontal collision and recommends STRAIGHT.
    """
    H, W = 640, 360
    # Create smooth linear depth ramp from bottom to top (floor perspective)
    rows = np.linspace(0.85, 0.10, H).reshape(H, 1).astype(np.float32)
    depth_map = np.repeat(rows, W, axis=1)

    res = geom_engine.analyze(depth_map)

    assert isinstance(res, Geometry3DResult)
    # The floor must not trigger an obstacle collision
    assert res.is_frontal_collision is False
    assert res.frontal_clearance_m >= 2.0
    assert res.best_direction == "STRAIGHT"
    assert res.corridors["STRAIGHT"].is_blocked is False


def test_near_obstacle_triggers_collision(geom_engine):
    """
    Synthetic depth map with a close obstacle (e.g. box / trash can / wall)
    directly in the center corridor at distance < 1.0m.
    Verify that 3D geometry flags frontal collision.
    """
    H, W = 640, 360
    # Background: far floor
    depth_map = np.ones((H, W), dtype=np.float32) * 0.70
    # Floor ramp at bottom
    depth_map[int(H * 0.7):, :] = np.linspace(0.4, 0.1, H - int(H * 0.7))[:, None]

    # Insert a protruding object in the lower-middle center:
    # row 300..450 (torso/eye level), cols 130..230 (central path) with near depth 0.05 (<1.0m)
    depth_map[300:460, 130:230] = 0.04

    res = geom_engine.analyze(depth_map)

    # Obstacle is right in center at near distance (< 1.0m)
    assert res.is_frontal_collision is True
    assert res.corridors["STRAIGHT"].is_blocked is True
    assert res.corridors["STRAIGHT"].clearance_m < 1.20


def test_lateral_obstacle_steers_away(geom_engine):
    """
    Obstacle blocked on the left flank, right flank is completely clear.
    Verify the 3D geometry engine recommends steering toward the open side.
    """
    H, W = 640, 360
    depth_map = np.ones((H, W), dtype=np.float32) * 0.80
    depth_map[int(H * 0.7):, :] = 0.25  # floor

    # Place a large obstacle on the left side: cols 0..150
    depth_map[200:500, 0:140] = 0.05  # near obstacle on left

    res = geom_engine.analyze(depth_map)

    # Left corridor should have lower clearance / be blocked
    left_clr = res.corridors["LEFT"].clearance_m
    right_clr = res.corridors["RIGHT"].clearance_m
    assert right_clr > left_clr
    assert res.corridors["RIGHT"].is_blocked is False
