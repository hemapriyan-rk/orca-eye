"""
Unit Tests for Dual-Primary Camera Geometry, Overlap Detection & Steering Avoidance
===================================================================================
Validates:
  1. Dual-primary symmetric camera models (-17.5 deg and +17.5 deg).
  2. Center overlap geometry: [-15.0 deg, +15.0 deg] centered at 0.0 deg (-left, 0, right+).
  3. Steering avoidance: Choosing SLIGHT_RIGHT or SLIGHT_LEFT instead of false STOP when frontal is blocked.
  4. Emergency STOP when obstacle is critically close (< 0.45m) or all corridors blocked.
  5. SpatialMap dual-primary vectorized update.
"""

import math
import numpy as np
import pytest

from navigation.camera_observation import CameraModel
from navigation.decision import DecisionMaker, NavigationReason
from navigation.path_generator import PathCandidate
from navigation.spatial_map import SpatialMap
from perception.freespace import FreeSpaceResult
from perception.depth import DepthResult


def test_dual_primary_camera_symmetric_geometry():
    """Verify symmetric primary camera geometry and 0-deg center overlap."""
    cam0 = CameraModel(camera_id="CAM0", yaw_deg=-17.5, half_hfov_rad=math.radians(32.5))
    cam1 = CameraModel(camera_id="CAM1", yaw_deg=17.5, half_hfov_rad=math.radians(32.5))

    # CAM0: [-50 deg, +15 deg]
    assert math.isclose(math.degrees(cam0.min_world_bearing_rad), -50.0, abs_tol=1e-3)
    assert math.isclose(math.degrees(cam0.max_world_bearing_rad), 15.0, abs_tol=1e-3)

    # CAM1: [-15 deg, +50 deg]
    assert math.isclose(math.degrees(cam1.min_world_bearing_rad), -15.0, abs_tol=1e-3)
    assert math.isclose(math.degrees(cam1.max_world_bearing_rad), 50.0, abs_tol=1e-3)

    # Overlap: [-15 deg, +15 deg] centered at 0 deg
    min_ov, max_ov, width = cam0.compute_overlap(cam1)
    assert math.isclose(math.degrees(min_ov), -15.0, abs_tol=1e-3)
    assert math.isclose(math.degrees(max_ov), 15.0, abs_tol=1e-3)
    assert math.isclose(math.degrees(width), 30.0, abs_tol=1e-3)

    # 0 deg (center heading) is in BOTH FOVs
    assert cam0.is_world_bearing_in_fov(0.0) is True
    assert cam1.is_world_bearing_in_fov(0.0) is True

    # -30 deg (LEFT) is in CAM0 only
    assert cam0.is_world_bearing_in_fov(math.radians(-30.0)) is True
    assert cam1.is_world_bearing_in_fov(math.radians(-30.0)) is False

    # +30 deg (RIGHT) is in CAM1 only
    assert cam0.is_world_bearing_in_fov(math.radians(30.0)) is False
    assert cam1.is_world_bearing_in_fov(math.radians(30.0)) is True


def test_steering_avoidance_when_frontal_blocked():
    """Verify system chooses SLIGHT_RIGHT instead of false STOP when frontal is blocked."""
    maker = DecisionMaker({"min_go_score": 0.35, "min_clearance": 0.20})

    cand_straight = PathCandidate(
        direction="STRAIGHT", angle_deg=0.0, points=[(10, 10), (8, 10)],
        clearance=0.15, progress=0.6, curvature=0.0, score=0.22, uncertainty=0.05,
    )
    cand_slight_right = PathCandidate(
        direction="SLIGHT_RIGHT", angle_deg=15.0, points=[(10, 10), (8, 12)],
        clearance=0.80, progress=1.0, curvature=0.33, score=0.74, uncertainty=0.05,
    )
    cand_slight_left = PathCandidate(
        direction="SLIGHT_LEFT", angle_deg=-15.0, points=[(10, 10), (8, 8)],
        clearance=0.45, progress=0.8, curvature=0.33, score=0.48, uncertainty=0.05,
    )

    candidates = [cand_slight_right, cand_slight_left, cand_straight]

    # Frontal collision flagged (e.g. wall/obstacle at 1.1m ahead)
    wall_coll = {
        "is_frontal_collision": True,
        "frontal_clearance_3d_m": 1.10,
        "min_frontal_free": 0.18,
        "lateral_warning": None,
    }

    state = maker.decide(candidates, frame_id=1, wall_proximity=wall_coll)

    # Must STEER around the hazard to SLIGHT_RIGHT, not STOP!
    assert state.command == "SLIGHT_RIGHT"
    assert state.decision == "TURN"
    assert state.is_stop is False
    assert "[AVOIDANCE]" in state.reason


def test_emergency_stop_triggered_when_too_close():
    """Verify emergency STOP is triggered when obstacle is within physical contact distance (<0.45m)."""
    maker = DecisionMaker({"min_go_score": 0.35, "min_clearance": 0.20})

    cand_straight = PathCandidate(direction="STRAIGHT", angle_deg=0.0, score=0.8, clearance=0.8, uncertainty=0.05)
    cand_right = PathCandidate(direction="SLIGHT_RIGHT", angle_deg=15.0, score=0.7, clearance=0.7, uncertainty=0.05)

    # Immediate danger (<0.45m)
    wall_emergency = {
        "is_frontal_collision": True,
        "frontal_clearance_3d_m": 0.32,
        "min_frontal_free": 0.02,
        "lateral_warning": None,
    }

    state = maker.decide([cand_straight, cand_right], frame_id=2, wall_proximity=wall_emergency)
    assert state.command == "STOP"
    assert state.is_stop is True
    assert NavigationReason.WALL_COLLISION in state.reason


def test_stop_triggered_when_all_corridors_blocked():
    """Verify safe STOP fallback occurs when no alternative corridor is safe."""
    maker = DecisionMaker({"min_go_score": 0.35, "min_clearance": 0.20})

    # All candidates blocked
    c_straight = PathCandidate(direction="STRAIGHT", angle_deg=0.0, score=0.15, clearance=0.08, uncertainty=0.05)
    c_left = PathCandidate(direction="LEFT", angle_deg=-30.0, score=0.12, clearance=0.05, uncertainty=0.05)
    c_right = PathCandidate(direction="RIGHT", angle_deg=30.0, score=0.10, clearance=0.04, uncertainty=0.05)

    wall_coll = {
        "is_frontal_collision": True,
        "frontal_clearance_3d_m": 0.90,
        "min_frontal_free": 0.10,
    }

    state = maker.decide([c_straight, c_left, c_right], frame_id=3, wall_proximity=wall_coll)
    assert state.command == "STOP"
    assert state.is_stop is True


def test_spatial_map_update_dual():
    """Verify SpatialMap.update_dual combines CAM0 and CAM1 across 20 columns."""
    smap = SpatialMap({"grid_rows": 12, "grid_cols": 20}, frame_width=640, frame_height=360)

    # Mock CAM0: Clear on left
    lmap0 = np.zeros((360, 640), dtype=np.uint8)
    fp0 = np.full((360, 640), 0.85, dtype=np.float32)
    fs0 = FreeSpaceResult(label_map=lmap0, free_prob_map=fp0, uncertainty=0.1, statistics={})
    dm0 = np.ones((360, 640), dtype=np.float32)
    dpr0 = DepthResult(depth_map=dm0, raw_depth=dm0, inference_ms=1.0, statistics={}, is_valid=True)

    # Mock CAM1: Obstacle on right (label 1)
    lmap1 = np.ones((360, 640), dtype=np.uint8)
    fp1 = np.full((360, 640), 0.10, dtype=np.float32)
    fs1 = FreeSpaceResult(label_map=lmap1, free_prob_map=fp1, uncertainty=0.2, statistics={})
    dm1 = np.full((360, 640), 0.3, dtype=np.float32)
    dpr1 = DepthResult(depth_map=dm1, raw_depth=dm1, inference_ms=1.0, statistics={}, is_valid=True)

    smap.update_dual(
        freespace_result0=fs0, depth_result0=dpr0, tracks0=[],
        freespace_result1=fs1, depth_result1=dpr1, tracks1=[],
    )

    # Left flank (cols 0..6) should have higher free probability (from CAM0)
    assert smap.free_arr[:, :5].mean() > 0.60

    # Right flank (cols 14..19) should have higher occupancy (from CAM1 obstacle)
    assert smap.occ_arr[:, 14:].mean() > 0.05
