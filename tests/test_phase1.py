"""
ORCA EYE — Unit Tests: Phase 1 Enhancements
===========================================
Verifies:
  1. SpatialEntity creation, relative bearing, and depth integration.
  2. DynamicConflictEngine trajectory projection and prospective intersection.
  3. InstructionGenerator priority arbitration, emergency STOP preemption, and cooldown.
  4. NavigationState serialization and reason code standards.
"""

import time
import numpy as np
import pytest

from navigation.decision import DecisionMaker, NavigationReason, NavigationState
from navigation.dynamic_conflict import DynamicConflictEngine, DynamicConflict
from navigation.spatial_entity import SpatialEntity, create_spatial_entities
from navigation.audio_guidance import InstructionGenerator, AudioInstruction
from navigation.path_generator import PathCandidate
from perception.tracker import TrackedObject, TrackState


# ---------------------------------------------------------------------------
# Test 1: SpatialEntity Creation & Bearing
# ---------------------------------------------------------------------------

def test_spatial_entity_creation():
    trk = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        bbox=[100, 100, 200, 300],
        center=[150.0, 200.0],
        bottom_center=[150.0, 300.0],
        velocity=[-2.0, 4.0],
        motion_state="approaching",
        track_state=TrackState.CONFIRMED,
        confidence=0.88,
    )

    entities = create_spatial_entities(
        tracks=[trk],
        depth_map=None,
        depth_estimator=None,
        frame_w=640,
        frame_h=480,
        hfov_deg=60.0,
    )

    assert len(entities) == 1
    e = entities[0]
    assert e.track_id == 1
    assert e.class_name == "person"
    assert e.motion_direction == "approaching"
    assert e.ground_position == [150.0, 300.0]
    # Center of camera is 320. 150 is to the left: bearing should be negative
    assert e.relative_bearing < 0.0


# ---------------------------------------------------------------------------
# Test 2: DynamicConflictEngine
# ---------------------------------------------------------------------------

def test_dynamic_conflict_detection():
    engine = DynamicConflictEngine({"dynamic_conflict": {"enabled": True, "time_horizon_s": 2.0}})

    # Straight walking corridor through center of grid
    cand_straight = PathCandidate(
        direction="STRAIGHT",
        angle_deg=0.0,
        points=[(r, 10) for r in range(11, 4, -1)],  # bottom to middle
        clearance=0.8,
        width=2.0,
        progress=0.8,
        curvature=0.0,
        uncertainty=0.1,
        risk=0.1,
    )

    # Moving pedestrian crossing from left directly into corridor
    # At 640x480 grid 12x20: col 10 is at x = 10 * 32 + 16 = 336
    # Entity starts at x=200, y=300 with velocity vx=+10 px/frame, vy=0
    # Will hit x=336 in ~14 frames (at 30fps -> ~0.47s)
    entity = SpatialEntity(
        track_id=42,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=[180, 250, 220, 350],
        image_position=[200.0, 300.0],
        ground_position=[200.0, 300.0],
        relative_bearing=-10.0,
        relative_distance=0.4,
        velocity=[8.0, 0.0],  # moving right towards center corridor
        motion_direction="crossing",
        tracking_confidence=0.95,
        depth_confidence=0.85,
        depth_valid=True,
    )

    conflicts = engine.evaluate_conflicts(
        spatial_entities=[entity],
        candidates=[cand_straight],
        frame_w=640,
        frame_h=480,
        grid_rows=12,
        grid_cols=20,
        fps=30.0,
    )

    assert "STRAIGHT" in conflicts
    straight_conflicts = conflicts["STRAIGHT"]
    assert len(straight_conflicts) >= 1
    c = straight_conflicts[0]
    assert c.track_id == 42
    assert c.has_conflict is True
    assert c.time_to_conflict is not None
    assert c.time_to_conflict < 2.0
    assert cand_straight.dynamic_conflict_score > 0.0


# ---------------------------------------------------------------------------
# Test 3: Audio Instruction Priority & Suppression
# ---------------------------------------------------------------------------

def test_audio_priority_and_suppression():
    cfg = {
        "audio": {
            "enabled": False,
            "cooldown_seconds": 2.0,
            "reminder_seconds": 5.0,
        }
    }
    gen = InstructionGenerator(cfg)

    # Initial state: STRAIGHT
    state_straight = NavigationState(
        command="STRAIGHT",
        selected_path_direction="STRAIGHT",
        score=0.8,
        clearance=0.8,
        uncertainty=0.1,
        reason="Clear",
    )

    # Emergency STOP must generate Priority 1 immediately
    state_stop = NavigationState(
        command="STOP",
        selected_path_direction="STOP",
        score=0.0,
        clearance=0.0,
        uncertainty=1.0,
        reason=f"[{NavigationReason.CORRIDOR_BLOCKED}] Path blocked",
        is_stop=True,
    )

    stop_instr = gen.generate_instruction(state_stop)
    assert stop_instr is not None
    assert stop_instr.priority == 1
    assert stop_instr.preempt is True
    assert "Stop" in stop_instr.text

    # Direction change requires confirmation frames
    gen_dir = InstructionGenerator(cfg)
    state_left = NavigationState(
        command="LEFT",
        selected_path_direction="LEFT",
        score=0.7,
        clearance=0.7,
        uncertainty=0.1,
        reason="Clear",
    )

    # Frame 1 of change -> withheld
    i1 = gen_dir.generate_instruction(state_left)
    assert i1 is None

    # Frame 2 of change -> confirmed
    i2 = gen_dir.generate_instruction(state_left)
    assert i2 is not None
    assert i2.priority == 3
    assert "Turn left" in i2.text


# ---------------------------------------------------------------------------
# Test 4: NavigationState Serialization & Backward Compatibility
# ---------------------------------------------------------------------------

def test_navigation_state_serialization():
    maker = DecisionMaker({"min_go_score": 0.35, "min_clearance": 0.20})
    cand = PathCandidate(
        direction="STRAIGHT",
        angle_deg=0.0,
        points=[(10, 10), (9, 10)],
        clearance=0.75,
        width=2.5,
        progress=1.0,
        curvature=0.0,
        uncertainty=0.05,
        risk=0.1,
    )
    cand.score = 0.72
    cand.total_score = 0.72

    state = maker.decide([cand], frame_id=10)
    assert isinstance(state, NavigationState)
    assert state.command == "STRAIGHT"
    assert state.decision == "GO"
    assert state.decision_confidence > 0.5
    assert NavigationReason.CLEAR_PATH in state.reason

    d_dict = state.to_dict()
    assert "decision" in d_dict
    assert "dynamic_conflict" in d_dict
    assert "active_track_ids" in d_dict
    assert "wall_warning" in d_dict


# ---------------------------------------------------------------------------
# Test 5: Wall Proximity & Frontal Collision Detection in SpatialMap
# ---------------------------------------------------------------------------

def test_wall_proximity_detection():
    from navigation.spatial_map import SpatialMap
    from perception.freespace import FreeSpaceResult
    from perception.depth import DepthResult

    cfg = {"grid": {"rows": 12, "cols": 20, "cell_size_m": 0.25}}
    smap = SpatialMap(cfg)

    def _make_fs(lm):
        free_prob = np.where(lm == 1, 0.0, 0.85).astype(np.float32)
        return FreeSpaceResult(
            label_map=lm,
            free_prob_map=free_prob,
            uncertainty=0.1,
            statistics={"free_fraction": 0.8, "obstacle_fraction": 0.1, "unknown_fraction": 0.1},
        )

    def _make_depth(dm):
        return DepthResult(
            depth_map=dm,
            raw_depth=dm,
            inference_ms=1.0,
            statistics={"min": 0.5, "max": 5.0, "mean": 2.0, "std": 0.5, "uncertainty": 0.1},
            is_valid=True,
        )

    # Initially completely free
    free_lm = np.zeros((120, 200), dtype=np.uint8)
    depth_m = np.ones((120, 200), dtype=np.float32) * 2.0
    smap.update(_make_fs(free_lm), _make_depth(depth_m), [])

    wp_clear = smap.evaluate_wall_proximity()
    assert wp_clear["is_frontal_collision"] is False
    assert wp_clear["lateral_warning"] is None
    assert wp_clear["min_frontal_free"] > 0.50

    # Simulate frontal obstacle / wall right in near center:
    # Cells row >= 8, cols 8..11 set to obstacle
    obs_lm = free_lm.copy()
    obs_lm[80:, 80:120] = 1  # obstacle
    smap.update(_make_fs(obs_lm), _make_depth(depth_m), [])

    wp_frontal = smap.evaluate_wall_proximity()
    assert wp_frontal["is_frontal_collision"] is True
    assert wp_frontal["min_frontal_free"] < 0.35

    # Simulate left wall proximity:
    # Clear center, but left flank (cols 0..3) heavily occupied
    left_wall_lm = free_lm.copy()
    left_wall_lm[80:, :40] = 1
    smap.update(_make_fs(left_wall_lm), _make_depth(depth_m), [])

    wp_left = smap.evaluate_wall_proximity()
    assert wp_left["lateral_warning"] == "LEFT"

    # Simulate right wall proximity:
    right_wall_lm = free_lm.copy()
    right_wall_lm[80:, 160:] = 1
    smap.update(_make_fs(right_wall_lm), _make_depth(depth_m), [])

    wp_right = smap.evaluate_wall_proximity()
    assert wp_right["lateral_warning"] == "RIGHT"


# ---------------------------------------------------------------------------
# Test 6: Safety Decision Maker Wall Proximity and Collision Handling
# ---------------------------------------------------------------------------

def test_wall_collision_and_proximity_decisions():
    maker = DecisionMaker({"min_go_score": 0.35, "min_clearance": 0.20})
    cand = PathCandidate(
        direction="STRAIGHT",
        angle_deg=0.0,
        points=[(10, 10), (9, 10), (8, 10)],
        clearance=0.85,
        width=2.5,
        progress=1.0,
        curvature=0.0,
        uncertainty=0.05,
        risk=0.05,
    )
    cand.score = 0.82
    cand.total_score = 0.82

    # Frontal collision hazard -> must trigger emergency STOP
    wall_coll = {"is_frontal_collision": True, "lateral_warning": None, "min_frontal_free": 0.15}
    state_stop = maker.decide([cand], frame_id=1, wall_proximity=wall_coll)
    assert state_stop.command == "STOP"
    assert state_stop.is_stop is True
    assert NavigationReason.WALL_COLLISION in state_stop.reason

    # Lateral wall proximity on left -> triggers CAUTION
    wall_lat = {"is_frontal_collision": False, "lateral_warning": "LEFT", "min_frontal_free": 0.80}
    state_lat = maker.decide([cand], frame_id=2, wall_proximity=wall_lat)
    assert state_lat.decision == "CAUTION"
    assert state_lat.wall_warning == "LEFT"
    assert NavigationReason.WALL_PROXIMITY in state_lat.reason


# ---------------------------------------------------------------------------
# Test 7: Wall Alerts Audio Guidance Instructions
# ---------------------------------------------------------------------------

def test_wall_audio_instructions():
    cfg = {"audio": {"enabled": False, "cooldown_seconds": 1.0}}
    gen = InstructionGenerator(cfg)

    # Frontal wall emergency STOP
    state_stop = NavigationState(
        command="STOP",
        decision="STOP",
        selected_path_direction="STOP",
        score=0.0,
        clearance=0.1,
        uncertainty=1.0,
        reason=f"[{NavigationReason.WALL_COLLISION}] Frontal wall hazard",
        is_stop=True,
    )
    instr1 = gen.generate_instruction(state_stop)
    assert instr1 is not None
    assert instr1.priority == 1
    assert instr1.preempt is True
    assert "wall" in instr1.text.lower()

    # Lateral wall warning
    state_lat = NavigationState(
        command="SLIGHT_RIGHT",
        decision="CAUTION",
        selected_path_direction="SLIGHT_RIGHT",
        score=0.70,
        clearance=0.75,
        uncertainty=0.1,
        reason=f"[{NavigationReason.WALL_PROXIMITY}] Wall close on left",
        wall_warning="LEFT",
    )
    # Advance time past cooldown
    gen._last_instruction_time = 0.0
    gen._last_stop_time = 0.0
    instr2 = gen.generate_instruction(state_lat)
    assert instr2 is not None
    assert instr2.priority == 2
    assert "wall close on left" in instr2.text.lower()


# ---------------------------------------------------------------------------
# Test 8: Renderer with Reversing Guidelines and Top-Right HUD
# ---------------------------------------------------------------------------

def test_renderer_with_reversing_guidelines_and_hud():
    from visualization.renderer import Renderer
    from perception.detector import DetectionResult
    from perception.freespace import FreeSpaceResult

    cfg = {
        "visualization": {
            "window_name": "Test Window",
            "show_freespace": True,
            "show_grid": True,
            "show_paths": True,
            "show_metrics": True,
            "tts_enabled": False,
        }
    }
    renderer = Renderer(cfg, frame_w=320, frame_h=240)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cand = PathCandidate(
        direction="STRAIGHT",
        angle_deg=0.0,
        points=[(11, 10), (10, 10), (9, 10)],
        clearance=0.75,
        width=2.0,
        progress=0.8,
        curvature=0.0,
        uncertainty=0.1,
        risk=0.1,
    )
    cand.score = 0.75
    cand.total_score = 0.75

    decision = NavigationState(
        command="STRAIGHT",
        decision="GO",
        selected_path_direction="STRAIGHT",
        score=0.75,
        clearance=0.75,
        uncertainty=0.1,
        reason="Clear path",
    )

    det_res = DetectionResult(
        frame_id=1,
        timestamp=time.time(),
        inference_ms=5.0,
        num_detections=0,
        objects=[],
    )
    fs_res = FreeSpaceResult(
        label_map=np.zeros((240, 320), dtype=np.uint8),
        free_prob_map=np.ones((240, 320), dtype=np.float32) * 0.8,
        uncertainty=0.1,
        statistics={"free_fraction": 0.8, "obstacle_fraction": 0.1, "unknown_fraction": 0.1},
    )
    wall_prox = {"is_frontal_collision": False, "lateral_warning": "LEFT", "min_frontal_free": 0.7}

    composite = renderer.render(
        frame=frame,
        detection_result=det_res,
        tracks=[],
        freespace_result=fs_res,
        depth_result=None,
        spatial_map=None,
        candidates=[cand],
        decision=decision,
        fps=32.5,
        frame_id=1,
        latency_ms=30.0,
        wall_proximity=wall_prox,
    )

    assert composite is not None
    # 4-panel composite: 2x height and 2x width plus bottom disclaimer bar
    assert composite.shape[0] >= 240 * 2
    assert composite.shape[1] == 320 * 2

