"""
Unit and Integration Tests for ORCA EYE — Stage A
=================================================
Tests:
  1. CameraCalibration & egocentric projection geometry
  2. Egocentric geometry, bearing, bearing-rate, and tracking quality
  3. ThreatModel, analytic TTC, and ThreatRecord
  4. CriticalRegionExtractor and Critical Entity Set E_k
  5. RhoFOVPredictor Monte Carlo continuous survival probability
  6. PerceptualSupportGate and Corridor Admissibility Gate
  7. CameraResponsibilityManager state machine
  8. Controlled Navigation Scenario Suite
"""

import math
import pytest
import numpy as np

from perception.calibration import CameraCalibration
from navigation.geometry import (
    calculate_bearing,
    calculate_bearing_deg,
    is_in_fov,
    calculate_bearing_rate,
    bearing_rate_quality,
    optical_to_wearer_ego,
    wearer_ego_to_optical,
    EntityKinematics,
)
from navigation.threat_model import (
    ThreatModel,
    ThreatLevel,
    ThreatRecord,
    compute_analytic_ttc,
)
from navigation.critical_region import CriticalRegionExtractor, CriticalRegion
from navigation.rho_fov import RhoFOVPredictor, RhoFOVPrediction
from navigation.camera_support import PerceptualSupportGate, CorridorSupportRecord
from navigation.camera_responsibility import CameraResponsibilityManager, CameraRole
from navigation.path_generator import PathCandidate
from navigation.decision import DecisionMaker, NavigationReason
from evaluation.scenarios import (
    ControlledScenarioType,
    create_synthetic_spatial_state,
    CONTROLLED_SUITE,
)


# ---------------------------------------------------------------------------
# 1. Camera Calibration
# ---------------------------------------------------------------------------

def test_camera_calibration_fov_construction():
    calib = CameraCalibration.from_fov(frame_width=640, frame_height=480, hfov_deg=65.0)
    assert calib.frame_width == 640
    assert calib.frame_height == 480
    assert round(calib.hfov_deg, 1) == 65.0
    assert calib.cx == 320.0
    assert calib.cy == 240.0
    assert calib.fx > 400.0


def test_camera_calibration_projections():
    calib = CameraCalibration.from_fov(frame_width=640, frame_height=480, hfov_deg=65.0)
    # Center pixel (cx, cy) at 2.0m depth
    xc, yc, zc = calib.pixel_and_depth_to_camera_point(calib.cx, calib.cy, depth_m=2.0)
    assert abs(xc) < 1e-4
    assert abs(yc) < 1e-4
    assert abs(zc - 2.0) < 1e-4

    # Wearer ego transform
    x, y, z = calib.camera_to_wearer_ego(xc, yc, zc)
    assert abs(x) < 1e-4          # lateral center
    assert abs(y - 2.0) < 1e-4    # 2 meters forward along ground plane
    assert abs(z) < 1e-4          # on optical ground height


# ---------------------------------------------------------------------------
# 2. Egocentric Geometry & Kinematics
# ---------------------------------------------------------------------------

def test_bearing_calculations():
    # Directly ahead (x=0, y=2)
    b_straight = calculate_bearing(0.0, 2.0)
    assert abs(b_straight) < 1e-5
    assert abs(calculate_bearing_deg(0.0, 2.0)) < 1e-4

    # Right 45 degrees (x=2, y=2)
    b_right = calculate_bearing(2.0, 2.0)
    assert abs(math.degrees(b_right) - 45.0) < 1e-4

    # Left 45 degrees (x=-2, y=2)
    b_left = calculate_bearing(-2.0, 2.0)
    assert abs(math.degrees(b_left) - (-45.0)) < 1e-4


def test_fov_bounds():
    half_hfov = math.radians(32.5)  # 65 deg full HFOV
    assert is_in_fov(0.0, half_hfov) is True
    assert is_in_fov(math.radians(20.0), half_hfov) is True
    assert is_in_fov(math.radians(-20.0), half_hfov) is True
    assert is_in_fov(math.radians(35.0), half_hfov) is False
    assert is_in_fov(math.radians(-40.0), half_hfov) is False


def test_bearing_rate_and_quality():
    # Rate of 0.2 rad in 0.1 s = 2.0 rad/s
    rate = calculate_bearing_rate(0.3, 0.1, 0.1)
    assert abs(rate - 2.0) < 1e-4

    # Stationary entity -> high quality
    q_stationary = bearing_rate_quality(0.0, sigma_br=0.6)
    assert abs(q_stationary - 1.0) < 1e-4

    # Fast crossing entity -> low quality
    q_fast = bearing_rate_quality(2.5, sigma_br=0.6)
    assert q_fast < 0.10


def test_entity_kinematics():
    half_hfov = math.radians(32.5)
    kin = EntityKinematics(track_id=1, timestamp=0.0, x=0.0, y=3.0)
    assert kin.bearing_rad == 0.0
    assert kin.in_fov is True

    # Move right after 0.1s to x=0.5, y=3.0
    kin.update(new_timestamp=0.1, new_x=0.5, new_y=3.0, half_hfov_rad=half_hfov)
    assert kin.vx > 0.0
    assert kin.bearing_rad > 0.0
    assert kin.bearing_rate > 0.0


# ---------------------------------------------------------------------------
# 3. Explicit Threat & Maneuver Conflict Model
# ---------------------------------------------------------------------------

def test_analytic_ttc_head_on_collision():
    # Object at y=4.0m, moving toward user at 2.0 m/s relative speed
    rel_pos = (0.0, 4.0)
    rel_vel = (0.0, -2.0)
    d_safe = 1.0

    ttc, dca = compute_analytic_ttc(rel_pos, rel_vel, d_safe=d_safe)
    # Reaches d_safe=1.0m when distance closes from 4.0m to 1.0m (3.0m / 2.0m/s = 1.5s)
    assert ttc is not None
    assert abs(ttc - 1.5) < 0.05
    assert abs(dca - 0.0) < 0.10


def test_analytic_ttc_diverging_motion():
    # Object moving away from user
    rel_pos = (0.0, 2.0)
    rel_vel = (0.0, 1.5)  # moving farther away
    ttc, dca = compute_analytic_ttc(rel_pos, rel_vel, d_safe=1.0)
    assert ttc is None
    assert dca >= 2.0


def test_threat_model_classification():
    threat_model = ThreatModel(d_safe=0.9, horizon_s=3.0, user_walk_speed_m_s=1.0)

    class MockEntity:
        track_id = 42
        relative_distance = 0.3  # approx 1.8m
        relative_bearing = 0.0
        velocity = [0.0, 0.0]
        tracking_confidence = 0.9

    rec = threat_model.evaluate_entity_corridor(
        entity=MockEntity(),
        corridor_direction="STRAIGHT",
        corridor_angle_deg=0.0,
    )
    assert rec.entity_id == 42
    assert rec.corridor_direction == "STRAIGHT"
    # Stationary obstacle in walking corridor should register as a conflict / threat
    assert rec.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.MEDIUM)


# ---------------------------------------------------------------------------
# 4. Critical Region & Critical Entity Set E_k
# ---------------------------------------------------------------------------

def test_critical_region_extraction():
    extractor = CriticalRegionExtractor(corridor_width_m=1.0, horizon_s=2.0)

    cand_straight = PathCandidate(direction="STRAIGHT", angle_deg=0.0, clearance=0.8)
    cand_right = PathCandidate(direction="RIGHT", angle_deg=30.0, clearance=0.8)

    class EntityCenter:
        track_id = 1
        relative_distance = 0.3
        relative_bearing = 0.0
        velocity = [0.0, 0.0]
        tracking_confidence = 0.95

    class EntityFarRight:
        track_id = 2
        relative_distance = 0.3
        relative_bearing = 30.0  # along right corridor (30 deg), well clear of straight
        velocity = [0.0, 0.0]
        tracking_confidence = 0.95

    crit_regions = extractor.extract_critical_regions(
        candidates=[cand_straight, cand_right],
        spatial_entities=[EntityCenter(), EntityFarRight()],
    )

    # Center entity should be in STRAIGHT's E_k
    assert 1 in crit_regions["STRAIGHT"].critical_entities
    # Far right entity should be in RIGHT's E_k
    assert 2 in crit_regions["RIGHT"].critical_entities
    # Far right entity should NOT be in STRAIGHT's E_k
    assert 2 not in crit_regions["STRAIGHT"].critical_entities


# ---------------------------------------------------------------------------
# 5. Monte Carlo rho_FOV Continuous Survival Prediction
# ---------------------------------------------------------------------------

def test_rho_fov_centered_stationary_entity():
    predictor = RhoFOVPredictor(num_samples=60, horizon_s=2.0)
    # Centered stationary entity: bearing 0.0, rate 0.0
    pred = predictor.predict_survival(
        entity_id=1,
        bearing_rad=0.0,
        bearing_rate=0.0,
        sigma_phi=0.02,
        sigma_phi_dot=0.05,
    )
    assert pred.initial_validity is True
    # Should maintain very high survival probability
    assert pred.rho_fov >= 0.95


def test_rho_fov_exiting_entity():
    predictor = RhoFOVPredictor(num_samples=60, horizon_s=2.0)
    # Entity right near edge of FOV moving rapidly outward
    edge_bearing = math.radians(30.0)  # half FOV is 32.5 deg
    rapid_rate = math.radians(15.0)    # 15 deg/sec outward
    pred = predictor.predict_survival(
        entity_id=2,
        bearing_rad=edge_bearing,
        bearing_rate=rapid_rate,
        sigma_phi=0.02,
        sigma_phi_dot=0.05,
    )
    # Rapidly exits FOV -> low continuous survival probability
    assert pred.rho_fov < 0.30


def test_rho_fov_calibration_evaluation():
    predictor = RhoFOVPredictor(horizon_s=1.0)
    pred = predictor.predict_survival(
        entity_id=5,
        bearing_rad=0.0,
        bearing_rate=0.0,
        current_time=100.0,
    )
    assert pred.ground_truth_valid is None

    # Evaluate at t = 101.5 (horizon elapsed)
    # Entity is still observed at (0.0, 0.0)
    evaluated = predictor.evaluate_pending_predictions(
        current_time=101.5,
        current_observations={5: (0.0, 0.0)},
    )
    assert len(evaluated) == 1
    assert evaluated[0].ground_truth_valid is True
    assert evaluated[0].brier_error is not None
    assert evaluated[0].brier_error < 0.05


# ---------------------------------------------------------------------------
# 6. Perceptual Support Gate & Corridor Admissibility
# ---------------------------------------------------------------------------

def test_perceptual_support_gate():
    gate = PerceptualSupportGate(tau_safe=0.35, tau_cam=0.40)

    cand_straight = PathCandidate(direction="STRAIGHT", angle_deg=0.0, clearance=0.7, score=0.7)
    cand_right = PathCandidate(direction="RIGHT", angle_deg=30.0, clearance=0.7, score=0.7)

    crit_regions = {
        "STRAIGHT": CriticalRegion(
            corridor_direction="STRAIGHT",
            corridor_angle_deg=0.0,
            critical_entities=[10],
        ),
        "RIGHT": CriticalRegion(
            corridor_direction="RIGHT",
            corridor_angle_deg=30.0,
            critical_entities=[20],
        ),
    }

    # Entity 10 has high survival (0.95), entity 20 has expired survival (0.15)
    rho_preds = {
        10: RhoFOVPrediction("PRIMARY", 10, 0.0, 2.0, rho_fov=0.95, initial_bearing_rad=0.0, initial_bearing_rate=0.0, initial_validity=True),
        20: RhoFOVPrediction("PRIMARY", 20, 0.0, 2.0, rho_fov=0.15, initial_bearing_rad=0.0, initial_bearing_rate=0.0, initial_validity=True),
    }

    support_records = gate.evaluate_corridors(
        candidates=[cand_straight, cand_right],
        critical_regions=crit_regions,
        rho_predictions=rho_preds,
    )

    # STRAIGHT should be ADMISSIBLE
    assert support_records["STRAIGHT"].is_admissible is True
    assert support_records["STRAIGHT"].status_label == "ADMISSIBLE"

    # RIGHT should be INADMISSIBLE_LOW_SUPPORT
    assert support_records["RIGHT"].is_admissible is False
    assert support_records["RIGHT"].status_label == "INADMISSIBLE_LOW_SUPPORT"


# ---------------------------------------------------------------------------
# 7. Camera Responsibility Manager
# ---------------------------------------------------------------------------

def test_camera_responsibility_manager():
    mgr = CameraResponsibilityManager(primary_camera_id="PRIMARY_CAM")
    primary = mgr.get_primary_assignment()
    assert primary.role == CameraRole.PRIMARY
    assert primary.is_physical is True
    assert primary.active is True

    mgr.update_responsibilities(
        selected_corridor="SLIGHT_LEFT",
        critical_entities=[1, 3],
        support_score=0.88,
    )
    assert primary.assigned_corridor == "SLIGHT_LEFT"
    assert primary.target_entities == [1, 3]
    assert primary.support_score == 0.88


# ---------------------------------------------------------------------------
# 8. Controlled Navigation Scenario Suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_type", [
    ControlledScenarioType.CLEAR_PATH,
    ControlledScenarioType.CENTRAL_STATIC_OBSTACLE,
    ControlledScenarioType.LEFT_BLOCKED_RIGHT_OPEN,
    ControlledScenarioType.RIGHT_BLOCKED_LEFT_OPEN,
    ControlledScenarioType.BOTH_SIDES_BLOCKED,
    ControlledScenarioType.CROSSING_PEDESTRIAN,
    ControlledScenarioType.BLIND_UNCERTAINTY,
])
def test_controlled_scenarios(scenario_type):
    spec = CONTROLLED_SUITE[scenario_type]
    smap, entities = create_synthetic_spatial_state(scenario_type)

    from navigation.path_generator import PathGenerator
    from navigation.path_scorer import PathScorer
    from navigation.dynamic_conflict import DynamicConflictEngine

    path_gen = PathGenerator({}, smap.rows, smap.cols)
    dyn_engine = DynamicConflictEngine({})
    scorer = PathScorer({})
    dm = DecisionMaker({})

    candidates = path_gen.generate(smap)
    conflicts_dict = dyn_engine.evaluate_conflicts(
        spatial_entities=entities,
        candidates=candidates,
        frame_w=640,
        frame_h=480,
        grid_rows=smap.rows,
        grid_cols=smap.cols,
    )
    active_conflicts = [c for clist in conflicts_dict.values() for c in clist]
    candidates = scorer.score(candidates)

    decision = dm.decide(
        candidates=candidates,
        frame_id=1,
        spatial_entities=entities,
        dynamic_conflicts=active_conflicts,
    )

    assert decision.command in spec.admissible_commands, (
        f"Scenario {scenario_type.value}: expected one of {spec.admissible_commands}, "
        f"got {decision.command} (reason: {decision.reason})"
    )
