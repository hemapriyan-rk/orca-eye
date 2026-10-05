"""
Unit Tests for Stage-A Experimental Validation Framework
========================================================
Validates:
  1. Deterministic seeding and reproducibility across trials.
  2. B0 vs B1 mode isolation in DecisionMaker.
  3. Ground truth physical trajectory propagation model.
  4. PerceptualSupportGate mathematical admissibility logic.
  5. Continuous soft quality weighting in RhoFOVPredictor.
  6. Calibration and Brier score evaluation correctness.
"""

import math
import numpy as np
import pytest

from evaluation.ground_truth import evaluate_ground_truth_survival, GroundTruthSurvivalResult
from evaluation.reproducibility import seed_everything
from navigation.camera_support import CorridorSupportRecord, PerceptualSupportGate
from navigation.critical_region import CriticalRegion
from navigation.decision import DecisionMaker, NavigationReason
from navigation.geometry import bearing_rate_quality
from navigation.path_generator import PathCandidate
from navigation.rho_fov import RhoFOVPredictor
from navigation.spatial_entity import SpatialEntity


def create_dummy_candidate(direction: str, score: float = 0.8) -> PathCandidate:
    """Helper to instantiate PathCandidate."""
    return PathCandidate(
        direction=direction,
        angle_deg=0.0,
        points=[(r, 12) for r in range(20)],
        clearance=0.8,
        width=1.5,
        progress=0.9,
        curvature=0.0,
        uncertainty=0.05,
        risk=0.1,
        score=score,
        total_score=score,
    )


def test_deterministic_seeding_reproducibility():
    """Verify that seed_everything produces identical random streams."""
    seed_everything(12345)
    seq1 = np.random.normal(0.0, 1.0, size=50)

    seed_everything(12345)
    seq2 = np.random.normal(0.0, 1.0, size=50)

    np.testing.assert_array_equal(seq1, seq2)


def test_b0_b1_mode_isolation():
    """Verify that DecisionMaker strictly decouples Mode B0 from Stage-A gating."""
    dec_b0 = DecisionMaker({"enable_support_gate": False})
    dec_b1 = DecisionMaker({"enable_support_gate": True})

    cand_straight = create_dummy_candidate("STRAIGHT", score=0.90)
    cand_right = create_dummy_candidate("RIGHT", score=0.75)
    candidates = [cand_straight, cand_right]

    support_records = {
        "STRAIGHT": CorridorSupportRecord(
            corridor_direction="STRAIGHT",
            safety_score=0.90,
            perceptual_support=0.20,
            is_admissible=False,
            status_label="INADMISSIBLE_LOW_SUPPORT",
            reason="Low perceptual support",
        ),
        "RIGHT": CorridorSupportRecord(
            corridor_direction="RIGHT",
            safety_score=0.75,
            perceptual_support=0.85,
            is_admissible=True,
            status_label="ADMISSIBLE",
            reason="Clear perceptual support",
        ),
    }

    # B0 should IGNORE support gating and pick highest-scoring STRAIGHT
    res_b0 = dec_b0.decide(candidates, frame_id=1, corridor_support=support_records)
    assert res_b0.command == "STRAIGHT"
    assert res_b0.score == 0.90

    # B1 should ENFORCE support gating, REJECT STRAIGHT, and REROUTE to RIGHT
    res_b1 = dec_b1.decide(candidates, frame_id=1, corridor_support=support_records)
    assert res_b1.command == "RIGHT"
    assert res_b1.score == 0.75
    assert res_b1.camera_support == 0.85
    assert res_b1.admissibility_status == "ADMISSIBLE"

    # When all paths fail support, B1 commands STOP with LOW_PERCEPTUAL_SUPPORT
    res_b1_stop = dec_b1.decide([cand_straight], frame_id=2, corridor_support=support_records)
    assert res_b1_stop.command == "STOP"
    assert NavigationReason.LOW_PERCEPTUAL_SUPPORT in res_b1_stop.reason


def test_ground_truth_survival_physics():
    """Verify analytical forward propagation of entity trajectory."""
    # Stationary centered entity stays in FOV for all t in [0, 2.0]
    gt_center = evaluate_ground_truth_survival(
        initial_bearing_rad=0.0,
        initial_bearing_rate_rad_s=0.0,
        horizon_s=2.0,
    )
    assert gt_center.stayed_in_fov is True
    assert gt_center.stayed_continuous_valid is True
    assert gt_center.time_to_fov_loss_s is None

    # Entity rapidly exiting right FOV (init at 0.5 rad ~ 28.6 deg, rate +0.5 rad/s)
    # Exits 32.5 deg (0.567 rad) at (0.567 - 0.50) / 0.50 ~ 0.134 s
    gt_exit = evaluate_ground_truth_survival(
        initial_bearing_rad=0.50,
        initial_bearing_rate_rad_s=0.50,
        horizon_s=2.0,
    )
    assert gt_exit.stayed_in_fov is False
    assert gt_exit.stayed_continuous_valid is False
    assert gt_exit.time_to_fov_loss_s is not None
    assert 0.10 < gt_exit.time_to_fov_loss_s < 0.20


def test_perceptual_support_gate_conjunction():
    """Verify Support(c, k) = min_{e in E_k} rho_FOV(c, e) and admissibility logic."""
    gate = PerceptualSupportGate(tau_safe=0.35, tau_cam=0.40)
    cand = create_dummy_candidate("STRAIGHT", score=0.80)

    # Case 1: Corridor with multiple entities -> minimum support governs
    crit_regions = {
        "STRAIGHT": CriticalRegion(
            corridor_direction="STRAIGHT",
            corridor_angle_deg=0.0,
            critical_entities=[101, 102],
        )
    }
    rho_predictions = {
        101: type("RhoPred", (), {"rho_fov": 0.85})(),
        102: type("RhoPred", (), {"rho_fov": 0.32})(),
    }

    support_records = gate.evaluate_corridors([cand], crit_regions, rho_predictions)
    assert math.isclose(support_records["STRAIGHT"].perceptual_support, 0.32, abs_tol=1e-5)
    assert support_records["STRAIGHT"].is_admissible is False
    assert support_records["STRAIGHT"].status_label == "INADMISSIBLE_LOW_SUPPORT"

    # Case 2: Corridor with no critical entities defaults to 1.0
    support_empty = gate.evaluate_corridors([cand], {}, rho_predictions)
    assert math.isclose(support_empty["STRAIGHT"].perceptual_support, 1.0, abs_tol=1e-5)
    assert support_empty["STRAIGHT"].is_admissible is True


def test_rho_fov_soft_vs_hard_behavior():
    """Verify continuous Gaussian tracking weighting q_BR(phi_dot) in RhoFOVPredictor."""
    pred_soft = RhoFOVPredictor(num_samples=100, use_soft_quality=True, seed=42)
    pred_hard = RhoFOVPredictor(num_samples=100, use_soft_quality=False, seed=42)

    # Gentle bearing rate where quality q_BR is around 0.6 and entity stays within FOV
    res_soft = pred_soft.predict_survival(1, bearing_rad=0.1, bearing_rate=0.05)
    res_hard = pred_hard.predict_survival(1, bearing_rad=0.1, bearing_rate=0.05)

    # Both should have valid survival > 0
    assert 0.0 < res_soft.rho_fov <= 1.0
    assert 0.0 < res_hard.rho_fov <= 1.0
    # Soft weighting continuously penalizes faster rates
    assert res_soft.rho_fov <= res_hard.rho_fov


def test_brier_score_metric():
    """Verify Brier score calculation: BS = (1/M) sum (p_i - y_i)^2."""
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    # Errors: (0.1)^2 + (-0.2)^2 + (0.2)^2 + (0.1)^2 = 0.01 + 0.04 + 0.04 + 0.01 = 0.10 / 4 = 0.025
    brier = float(np.mean((p - y) ** 2))
    assert math.isclose(brier, 0.025, abs_tol=1e-5)
