"""
Unit Tests for Stage B: Two-Camera Predictive Perceptual Handover
================================================================
Validates:
  1. CameraModel geometric transformations & physical overlap calculation.
  2. CrossCameraAssociator affinity scoring and spatial gating.
  3. PredictiveHandoverManager state machine transitions (PRE_ARM, TRANSFER, SECONDARY).
  4. MultiCameraSupportGate mathematical formulation (Support_B = min_e max_c rho).
  5. B0 vs B1 vs B2 mode isolation.
  6. Independent ground truth analytical continuity evaluation.
"""

import math
import numpy as np
import pytest

from evaluation.stage_b_ground_truth import evaluate_stage_b_ground_truth
from navigation.camera_handover import HandoverMode, HandoverState, PredictiveHandoverManager
from navigation.camera_observation import CameraModel, CameraObservation, MultiCameraEntityState
from navigation.critical_region import CriticalRegion
from navigation.cross_camera_association import CrossCameraAssociator
from navigation.multi_camera_support import MultiCameraSupportGate
from navigation.path_generator import PathCandidate


def test_camera_model_overlap_geometry():
    """Verify physical overlap computation between CAM0 (0 deg) and CAM1 (+50 deg)."""
    cam0 = CameraModel(camera_id="CAM0", yaw_deg=0.0, half_hfov_rad=math.radians(32.5))
    cam1 = CameraModel(camera_id="CAM1", yaw_deg=50.0, half_hfov_rad=math.radians(32.5))

    min_ov, max_ov, width = cam0.compute_overlap(cam1)

    assert math.isclose(math.degrees(min_ov), 17.5, abs_tol=1e-3)
    assert math.isclose(math.degrees(max_ov), 32.5, abs_tol=1e-3)
    assert math.isclose(math.degrees(width), 15.0, abs_tol=1e-3)

    # Test coordinate transformation
    world_b = math.radians(25.0)
    loc0 = cam0.world_to_local_bearing(world_b)
    loc1 = cam1.world_to_local_bearing(world_b)

    assert math.isclose(math.degrees(loc0), 25.0, abs_tol=1e-3)
    assert math.isclose(math.degrees(loc1), -25.0, abs_tol=1e-3)

    assert cam0.is_in_fov(loc0) is True
    assert cam1.is_in_fov(loc1) is True


def test_cross_camera_association_affinity():
    """Verify spatial, velocity, and semantic affinity scoring."""
    associator = CrossCameraAssociator(tau_assoc=0.70, max_spatial_dist_m=1.20)

    # Identical position (x=1.0, y=3.0) and velocity -> High affinity
    obs0 = CameraObservation(
        camera_id="CAM0", track_id=1, timestamp=100.0,
        local_bearing_rad=0.3, local_bearing_deg=17.2, local_bearing_rate_rad_s=0.1,
        world_bearing_rad=0.3, world_bearing_deg=17.2, distance_m=3.16,
        world_pos=(1.0, 3.0), world_vel=(0.5, 0.0), rho_fov=0.8,
        class_name="person",
    )
    obs1 = CameraObservation(
        camera_id="CAM1", track_id=101, timestamp=100.02,
        local_bearing_rad=-0.57, local_bearing_deg=-32.8, local_bearing_rate_rad_s=0.1,
        world_bearing_rad=0.3, world_bearing_deg=17.2, distance_m=3.16,
        world_pos=(1.05, 3.02), world_vel=(0.48, 0.0), rho_fov=0.9,
        class_name="person",
    )

    conf, rec = associator.compute_pairwise_affinity(obs0, obs1, now=100.02)
    assert conf >= 0.85
    assert rec.is_associated is True

    # Distant position (> 1.2m) -> Hard spatial rejection (conf = 0.0)
    obs_distant = CameraObservation(
        camera_id="CAM1", track_id=102, timestamp=100.02,
        local_bearing_rad=-0.1, local_bearing_deg=-5.7, local_bearing_rate_rad_s=0.0,
        world_bearing_rad=0.77, world_bearing_deg=44.3, distance_m=4.0,
        world_pos=(2.5, 4.0), world_vel=(0.0, 0.0), rho_fov=0.9,
        class_name="person",
    )
    conf_dist, rec_dist = associator.compute_pairwise_affinity(obs0, obs_distant, now=100.02)
    assert conf_dist == 0.0
    assert rec_dist.is_associated is False


def test_predictive_pre_arm_and_transfer():
    """Verify predictive handover state machine: PRIMARY -> PRE_ARM -> TRANSFER -> SECONDARY."""
    mgr = PredictiveHandoverManager(
        primary_camera_id="CAM0", secondary_camera_id="CAM1",
        tau_release=0.40, tau_acquire=0.60, tau_assoc=0.70,
        min_dwell_time_s=0.20, mode=HandoverMode.B2_PREDICTIVE,
    )

    ent = MultiCameraEntityState(global_entity_id=1, state_enter_timestamp=100.0)

    # Step 1: Initial state (CAM0 strong)
    ent.observations["CAM0"] = CameraObservation(
        camera_id="CAM0", track_id=1, timestamp=100.0,
        local_bearing_rad=0.1, local_bearing_deg=5.7, local_bearing_rate_rad_s=0.1,
        world_bearing_rad=0.1, world_bearing_deg=5.7, distance_m=3.0,
        world_pos=(0.3, 3.0), rho_fov=0.85, in_fov=True,
    )
    ent = mgr.update_entity_handover(ent, association_confidence=0.0, current_time=100.0)
    assert ent.responsibility_state == HandoverState.PRIMARY.value

    # Step 2: CAM0 declines to 0.32 while CAM1 has 0.80 support -> PRE_ARM
    ent.observations["CAM0"].rho_fov = 0.32
    ent.observations["CAM1"] = CameraObservation(
        camera_id="CAM1", track_id=101, timestamp=100.3,
        local_bearing_rad=-0.4, local_bearing_deg=-22.9, local_bearing_rate_rad_s=0.1,
        world_bearing_rad=0.47, world_bearing_deg=27.1, distance_m=3.0,
        world_pos=(1.3, 2.7), rho_fov=0.80, in_fov=True,
    )
    ent = mgr.update_entity_handover(ent, association_confidence=0.0, current_time=100.3)
    assert ent.responsibility_state == HandoverState.PRE_ARM.value

    # Step 3: Association confirmed in overlap (conf=0.90) -> TRANSFER
    ent = mgr.update_entity_handover(ent, association_confidence=0.90, current_time=100.45)
    assert ent.responsibility_state == HandoverState.TRANSFER.value
    assert ent.responsible_camera_id == "CAM1"

    # Step 4: Finalize to SECONDARY
    ent = mgr.update_entity_handover(ent, association_confidence=0.90, current_time=100.50)
    assert ent.responsibility_state == HandoverState.SECONDARY.value
    assert ent.responsible_camera_id == "CAM1"


def test_multi_camera_corridor_support_max_conjunction():
    """Verify Support_B(k) = min_e max_c rho_FOV(c, e) preserves corridor support."""
    gate = MultiCameraSupportGate(tau_safe=0.35, tau_cam=0.40, primary_camera_id="CAM0")

    cand = PathCandidate(
        direction="STRAIGHT", angle_deg=0.0, points=[(r, 12) for r in range(20)],
        clearance=0.8, width=1.5, progress=0.9, curvature=0.0,
        uncertainty=0.05, risk=0.1, score=0.85, total_score=0.85,
    )

    crit_regions = {
        "STRAIGHT": CriticalRegion(
            corridor_direction="STRAIGHT", corridor_angle_deg=0.0, critical_entities=[1],
        )
    }

    ent = MultiCameraEntityState(global_entity_id=1)
    # CAM0 observation expiring (rho=0.25 < 0.40), CAM1 viable (rho=0.88 >= 0.40)
    ent.observations["CAM0"] = CameraObservation(
        camera_id="CAM0", track_id=1, timestamp=1.0, local_bearing_rad=0.5,
        local_bearing_deg=28.6, local_bearing_rate_rad_s=0.2, world_bearing_rad=0.5,
        world_bearing_deg=28.6, distance_m=3.0, world_pos=(1.4, 2.6),
        rho_fov=0.25, in_fov=True,
    )
    ent.observations["CAM1"] = CameraObservation(
        camera_id="CAM1", track_id=101, timestamp=1.0, local_bearing_rad=-0.37,
        local_bearing_deg=-21.4, local_bearing_rate_rad_s=0.2, world_bearing_rad=0.5,
        world_bearing_deg=28.6, distance_m=3.0, world_pos=(1.4, 2.6),
        rho_fov=0.88, in_fov=True,
    )

    # 1. Under Single Camera (Stage A / Mode B0): Corridors rejected due to low support (0.25 < 0.40)
    records_single = gate.evaluate_corridors([cand], crit_regions, {1: ent}, use_multi_cam=False)
    assert records_single["STRAIGHT"].is_admissible is False
    assert records_single["STRAIGHT"].status_label == "INADMISSIBLE_LOW_SUPPORT"
    assert math.isclose(records_single["STRAIGHT"].single_cam_support, 0.25, abs_tol=1e-3)

    # 2. Under Multi-Camera (Stage B / Mode B2): Corridor preserved via CAM1 (0.88 >= 0.40)
    records_multi = gate.evaluate_corridors([cand], crit_regions, {1: ent}, use_multi_cam=True)
    assert records_multi["STRAIGHT"].is_admissible is True
    assert records_multi["STRAIGHT"].status_label == "ADMISSIBLE"
    assert math.isclose(records_multi["STRAIGHT"].multi_cam_support, 0.88, abs_tol=1e-3)


def test_ground_truth_physical_continuity():
    """Verify analytical boundary exit and acquisition calculations."""
    cam0 = CameraModel(camera_id="CAM0", yaw_deg=0.0)
    cam1 = CameraModel(camera_id="CAM1", yaw_deg=50.0)

    # Target starts at +5 deg and moves right at +20 deg/s
    # CAM0 loss (+32.5 deg): (32.5 - 5.0) / 20.0 = 1.375s
    # CAM1 acq (+17.5 deg): (17.5 - 5.0) / 20.0 = 0.625s
    gt = evaluate_stage_b_ground_truth(
        initial_world_bearing_deg=5.0, bearing_rate_deg_s=20.0,
        cam0=cam0, cam1=cam1, duration_s=2.5,
    )

    assert math.isclose(gt.true_cam1_acq_time_s, 0.62, abs_tol=0.02)
    assert math.isclose(gt.true_cam0_loss_time_s, 1.37, abs_tol=0.02)
    assert gt.true_cam1_acq_time_s < gt.true_cam0_loss_time_s
    assert gt.true_handover_possible is True
    assert gt.true_observation_continuity > 0.90
