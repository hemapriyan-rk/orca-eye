"""
ORCA EYE — Stage B: 8 Controlled Multi-Camera Evaluation Scenarios
==================================================================
Defines controlled synthetic spatial benchmarks for two-camera predictive handover:
  S1: Center -> Right boundary crossing (CAM0 -> Overlap -> CAM1)
  S2: Center -> Left boundary crossing (CAM0 -> Blind Left Flank)
  S3: Fast peripheral crossing (High angular velocity > 25 deg/s)
  S4: Slow peripheral crossing (Low angular velocity ~ 6 deg/s)
  S5: Overlap-zone traversal (Prolonged dwell in [17.5 deg, 32.5 deg])
  S6: Partial occlusion during handover (Detection dropout in seam)
  S7: Visually similar entities (Close proximity association ambiguity)
  S8: Multiple simultaneous entities (CAM0, CAM1, and overlap simultaneously)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from navigation.camera_observation import CameraModel, CameraObservation, MultiCameraEntityState
from navigation.rho_fov import RhoFOVPredictor


class StageBScenarioType(str, Enum):
    S1_CENTER_TO_RIGHT = "S1_CENTER_TO_RIGHT"
    S2_CENTER_TO_LEFT = "S2_CENTER_TO_LEFT"
    S3_FAST_PERIPHERAL = "S3_FAST_PERIPHERAL"
    S4_SLOW_PERIPHERAL = "S4_SLOW_PERIPHERAL"
    S5_OVERLAP_TRAVERSAL = "S5_OVERLAP_TRAVERSAL"
    S6_OCCLUSION_HANDOVER = "S6_OCCLUSION_HANDOVER"
    S7_SIMILAR_ENTITIES = "S7_SIMILAR_ENTITIES"
    S8_MULTIPLE_ENTITIES = "S8_MULTIPLE_ENTITIES"


@dataclass
class SyntheticEntityTrajectory:
    """Parametric ground-plane motion model for a synthetic entity."""
    entity_id: int
    initial_world_bearing_deg: float
    bearing_rate_deg_s: float
    initial_distance_m: float
    radial_speed_mps: float = 0.0
    class_name: str = "person"
    occluded_in_overlap: bool = False
    noise_sigma_bearing_deg: float = 0.5
    noise_sigma_rate_deg_s: float = 1.0


def get_scenario_trajectories(
    scen_type: StageBScenarioType,
    rng: Optional[np.random.Generator] = None,
) -> List[SyntheticEntityTrajectory]:
    """
    Returns initial parametric trajectories for the specified scenario with stochastic noise.
    """
    r = np.random.default_rng() if rng is None else rng

    if scen_type == StageBScenarioType.S1_CENTER_TO_RIGHT:
        # Entity starts in CAM0 center (+5 deg) and moves right at +16 deg/s into CAM1
        b_noise = float(r.normal(0.0, 1.0))
        rate_noise = float(r.normal(0.0, 1.5))
        return [
            SyntheticEntityTrajectory(
                entity_id=1,
                initial_world_bearing_deg=5.0 + b_noise,
                bearing_rate_deg_s=16.0 + rate_noise,
                initial_distance_m=3.0,
            )
        ]

    elif scen_type == StageBScenarioType.S2_CENTER_TO_LEFT:
        # Entity starts in CAM0 center (-5 deg) and moves left at -16 deg/s into blind flank
        b_noise = float(r.normal(0.0, 1.0))
        rate_noise = float(r.normal(0.0, 1.5))
        return [
            SyntheticEntityTrajectory(
                entity_id=2,
                initial_world_bearing_deg=-5.0 + b_noise,
                bearing_rate_deg_s=-16.0 + rate_noise,
                initial_distance_m=3.0,
            )
        ]

    elif scen_type == StageBScenarioType.S3_FAST_PERIPHERAL:
        # Entity starts near boundary (+12 deg) crossing at fast +28 deg/s
        b_noise = float(r.normal(0.0, 1.0))
        rate_noise = float(r.normal(0.0, 2.0))
        return [
            SyntheticEntityTrajectory(
                entity_id=3,
                initial_world_bearing_deg=12.0 + b_noise,
                bearing_rate_deg_s=18.0 + rate_noise,
                initial_distance_m=2.5,
            )
        ]

    elif scen_type == StageBScenarioType.S4_SLOW_PERIPHERAL:
        # Entity starts at +10 deg drifting slowly at +5 deg/s
        b_noise = float(r.normal(0.0, 0.8))
        rate_noise = float(r.normal(0.0, 0.8))
        return [
            SyntheticEntityTrajectory(
                entity_id=4,
                initial_world_bearing_deg=10.0 + b_noise,
                bearing_rate_deg_s=5.0 + rate_noise,
                initial_distance_m=3.5,
            )
        ]

    elif scen_type == StageBScenarioType.S5_OVERLAP_TRAVERSAL:
        # Entity enters and loiters inside the 15 deg overlap zone (+22 deg to +28 deg)
        b_noise = float(r.normal(0.0, 1.0))
        return [
            SyntheticEntityTrajectory(
                entity_id=5,
                initial_world_bearing_deg=22.0 + b_noise,
                bearing_rate_deg_s=2.5,
                initial_distance_m=3.0,
            )
        ]

    elif scen_type == StageBScenarioType.S6_OCCLUSION_HANDOVER:
        # Entity crosses right, but suffers partial occlusion in overlap
        return [
            SyntheticEntityTrajectory(
                entity_id=6,
                initial_world_bearing_deg=10.0,
                bearing_rate_deg_s=15.0,
                initial_distance_m=3.0,
                occluded_in_overlap=True,
            )
        ]

    elif scen_type == StageBScenarioType.S7_SIMILAR_ENTITIES:
        # Two entities in close proximity in overlap zone (dist ~0.5m)
        return [
            SyntheticEntityTrajectory(
                entity_id=71,
                initial_world_bearing_deg=22.0,
                bearing_rate_deg_s=12.0,
                initial_distance_m=3.0,
                class_name="person",
            ),
            SyntheticEntityTrajectory(
                entity_id=72,
                initial_world_bearing_deg=26.0,
                bearing_rate_deg_s=11.5,
                initial_distance_m=3.3,
                class_name="person",
            ),
        ]

    elif scen_type == StageBScenarioType.S8_MULTIPLE_ENTITIES:
        # 3 entities in distinct zones: E1 in CAM0 (-15 deg), E2 in CAM1 (+60 deg), E3 in overlap (+25 deg)
        return [
            SyntheticEntityTrajectory(
                entity_id=81,
                initial_world_bearing_deg=-15.0,
                bearing_rate_deg_s=2.0,
                initial_distance_m=3.5,
            ),
            SyntheticEntityTrajectory(
                entity_id=82,
                initial_world_bearing_deg=60.0,
                bearing_rate_deg_s=-3.0,
                initial_distance_m=4.0,
            ),
            SyntheticEntityTrajectory(
                entity_id=83,
                initial_world_bearing_deg=22.0,
                bearing_rate_deg_s=14.0,
                initial_distance_m=2.8,
            ),
        ]

    return []


def generate_camera_observations_at_time(
    trajectories: List[SyntheticEntityTrajectory],
    elapsed_time_s: float,
    cam0: CameraModel,
    cam1: CameraModel,
    rho_pred: RhoFOVPredictor,
    now_ts: float,
) -> Tuple[List[CameraObservation], List[CameraObservation]]:
    """
    Simulates observations seen by CAM0 and CAM1 at a given time step.
    """
    obs_cam0: List[CameraObservation] = []
    obs_cam1: List[CameraObservation] = []

    for traj in trajectories:
        # Compute ground world bearing at this instant
        w_bearing_deg = traj.initial_world_bearing_deg + traj.bearing_rate_deg_s * elapsed_time_s
        w_bearing_rad = math.radians(w_bearing_deg)
        dist_m = traj.initial_distance_m + traj.radial_speed_mps * elapsed_time_s
        dist_m = max(0.5, dist_m)

        gx = dist_m * math.sin(w_bearing_rad)
        gy = dist_m * math.cos(w_bearing_rad)
        w_pos = (gx, gy)

        rate_rad_s = math.radians(traj.bearing_rate_deg_s)
        w_vel = (
            traj.radial_speed_mps * math.sin(w_bearing_rad) + dist_m * rate_rad_s * math.cos(w_bearing_rad),
            traj.radial_speed_mps * math.cos(w_bearing_rad) - dist_m * rate_rad_s * math.sin(w_bearing_rad),
        )

        # 1. Camera 0
        loc_b0_rad = cam0.world_to_local_bearing(w_bearing_rad)
        in_fov_0 = cam0.is_in_fov(loc_b0_rad)

        if in_fov_0:
            pred0 = rho_pred.predict_survival(
                entity_id=traj.entity_id,
                bearing_rad=loc_b0_rad,
                bearing_rate=rate_rad_s,
                camera_id=cam0.camera_id,
                current_time=now_ts,
            )
            obs0 = CameraObservation(
                camera_id=cam0.camera_id,
                track_id=traj.entity_id,
                timestamp=now_ts,
                local_bearing_rad=loc_b0_rad,
                local_bearing_deg=math.degrees(loc_b0_rad),
                local_bearing_rate_rad_s=rate_rad_s,
                world_bearing_rad=w_bearing_rad,
                world_bearing_deg=w_bearing_deg,
                distance_m=dist_m,
                world_pos=w_pos,
                world_vel=w_vel,
                rho_fov=pred0.rho_fov,
                detection_confidence=0.95,
                class_name=traj.class_name,
                in_fov=True,
            )
            obs_cam0.append(obs0)

        # 2. Camera 1
        loc_b1_rad = cam1.world_to_local_bearing(w_bearing_rad)
        in_fov_1 = cam1.is_in_fov(loc_b1_rad)

        # Check occlusion in overlap for S6
        is_occluded = traj.occluded_in_overlap and (17.5 <= w_bearing_deg <= 32.5)

        if in_fov_1 and not is_occluded:
            pred1 = rho_pred.predict_survival(
                entity_id=traj.entity_id,
                bearing_rad=loc_b1_rad,
                bearing_rate=rate_rad_s,
                camera_id=cam1.camera_id,
                current_time=now_ts,
            )
            obs1 = CameraObservation(
                camera_id=cam1.camera_id,
                track_id=traj.entity_id + 100,  # distinct local sensor track ID
                timestamp=now_ts,
                local_bearing_rad=loc_b1_rad,
                local_bearing_deg=math.degrees(loc_b1_rad),
                local_bearing_rate_rad_s=rate_rad_s,
                world_bearing_rad=w_bearing_rad,
                world_bearing_deg=w_bearing_deg,
                distance_m=dist_m,
                world_pos=w_pos,
                world_vel=w_vel,
                rho_fov=pred1.rho_fov,
                detection_confidence=0.92,
                class_name=traj.class_name,
                in_fov=True,
            )
            obs_cam1.append(obs1)

    return obs_cam0, obs_cam1
