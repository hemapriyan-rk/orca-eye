"""
ORCA EYE — Stage B: Independent Ground-Truth Evaluator
======================================================
Calculates exact physical ground-truth metrics for two-camera trajectories:
  - True CAM0 boundary loss time
  - True CAM1 optical acquisition time
  - True overlap entry and exit times
  - True multi-camera physical observation continuity
Completely decoupled from rho_FOV predictions and tracker state.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from navigation.camera_observation import CameraModel


@dataclass
class StageBGroundTruthResult:
    """True deterministic physical outcome for a two-camera trajectory."""
    entity_id: int
    true_cam0_loss_time_s: Optional[float]
    true_cam1_acq_time_s: Optional[float]
    true_overlap_entry_time_s: Optional[float]
    true_overlap_exit_time_s: Optional[float]
    true_observation_continuity: float        # Fraction of time observed by at least one camera in [0, duration]
    true_gap_duration_s: float                # Duration where entity was in blind zone between cameras
    true_handover_possible: bool


def evaluate_stage_b_ground_truth(
    initial_world_bearing_deg: float,
    bearing_rate_deg_s: float,
    cam0: CameraModel,
    cam1: CameraModel,
    duration_s: float = 3.0,
    dt_eval_s: float = 0.01,
) -> StageBGroundTruthResult:
    """
    Computes exact physical ground-truth optical boundary crossings.
    """
    n_steps = int(duration_s / dt_eval_s)

    cam0_loss_t = None
    cam1_acq_t = None
    overlap_entry_t = None
    overlap_exit_t = None

    observed_steps = 0
    blind_gap_steps = 0

    was_in_cam0 = False
    was_in_overlap = False

    for step in range(n_steps):
        t = step * dt_eval_s
        wb_deg = initial_world_bearing_deg + bearing_rate_deg_s * t
        wb_rad = math.radians(wb_deg)

        in_0 = cam0.is_world_bearing_in_fov(wb_rad)
        in_1 = cam1.is_world_bearing_in_fov(wb_rad)

        if in_0 or in_1:
            observed_steps += 1
        else:
            blind_gap_steps += 1

        # Track CAM0 loss
        if step == 0 and in_0:
            was_in_cam0 = True
        if was_in_cam0 and not in_0 and cam0_loss_t is None:
            cam0_loss_t = t

        # Track CAM1 acquisition
        if in_1 and cam1_acq_t is None:
            cam1_acq_t = t

        # Track Overlap
        in_overlap = (in_0 and in_1)
        if in_overlap and overlap_entry_t is None:
            overlap_entry_t = t
            was_in_overlap = True
        if was_in_overlap and not in_overlap and overlap_exit_t is None:
            overlap_exit_t = t

    continuity = observed_steps / n_steps if n_steps > 0 else 0.0
    gap_duration = blind_gap_steps * dt_eval_s

    # Handover is physically possible if CAM1 acquires before or within overlap of CAM0
    can_handover = (cam1_acq_t is not None) and (cam0_loss_t is not None) and (cam1_acq_t <= cam0_loss_t)

    return StageBGroundTruthResult(
        entity_id=1,
        true_cam0_loss_time_s=round(cam0_loss_t, 3) if cam0_loss_t is not None else None,
        true_cam1_acq_time_s=round(cam1_acq_t, 3) if cam1_acq_t is not None else None,
        true_overlap_entry_time_s=round(overlap_entry_t, 3) if overlap_entry_t is not None else None,
        true_overlap_exit_time_s=round(overlap_exit_t, 3) if overlap_exit_t is not None else None,
        true_observation_continuity=round(continuity, 4),
        true_gap_duration_s=round(gap_duration, 4),
        true_handover_possible=can_handover,
    )
