"""
ORCA EYE — Stage A: Ground-Truth Observation Survival Calculator
================================================================
Independent ground-truth evaluator for entity camera observation survival.
Calculates actual physical outcome from true reference trajectories, completely
independent of Monte Carlo rho_FOV predictions.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from navigation.geometry import bearing_rate_quality, is_in_fov


@dataclass
class GroundTruthSurvivalResult:
    """Independent ground truth observation outcome for an entity over horizon H."""
    entity_id: int
    stayed_in_fov: bool                 # True if |phi(t)| <= theta_c for all t in [0, H]
    stayed_continuous_valid: bool       # True if in FOV AND q_BR >= q_min for all t in [0, H]
    min_observed_quality: float         # min q_BR along reference path
    time_to_fov_loss_s: Optional[float] # Time (s) at which FOV was first exited, or None if survived
    final_bearing_rad: float
    max_bearing_rate_rad_s: float
    trajectory_points: List[Tuple[float, float]]  # [(tau, phi_tau)]


def evaluate_ground_truth_survival(
    initial_bearing_rad: float,
    initial_bearing_rate_rad_s: float,
    half_hfov_rad: float = math.radians(32.5),
    horizon_s: float = 2.0,
    dt_eval_s: float = 0.05,            # High temporal resolution reference evaluation
    sigma_br: float = 0.60,
    q_min: float = 0.20,
    angular_acceleration_rad_s2: float = 0.0,
) -> GroundTruthSurvivalResult:
    """
    Calculates the true deterministic physical outcome for an entity reference trajectory.

    Parameters
    ----------
    initial_bearing_rad : True initial ground bearing phi(0)
    initial_bearing_rate_rad_s : True initial angular rate phi_dot(0)
    half_hfov_rad : Camera horizontal half-HFOV
    horizon_s : Evaluation horizon (seconds)
    dt_eval_s : Fine temporal step for ground-truth simulation
    sigma_br : Quality model Gaussian spread
    q_min : Minimum acceptable tracking quality
    angular_acceleration_rad_s2 : True trajectory curvature / acceleration

    Returns
    -------
    GroundTruthSurvivalResult
    """
    steps = int(math.ceil(horizon_s / dt_eval_s))
    time_points = [s * dt_eval_s for s in range(1, steps + 1)]

    stayed_in_fov = True
    stayed_valid = True
    time_to_loss = None
    min_q = 1.0
    max_rate = abs(initial_bearing_rate_rad_s)
    traj = []

    curr_phi = initial_bearing_rad
    curr_rate = initial_bearing_rate_rad_s

    for tau in time_points:
        curr_rate += angular_acceleration_rad_s2 * dt_eval_s
        curr_phi += curr_rate * dt_eval_s
        traj.append((tau, curr_phi))

        # Check FOV boundary
        in_fov = is_in_fov(curr_phi, half_hfov_rad)
        if not in_fov and stayed_in_fov:
            stayed_in_fov = False
            time_to_loss = tau

        # Check quality
        q = bearing_rate_quality(curr_rate, sigma_br=sigma_br)
        min_q = min(min_q, q)
        max_rate = max(max_rate, abs(curr_rate))

        if not in_fov or q < q_min:
            stayed_valid = False

    return GroundTruthSurvivalResult(
        entity_id=1,
        stayed_in_fov=stayed_in_fov,
        stayed_continuous_valid=stayed_valid,
        min_observed_quality=round(min_q, 4),
        time_to_fov_loss_s=round(time_to_loss, 3) if time_to_loss is not None else None,
        final_bearing_rad=round(curr_phi, 4),
        max_bearing_rate_rad_s=round(max_rate, 4),
        trajectory_points=traj,
    )
