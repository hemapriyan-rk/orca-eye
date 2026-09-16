"""
ORCA EYE — Stage A: Monte Carlo Continuous Survival Probability (rho_FOV)
========================================================================
Implements the continuous survival probability rho_FOV(c, e, t, H) for single-camera
observation validity:

  V_{c,e}(t) = 1 if |phi_{c,e}(t)| <= theta_c  and  q_BR(|phi_dot_{c,e}(t)|) >= q_min
               0 otherwise

  rho_FOV(c, e, t, H) = P( V_{c,e}(t + tau) == 1  forall tau in [0, H] | X_{c,e}(t) )

Evaluated via a vectorized Monte Carlo trajectory propagation reference model
(N = 50--100 samples) to ensure high scientific rigor and real-time execution (< 1 ms).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from navigation.geometry import (
    bearing_rate_quality,
    calculate_bearing,
    calculate_bearing_rate,
    is_in_fov,
)


@dataclass
class RhoFOVPrediction:
    """
    Prediction record for a specific camera and entity.
    """
    camera_id: str
    entity_id: int
    timestamp: float
    horizon_s: float
    rho_fov: float                           # continuous survival probability [0.0, 1.0]
    initial_bearing_rad: float
    initial_bearing_rate: float
    initial_validity: bool
    samples_count: int = 50
    survived_count: int = 0
    ground_truth_valid: Optional[bool] = None  # filled when t + H is reached
    brier_error: Optional[float] = None        # (rho_fov - y)^2


class RhoFOVPredictor:
    """
    Computes continuous FOV survival probability using vectorized Monte Carlo sampling.
    """

    def __init__(
        self,
        half_hfov_rad: float = math.radians(32.5),  # 65 deg full HFOV default
        q_min: float = 0.20,
        sigma_br: float = 0.60,
        num_samples: int = 50,
        horizon_s: float = 2.0,
        dt_sample_s: float = 0.20,
    ) -> None:
        self.half_hfov_rad = half_hfov_rad
        self.q_min = q_min
        self.sigma_br = sigma_br
        self.num_samples = num_samples
        self.horizon_s = horizon_s
        self.dt_sample_s = dt_sample_s

        # Prediction evaluation buffer: stores predictions awaiting ground truth validation
        self.prediction_buffer: List[RhoFOVPrediction] = []
        self.evaluated_history: List[RhoFOVPrediction] = []

    def compute_instantaneous_validity(
        self,
        bearing_rad: float,
        bearing_rate: float,
    ) -> bool:
        """
        Joint Instantaneous Validity V_{c,e}(t):
          V = 1 if |phi| <= theta_c and q_BR(|phi_dot|) >= q_min
        """
        if not is_in_fov(bearing_rad, self.half_hfov_rad):
            return False
        q = bearing_rate_quality(bearing_rate, sigma_br=self.sigma_br)
        return q >= self.q_min

    def predict_survival(
        self,
        entity_id: int,
        bearing_rad: float,
        bearing_rate: float,
        sigma_phi: float = 0.05,       # bearing angular standard deviation (rad)
        sigma_phi_dot: float = 0.15,   # bearing rate standard deviation (rad/s)
        camera_id: str = "PRIMARY",
        current_time: Optional[float] = None,
    ) -> RhoFOVPrediction:
        """
        Estimate continuous survival probability rho_FOV using vectorized Monte Carlo.
        """
        now = time.time() if current_time is None else current_time
        init_valid = self.compute_instantaneous_validity(bearing_rad, bearing_rate)

        # If already outside FOV or tracking quality failed, survival is 0.0
        if not init_valid:
            pred = RhoFOVPrediction(
                camera_id=camera_id,
                entity_id=entity_id,
                timestamp=now,
                horizon_s=self.horizon_s,
                rho_fov=0.0,
                initial_bearing_rad=bearing_rad,
                initial_bearing_rate=bearing_rate,
                initial_validity=False,
                samples_count=self.num_samples,
                survived_count=0,
            )
            self._buffer_prediction(pred)
            return pred

        # Vectorized Monte Carlo perturbation sampling
        # Sample N perturbations for initial bearing and bearing rate
        phi_0_samples = np.random.normal(bearing_rad, sigma_phi, size=self.num_samples)
        phi_dot_samples = np.random.normal(bearing_rate, sigma_phi_dot, size=self.num_samples)

        # Discrete evaluation time steps tau in [dt, 2*dt, ..., H]
        time_steps = np.arange(self.dt_sample_s, self.horizon_s + 1e-4, self.dt_sample_s)
        num_steps = len(time_steps)

        # Broadcast trajectory propagation: (num_steps, num_samples)
        # phi(tau) = phi_0 + phi_dot * tau
        # Assume constant bearing-rate kinematics with stochastic drift
        tau_grid = time_steps[:, np.newaxis]  # (num_steps, 1)
        phi_traj = phi_0_samples + phi_dot_samples * tau_grid  # (num_steps, num_samples)
        phi_dot_traj = np.repeat(phi_dot_samples[np.newaxis, :], num_steps, axis=0)

        # Evaluate FOV boundary condition: |phi(tau)| <= theta_c
        fov_valid = np.abs(phi_traj) <= self.half_hfov_rad

        # Evaluate Tracking Quality condition: q_BR(|phi_dot|) >= q_min
        # q = exp(- phi_dot^2 / (2 * sigma_br^2)) >= q_min <=> |phi_dot| <= sqrt(-2 * sigma_br^2 * ln(q_min))
        max_allowed_phi_dot = math.sqrt(-2.0 * (self.sigma_br ** 2) * math.log(max(self.q_min, 1e-5)))
        quality_valid = np.abs(phi_dot_traj) <= max_allowed_phi_dot

        # Joint validity matrix: (num_steps, num_samples)
        joint_valid = fov_valid & quality_valid

        # Continuous survival requires joint_valid == True across ALL time steps in the horizon
        survived_mask = np.all(joint_valid, axis=0)  # (num_samples,)
        survived_count = int(np.sum(survived_mask))
        rho_fov = float(survived_count) / float(self.num_samples)

        pred = RhoFOVPrediction(
            camera_id=camera_id,
            entity_id=entity_id,
            timestamp=now,
            horizon_s=self.horizon_s,
            rho_fov=round(rho_fov, 4),
            initial_bearing_rad=round(bearing_rad, 4),
            initial_bearing_rate=round(bearing_rate, 4),
            initial_validity=True,
            samples_count=self.num_samples,
            survived_count=survived_count,
        )
        self._buffer_prediction(pred)
        return pred

    def evaluate_pending_predictions(
        self,
        current_time: float,
        current_observations: Dict[int, Tuple[float, float]],  # entity_id -> (bearing_rad, bearing_rate)
    ) -> List[RhoFOVPrediction]:
        """
        Check predictions whose horizon (timestamp + horizon_s) has elapsed,
        compare against observed validity, and calculate calibration error (Brier Score).
        """
        evaluated_now: List[RhoFOVPrediction] = []
        remaining_buffer: List[RhoFOVPrediction] = []

        for pred in self.prediction_buffer:
            if current_time >= (pred.timestamp + pred.horizon_s):
                # Target time reached! Check if entity is currently observed and valid
                if pred.entity_id in current_observations:
                    obs_bearing, obs_rate = current_observations[pred.entity_id]
                    obs_valid = self.compute_instantaneous_validity(obs_bearing, obs_rate)
                else:
                    # Entity was lost/disappeared
                    obs_valid = False

                pred.ground_truth_valid = obs_valid
                y = 1.0 if obs_valid else 0.0
                pred.brier_error = round((pred.rho_fov - y) ** 2, 4)

                evaluated_now.append(pred)
                self.evaluated_history.append(pred)
            else:
                remaining_buffer.append(pred)

        self.prediction_buffer = remaining_buffer
        return evaluated_now

    def get_mean_brier_score(self) -> Optional[float]:
        """Compute mean Brier Score across all evaluated predictions."""
        if not self.evaluated_history:
            return None
        errors = [p.brier_error for p in self.evaluated_history if p.brier_error is not None]
        return float(np.mean(errors)) if errors else None

    def _buffer_prediction(self, pred: RhoFOVPrediction) -> None:
        self.prediction_buffer.append(pred)
        if len(self.prediction_buffer) > 200:
            self.prediction_buffer.pop(0)
