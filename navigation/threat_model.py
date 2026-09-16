"""
ORCA EYE — Stage A: Explicit Threat & Maneuver Conflict Model
=============================================================
Decouples raw object presence from prospective maneuver conflict.

Evaluates prospective trajectory intersections between the wearer's planned
maneuver corridor and moving or stationary spatial entities in the egocentric
ground plane.

Kinematics:
  r(t + tau) = [p_e(t) - p_u(t)] + [v_e(t) - v_u(t)] * tau
  Conflict condition: ||r(t + tau)|| <= d_safe

Analytic Time-To-Conflict (TTC):
  TTC = min { tau >= 0 : ||r(t + tau)|| <= d_safe }
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ThreatLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    CRITICAL = "CRITICAL"


@dataclass
class ThreatRecord:
    """
    Structured threat evaluation for a specific entity against a candidate corridor.
    """
    entity_id: int
    corridor_direction: str
    threat_level: ThreatLevel
    ttc_seconds: Optional[float] = None
    closest_approach_m: float = 999.0
    corridor_overlap_m: float = 0.0
    confidence: float = 1.0
    reason: str = ""


def compute_analytic_ttc(
    rel_pos: Tuple[float, float],
    rel_vel: Tuple[float, float],
    d_safe: float = 0.9,
    max_horizon_s: float = 5.0,
) -> Tuple[Optional[float], float]:
    """
    Compute the continuous analytic Time-To-Conflict (TTC) and distance of
    closest approach (DCA) between wearer and entity.

    Parameters
    ----------
    rel_pos : (rx0, ry0) initial relative position [p_e - p_u] (meters)
    rel_vel : (vx, vy) relative velocity [v_e - v_u] (m/s)
    d_safe  : safety sphere radius (meters)
    max_horizon_s : maximum lookahead horizon (seconds)

    Returns
    -------
    (ttc, dca)
      ttc : float seconds to conflict (or None if no collision within max_horizon)
      dca : float minimum relative distance achieved
    """
    rx, ry = rel_pos
    vx, vy = rel_vel

    r0_sq = rx * rx + ry * ry
    r0 = math.sqrt(r0_sq)

    # If already inside safety bubble, collision is instantaneous
    if r0 <= d_safe:
        return 0.0, r0

    v_sq = vx * vx + vy * vy
    if v_sq < 1e-6:
        # Stationary relative motion
        return None, r0

    # Quadratic coefficients: a*tau^2 + b*tau + c = 0
    a = v_sq
    b = 2.0 * (rx * vx + ry * vy)
    c = r0_sq - (d_safe * d_safe)

    # Time of closest approach: tau_c = - (r0 . v) / ||v||^2 = -b / (2a)
    tau_c = -b / (2.0 * a)
    if tau_c <= 0.0:
        # Diverging trajectories
        dca = r0
    else:
        dca_sq = max(0.0, r0_sq - (b * b) / (4.0 * a))
        dca = math.sqrt(dca_sq)

    # Check for intersection with d_safe sphere
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        # Never intersects d_safe sphere
        return None, dca

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Smallest non-negative root
    valid_roots = [t for t in (t1, t2) if 0.0 <= t <= max_horizon_s]
    if valid_roots:
        ttc = min(valid_roots)
        return float(round(ttc, 2)), float(round(dca, 2))

    return None, float(round(dca, 2))


class ThreatModel:
    """
    Evaluates dynamic conflicts and generates ThreatRecords for spatial entities
    relative to proposed walking trajectories.
    """

    def __init__(
        self,
        d_safe: float = 0.9,
        horizon_s: float = 3.0,
        user_walk_speed_m_s: float = 1.0,
    ) -> None:
        self.d_safe = d_safe
        self.horizon_s = horizon_s
        self.user_walk_speed = user_walk_speed_m_s

    def evaluate_entity_corridor(
        self,
        entity,  # SpatialEntity or EntityKinematics
        corridor_direction: str,
        corridor_angle_deg: float,
        wearer_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> ThreatRecord:
        """
        Evaluate threat level and TTC for an entity against a corridor trajectory.
        """
        # Egocentric coordinates (meters)
        # SpatialEntity ground_position is [px, py] in pixel frame;
        # if entity has relative_distance and relative_bearing, compute metric ground coords:
        if hasattr(entity, "ground_x") and hasattr(entity, "ground_y"):
            ex = float(entity.ground_x)
            ey = float(entity.ground_y)
        else:
            # Convert from relative bearing and distance
            # relative_distance [0, 1] normalized: roughly 0.5m to 6.0m range
            dist_m = max(0.5, float(getattr(entity, "relative_distance", 0.5)) * 6.0)
            bearing_rad = math.radians(float(getattr(entity, "relative_bearing", 0.0)))
            ex = dist_m * math.sin(bearing_rad)
            ey = dist_m * math.cos(bearing_rad)

        # Entity ground velocity (m/s)
        if hasattr(entity, "vx_mps") and hasattr(entity, "vy_mps"):
            evx = float(entity.vx_mps)
            evy = float(entity.vy_mps)
        elif hasattr(entity, "velocity"):
            # Normalize pixel velocity to approx m/s (assuming ~30 fps)
            evx = float(entity.velocity[0]) * 0.02
            evy = float(entity.velocity[1]) * 0.02
        else:
            evx, evy = 0.0, 0.0

        # Wearer planned velocity along corridor angle
        if corridor_direction == "STOP":
            uvx, uvy = 0.0, 0.0
        else:
            corridor_rad = math.radians(corridor_angle_deg)
            uvx = self.user_walk_speed * math.sin(corridor_rad)
            uvy = self.user_walk_speed * math.cos(corridor_rad)

        rel_pos = (ex - wearer_pos[0], ey - wearer_pos[1])
        rel_vel = (evx - uvx, evy - uvy)

        ttc, dca = compute_analytic_ttc(
            rel_pos=rel_pos,
            rel_vel=rel_vel,
            d_safe=self.d_safe,
            max_horizon_s=self.horizon_s,
        )

        # Lateral corridor overlap
        corridor_rad = math.radians(corridor_angle_deg)
        # Lateral offset from the candidate corridor's projected centerline
        lat_offset = abs(ex * math.cos(corridor_rad) - ey * math.sin(corridor_rad))
        corridor_half_w = 0.45  # 0.9m wide corridor
        entity_radius = 0.25
        corridor_overlap = max(0.0, (corridor_half_w + entity_radius) - lat_offset)

        # Classify ThreatLevel
        if ttc is not None and lat_offset <= (corridor_half_w + self.d_safe):
            if ttc <= 1.2:
                threat_level = ThreatLevel.CRITICAL
                reason = f"Imminent collision in {ttc:.1f}s"
            elif ttc <= 2.2:
                threat_level = ThreatLevel.MEDIUM
                reason = f"Dynamic conflict predicted in {ttc:.1f}s"
            else:
                threat_level = ThreatLevel.LOW
                reason = f"Approaching conflict in {ttc:.1f}s"
        else:
            if dca <= self.d_safe and corridor_overlap > 0:
                threat_level = ThreatLevel.MEDIUM
                reason = f"Static or close hazard (DCA: {dca:.2f}m)"
            elif corridor_overlap > 0.05:
                threat_level = ThreatLevel.LOW
                reason = "Spatial footprint in corridor"
            else:
                threat_level = ThreatLevel.NONE
                reason = "No trajectory conflict"

        track_id = int(getattr(entity, "track_id", 0))
        confidence = float(getattr(entity, "tracking_confidence", 1.0))

        return ThreatRecord(
            entity_id=track_id,
            corridor_direction=corridor_direction,
            threat_level=threat_level,
            ttc_seconds=ttc,
            closest_approach_m=dca,
            corridor_overlap_m=round(corridor_overlap, 2),
            confidence=confidence,
            reason=reason,
        )
