"""
ORCA EYE — Stage A: Egocentric Physical Geometry & Angular Kinematics
====================================================================
Establishes the physical egocentric coordinate system, ground bearing,
calibrated field-of-view validity bounds, bearing-rate kinematics, and
tracking quality metrics.

Coordinate Conventions:
  Wearer Egocentric Frame:
    x : lateral offset (meters, positive = right, negative = left)
    y : forward distance along ground plane (meters, positive = forward)
    z : vertical elevation (meters, positive = upward)

  Ground Bearing:
    phi = atan2(x, y)
    phi = 0 is straight ahead along the forward heading axis (+y).
    phi > 0 is to the wearer's right.
    phi < 0 is to the wearer's left.

  Horizontal FOV Boundary Condition:
    |phi| <= theta_c  (where theta_c is the calibrated half-HFOV in radians)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def calculate_bearing(x: float, y: float) -> float:
    """
    Calculate the horizontal ground bearing phi in radians from egocentric (x, y).
    x: lateral offset (meters, right positive)
    y: forward distance (meters, forward positive)

    Returns phi in [-pi, pi]. Straight ahead is 0.0 rad.
    """
    if y <= 0.0 and x == 0.0:
        return 0.0
    return math.atan2(x, max(y, 1e-6))


def calculate_bearing_deg(x: float, y: float) -> float:
    """Calculate horizontal ground bearing in degrees (-180 to +180)."""
    return math.degrees(calculate_bearing(x, y))


def is_in_fov(bearing_rad: float, half_hfov_rad: float) -> bool:
    """
    Evaluate if ground bearing phi satisfies the horizontal camera FOV boundary condition:
      |phi| <= theta_c
    """
    return abs(bearing_rad) <= half_hfov_rad


def calculate_bearing_rate(phi_curr: float, phi_prev: float, dt: float) -> float:
    """
    Calculate angular bearing rate (phi_dot) in radians per second:
      phi_dot = (phi(t) - phi(t - dt)) / dt
    Handles circular wrap-around across [-pi, pi].
    """
    if dt <= 1e-6:
        return 0.0

    # Angular difference wrapped to [-pi, pi]
    diff = phi_curr - phi_prev
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi

    return diff / dt


def bearing_rate_quality(
    bearing_rate: float,
    sigma_br: float = 0.6,
    min_quality: float = 0.05,
) -> float:
    """
    Evaluate tracking quality metric q_BR(|phi_dot|) based on bearing rate:
      q_BR(|phi_dot|) = exp( - phi_dot^2 / (2 * sigma_br^2) )

    High bearing rates (e.g. fast crossing entities or aggressive head turns)
    degrade tracking quality and perceptual stability.
    """
    q = math.exp(- (bearing_rate ** 2) / (2.0 * (sigma_br ** 2)))
    return float(np.clip(q, min_quality, 1.0))


def optical_to_wearer_ego(x_c: float, y_c: float, z_c: float) -> Tuple[float, float, float]:
    """
    Transform from OpenCV camera optical frame (X_c right, Y_c down, Z_c forward)
    to wearer egocentric frame (x right, y forward ground, z up).
    """
    return (x_c, z_c, -y_c)


def wearer_ego_to_optical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Transform from wearer egocentric frame (x right, y forward ground, z up)
    to camera optical frame (X_c right, Y_c down, Z_c forward).
    """
    return (x, -z, y)


@dataclass
class EntityKinematics:
    """
    Kinematic state of a tracked spatial entity in the egocentric frame.
    Maintains monotonic timestamps and history for accurate numerical differentiation.
    """
    track_id: int
    timestamp: float                       # monotonic timestamp (seconds)
    x: float                               # lateral offset (m)
    y: float                               # forward distance (m)
    z: float = 0.0                         # vertical elevation (m)
    vx: float = 0.0                        # lateral velocity (m/s)
    vy: float = 0.0                        # forward velocity (m/s)
    bearing_rad: float = 0.0               # ground bearing phi (rad)
    bearing_rate: float = 0.0              # bearing rate phi_dot (rad/s)
    tracking_quality: float = 1.0          # q_BR(|phi_dot|) [0, 1]
    in_fov: bool = True                    # |phi| <= theta_c
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # [(t, x, y)]

    def update(
        self,
        new_timestamp: float,
        new_x: float,
        new_y: float,
        half_hfov_rad: float,
        new_z: float = 0.0,
        sigma_br: float = 0.6,
    ) -> None:
        """Update kinematic state with a new observation."""
        dt = new_timestamp - self.timestamp
        if dt > 1e-4:
            # Velocity estimates
            self.vx = (new_x - self.x) / dt
            self.vy = (new_y - self.y) / dt

            # Ground bearing and bearing rate
            prev_bearing = self.bearing_rad
            new_bearing = calculate_bearing(new_x, new_y)
            self.bearing_rad = new_bearing
            self.bearing_rate = calculate_bearing_rate(new_bearing, prev_bearing, dt)
        else:
            self.bearing_rad = calculate_bearing(new_x, new_y)

        self.x = new_x
        self.y = new_y
        self.z = new_z
        self.timestamp = new_timestamp
        self.in_fov = is_in_fov(self.bearing_rad, half_hfov_rad)
        self.tracking_quality = bearing_rate_quality(self.bearing_rate, sigma_br=sigma_br)

        self.history.append((new_timestamp, new_x, new_y))
        if len(self.history) > 30:
            self.history.pop(0)
