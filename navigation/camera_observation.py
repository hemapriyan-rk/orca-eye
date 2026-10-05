"""
ORCA EYE — Stage B: Multi-Camera Observation & Shared Entity State Model
========================================================================
Represents:
  1. CameraModel: Calibrated geometric definition and coordinate transformations.
  2. CameraObservation: Single-camera detection, kinematics, and local rho_FOV.
  3. MultiCameraEntityState: Fused global entity state with cross-camera tracking,
     predictive responsibility assignment, and handover metrics.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from navigation.geometry import is_in_fov


@dataclass
class CameraModel:
    """
    Calibrated representation of a camera in the multi-camera array.
    """
    camera_id: str
    yaw_deg: float
    half_hfov_rad: float = math.radians(32.5)  # 65.0 deg full HFOV default
    mount_offset_m: Tuple[float, float, float] = (0.0, 0.0, 1.4)  # (x, y, z) in wearer frame

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(self.half_hfov_rad * 2.0)

    @property
    def min_world_bearing_rad(self) -> float:
        return self.yaw_rad - self.half_hfov_rad

    @property
    def max_world_bearing_rad(self) -> float:
        return self.yaw_rad + self.half_hfov_rad

    def world_to_local_bearing(self, world_bearing_rad: float) -> float:
        """Transforms egocentric world bearing phi_w into camera optical bearing phi_c."""
        local = world_bearing_rad - self.yaw_rad
        # Normalize to [-pi, pi]
        return (local + math.pi) % (2.0 * math.pi) - math.pi

    def local_to_world_bearing(self, local_bearing_rad: float) -> float:
        """Transforms camera optical bearing phi_c into egocentric world bearing phi_w."""
        world = local_bearing_rad + self.yaw_rad
        return (world + math.pi) % (2.0 * math.pi) - math.pi

    def is_in_fov(self, local_bearing_rad: float) -> bool:
        """Checks if local optical bearing is within physical sensor bounds."""
        return is_in_fov(local_bearing_rad, self.half_hfov_rad)

    def is_world_bearing_in_fov(self, world_bearing_rad: float) -> bool:
        """Checks if an egocentric world bearing falls within this camera's optical cone."""
        loc = self.world_to_local_bearing(world_bearing_rad)
        return self.is_in_fov(loc)

    def compute_overlap(self, other: "CameraModel") -> Tuple[float, float, float]:
        """
        Computes the physical angular overlap between this camera and another.
        Returns: (min_overlap_world_rad, max_overlap_world_rad, overlap_width_rad)
        """
        min_overlap = max(self.min_world_bearing_rad, other.min_world_bearing_rad)
        max_overlap = min(self.max_world_bearing_rad, other.max_world_bearing_rad)
        width = max(0.0, max_overlap - min_overlap)
        return min_overlap, max_overlap, width


@dataclass
class CameraObservation:
    """
    Observation of an entity from a specific camera sensor.
    """
    camera_id: str
    track_id: int
    timestamp: float
    local_bearing_rad: float
    local_bearing_deg: float
    local_bearing_rate_rad_s: float
    world_bearing_rad: float
    world_bearing_deg: float
    distance_m: float
    world_pos: Tuple[float, float]               # (x_w, y_w) in meters
    world_vel: Tuple[float, float] = (0.0, 0.0)  # (vx_w, vy_w) in m/s
    rho_fov: float = 1.0                         # Predicted continuous survival [0.0, 1.0]
    detection_confidence: float = 0.90
    class_name: str = "person"
    in_fov: bool = True
    bbox: Optional[List[float]] = None           # [x1, y1, x2, y2] normalized


@dataclass
class MultiCameraEntityState:
    """
    Fused multi-camera state for a single physical entity.
    Tracks responsibility, cross-camera associations, and predictive metrics.
    """
    global_entity_id: int
    class_name: str = "person"
    primary_camera_id: str = "CAM0"
    responsible_camera_id: str = "CAM0"
    responsibility_state: str = "PRIMARY"        # "PRIMARY" | "PRE_ARM" | "TRANSFER" | "SECONDARY" | "RELEASED"

    # Per-camera observation cache
    observations: Dict[str, CameraObservation] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    # State machine transition timestamps
    state_enter_timestamp: Optional[float] = None
    pre_arm_timestamp: Optional[float] = None
    transfer_timestamp: Optional[float] = None
    primary_loss_timestamp: Optional[float] = None
    secondary_acq_timestamp: Optional[float] = None

    # Association telemetry
    association_confidence: float = 0.0
    associated_track_ids: Dict[str, int] = field(default_factory=dict)

    # Handover metrics
    handover_lead_time_s: Optional[float] = None
    observation_gap_s: Optional[float] = None

    @property
    def dwell_time_s(self) -> float:
        """Elapsed time since last responsibility state change."""
        if self.state_enter_timestamp is None:
            return 0.0
        return time.time() - self.state_enter_timestamp

    def get_max_rho_fov(self) -> float:
        """Returns max_c rho_FOV(c, e) across all reporting cameras."""
        if not self.observations:
            return 0.0
        return max(obs.rho_fov for obs in self.observations.values())

    def get_responsible_rho_fov(self) -> float:
        """Returns rho_FOV of the currently responsible camera."""
        if self.responsible_camera_id in self.observations:
            return self.observations[self.responsible_camera_id].rho_fov
        return 0.0

    def get_representative_world_pos(self) -> Tuple[float, float]:
        """Returns best ground position estimate from responsible camera or first available."""
        if self.responsible_camera_id in self.observations:
            return self.observations[self.responsible_camera_id].world_pos
        if self.observations:
            return next(iter(self.observations.values())).world_pos
        return (0.0, 0.0)

    def get_representative_world_bearing_rad(self) -> float:
        """Returns best ground bearing estimate from responsible camera or first available."""
        if self.responsible_camera_id in self.observations:
            return self.observations[self.responsible_camera_id].world_bearing_rad
        if self.observations:
            return next(iter(self.observations.values())).world_bearing_rad
        return 0.0
