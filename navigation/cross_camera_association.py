"""
ORCA EYE — Stage B: Cross-Camera Entity Association
===================================================
Establishes cross-camera entity correspondences between Camera 0 and Camera 1
in the overlap zone based on:
  1. Spatial ground-plane proximity: ||p0 - p1|| <= d_max
  2. Velocity vector alignment: ||v0 - v1||
  3. Semantic class consistency
  4. Temporal sync proximity

Emits association confidence A_{01} in [0.0, 1.0] and logs all pairing events.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from navigation.camera_observation import CameraObservation


@dataclass
class AssociationRecord:
    """Detailed telemetry for a single cross-camera candidate pairing."""
    timestamp: float
    cam0_track_id: int
    cam1_track_id: int
    spatial_dist_m: float
    velocity_dist_mps: float
    class_match: bool
    time_delta_s: float
    spatial_score: float
    velocity_score: float
    class_score: float
    temporal_score: float
    total_confidence: float
    is_associated: bool


class CrossCameraAssociator:
    """
    Associates entity tracks across CAM0 and CAM1.
    """

    def __init__(
        self,
        tau_assoc: float = 0.70,
        max_spatial_dist_m: float = 1.20,
        sigma_pos: float = 0.60,
        sigma_vel: float = 0.80,
        tau_sync: float = 0.10,
        w_pos: float = 0.45,
        w_vel: float = 0.25,
        w_class: float = 0.20,
        w_time: float = 0.10,
    ) -> None:
        self.tau_assoc = tau_assoc
        self.max_spatial_dist_m = max_spatial_dist_m
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel
        self.tau_sync = tau_sync
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_class = w_class
        self.w_time = w_time

        self.association_history: List[AssociationRecord] = []

    def compute_pairwise_affinity(
        self,
        obs0: CameraObservation,
        obs1: CameraObservation,
        now: Optional[float] = None,
    ) -> Tuple[float, AssociationRecord]:
        """
        Calculates association affinity A(c0, c1, e0, e1) in [0.0, 1.0].
        """
        current_time = time.time() if now is None else now

        # 1. Spatial distance in egocentric ground plane
        dx = obs0.world_pos[0] - obs1.world_pos[0]
        dy = obs0.world_pos[1] - obs1.world_pos[1]
        dist_m = math.hypot(dx, dy)

        # Hard spatial gate: if beyond max distance, affinity is 0.0
        if dist_m > self.max_spatial_dist_m:
            rec = AssociationRecord(
                timestamp=current_time,
                cam0_track_id=obs0.track_id,
                cam1_track_id=obs1.track_id,
                spatial_dist_m=round(dist_m, 3),
                velocity_dist_mps=0.0,
                class_match=(obs0.class_name == obs1.class_name),
                time_delta_s=round(abs(obs0.timestamp - obs1.timestamp), 4),
                spatial_score=0.0,
                velocity_score=0.0,
                class_score=0.0,
                temporal_score=0.0,
                total_confidence=0.0,
                is_associated=False,
            )
            return 0.0, rec

        # Soft continuous spatial score
        s_pos = math.exp(-(dist_m ** 2) / (2.0 * (self.sigma_pos ** 2)))

        # 2. Velocity vector alignment
        dvx = obs0.world_vel[0] - obs1.world_vel[0]
        dvy = obs0.world_vel[1] - obs1.world_vel[1]
        v_dist = math.hypot(dvx, dvy)
        s_vel = math.exp(-(v_dist ** 2) / (2.0 * (self.sigma_vel ** 2)))

        # 3. Class match
        is_class_match = (obs0.class_name == obs1.class_name)
        s_class = 1.0 if is_class_match else 0.0

        # 4. Temporal synchronization
        dt = abs(obs0.timestamp - obs1.timestamp)
        s_time = math.exp(-dt / self.tau_sync)

        # Weighted combination
        confidence = (
            self.w_pos * s_pos +
            self.w_vel * s_vel +
            self.w_class * s_class +
            self.w_time * s_time
        )
        confidence = max(0.0, min(1.0, confidence))

        is_assoc = confidence >= self.tau_assoc

        rec = AssociationRecord(
            timestamp=current_time,
            cam0_track_id=obs0.track_id,
            cam1_track_id=obs1.track_id,
            spatial_dist_m=round(dist_m, 3),
            velocity_dist_mps=round(v_dist, 3),
            class_match=is_class_match,
            time_delta_s=round(dt, 4),
            spatial_score=round(s_pos, 4),
            velocity_score=round(s_vel, 4),
            class_score=round(s_class, 4),
            temporal_score=round(s_time, 4),
            total_confidence=round(confidence, 4),
            is_associated=is_assoc,
        )

        return confidence, rec

    def associate(
        self,
        cam0_observations: List[CameraObservation],
        cam1_observations: List[CameraObservation],
        now: Optional[float] = None,
    ) -> List[Tuple[CameraObservation, CameraObservation, float]]:
        """
        Bipartite greedy association between CAM0 and CAM1 entity observations.
        Returns list of matched tuples: (obs0, obs1, confidence) where confidence >= tau_assoc.
        """
        if not cam0_observations or not cam1_observations:
            return []

        candidates = []
        for obs0 in cam0_observations:
            for obs1 in cam1_observations:
                conf, rec = self.compute_pairwise_affinity(obs0, obs1, now=now)
                self.association_history.append(rec)
                if conf >= self.tau_assoc:
                    candidates.append((conf, obs0, obs1))

        # Sort descending by confidence (greedy optimal assignment)
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched = []
        used_cam0 = set()
        used_cam1 = set()

        for conf, obs0, obs1 in candidates:
            if obs0.track_id not in used_cam0 and obs1.track_id not in used_cam1:
                matched.append((obs0, obs1, conf))
                used_cam0.add(obs0.track_id)
                used_cam1.add(obs1.track_id)

        return matched
