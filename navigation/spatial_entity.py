"""
ORCA EYE — Stage 5a: Unified Spatial Entity
===========================================
Canonical cross-module data contract connecting perception (Detection + Tracking + Depth)
to navigation and spatial planning.

SpatialEntity fuses:
  - Detection identity and classification
  - Temporal continuity from tracking
  - Robust depth ROI relative distance
  - Motion direction and velocity
  - Ground-contact coordinates and lateral bearing
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpatialEntity:
    """
    Unified spatial representation of an obstacle or agent in the environment.
    All distance values are explicitly RELATIVE (0.0 = closest to user feet, 1.0 = farthest).
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]                                    # [x1, y1, x2, y2]
    image_position: List[float]                        # [cx, cy]
    ground_position: List[float]                       # [bottom_cx, bottom_y]
    relative_bearing: float                            # degrees from optical axis (-left, +right)
    relative_distance: float                           # normalized depth [0.0, 1.0] (0=nearest)
    velocity: List[float]                              # [vx, vy] px/frame
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0])
    motion_direction: str = "uncertain"                # stationary | approaching | receding | crossing | parallel | uncertain
    tracking_confidence: float = 1.0
    depth_confidence: float = 1.0
    depth_valid: bool = True
    navigation_relevance: float = 0.5                  # [0.0, 1.0] priority for navigation


def create_spatial_entities(
    tracks: List,
    depth_map: Optional[np.ndarray],
    depth_estimator,
    frame_w: int,
    frame_h: int,
    hfov_deg: float = 65.0,
) -> List[SpatialEntity]:
    """
    Construct unified SpatialEntity instances from active tracks and depth.

    Parameters
    ----------
    tracks          : List[TrackedObject]
    depth_map       : np.ndarray | None (normalized relative depth map)
    depth_estimator : DepthEstimator
    frame_w         : int
    frame_h         : int
    hfov_deg        : float (horizontal FOV assumption for bearing calculation)

    Returns
    -------
    List[SpatialEntity]
    """
    entities: List[SpatialEntity] = []
    cx_cam = frame_w / 2.0

    for trk in tracks:
        # 1. Ground contact position
        if hasattr(trk, "bottom_center") and trk.bottom_center:
            bcx, bcy = trk.bottom_center
        else:
            bcx = (trk.bbox[0] + trk.bbox[2]) / 2.0
            bcy = float(trk.bbox[3])

        # 2. Relative bearing in degrees from optical axis
        # Normalized x offset [-1, 1] * (hfov / 2)
        norm_x = (bcx - cx_cam) / max(cx_cam, 1.0)
        bearing_deg = norm_x * (hfov_deg / 2.0)

        # 3. Robust depth extraction for object ROI
        if depth_map is not None and depth_estimator is not None:
            depth_stat = depth_estimator.extract_object_depth(depth_map, trk.bbox)
            est_dist = depth_stat.estimated_distance
            dist_conf = depth_stat.distance_confidence
            depth_valid = depth_stat.depth_valid
        else:
            est_dist = 0.5
            dist_conf = 0.5
            depth_valid = False

        # 4. Tracking confidence based on track age and disappeared status
        age_factor = min(1.0, trk.age / max(trk.age + trk.disappeared, 1))
        track_conf = float(np.clip(trk.confidence * age_factor, 0.1, 1.0))

        # 5. Motion direction
        motion_dir = getattr(trk, "motion_state", "uncertain")

        # 6. Navigation relevance:
        # Objects closer to the camera (lower est_dist), near the center (small bearing),
        # or approaching have higher relevance
        dist_factor = 1.0 - est_dist  # 0=farthest, 1=closest
        bearing_factor = 1.0 - min(1.0, abs(bearing_deg) / (hfov_deg / 2.0))
        approach_bonus = 0.25 if motion_dir == "approaching" else (0.15 if motion_dir == "crossing" else 0.0)

        nav_relevance = float(np.clip(
            0.50 * dist_factor + 0.30 * bearing_factor + approach_bonus,
            0.0, 1.0
        ))

        entity = SpatialEntity(
            track_id=trk.track_id,
            class_id=trk.class_id,
            class_name=trk.class_name,
            confidence=trk.confidence,
            bbox=list(trk.bbox),
            image_position=[float(trk.center[0]), float(trk.center[1])],
            ground_position=[float(bcx), float(bcy)],
            relative_bearing=round(bearing_deg, 1),
            relative_distance=round(est_dist, 3),
            velocity=list(trk.velocity),
            acceleration=getattr(trk, "acceleration", [0.0, 0.0]),
            motion_direction=motion_dir,
            tracking_confidence=round(track_conf, 3),
            depth_confidence=round(dist_conf, 3),
            depth_valid=depth_valid,
            navigation_relevance=round(nav_relevance, 3),
        )
        entities.append(entity)

    # Sort entities by navigation relevance (most critical first)
    entities.sort(key=lambda e: e.navigation_relevance, reverse=True)
    return entities
