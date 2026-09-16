"""
ORCA EYE — Stage 6b: Dynamic Object Conflict Reasoning
=====================================================
Evaluates moving obstacles (pedestrians, vehicles, cyclists) against candidate
walking corridors to predict prospective intersections.

For each dynamic entity:
  - Extrapolates kinematic trajectory over a forward time horizon
  - Determines if the prospective trajectory intersects candidate walking corridors
  - Estimates time-to-conflict and conflict confidence
  - Feeds dynamic conflict penalties directly into the path scorer

Inputs  : List[SpatialEntity], List[PathCandidate], frame dimensions, fps
Outputs : Dict[str, List[DynamicConflict]], per-candidate dynamic penalties
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DynamicConflict:
    """
    A prospective spatio-temporal collision between a moving entity and a walking corridor.
    """
    track_id: int
    class_name: str
    motion_direction: str                              # approaching | crossing | parallel
    corridor_direction: str                            # STRAIGHT | LEFT | RIGHT | etc.
    time_to_conflict_s: Optional[float]                # seconds until conflict, or None if uncertain
    conflict_confidence: float                         # [0.0, 1.0]
    predicted_crossing_pos: List[float]                # [px, py] pixel coordinates
    is_imminent: bool = False                          # True if conflict is < 1.5s away
    has_conflict: bool = True
    time_to_conflict: Optional[float] = None

    def __post_init__(self) -> None:
        if self.time_to_conflict is None:
            self.time_to_conflict = self.time_to_conflict_s


class DynamicConflictEngine:
    """
    Predictive collision checker between dynamic SpatialEntities and walking corridors.
    """

    def __init__(self, cfg: dict) -> None:
        dyn_cfg = cfg.get("dynamic_conflict", {})
        self.enabled: bool = dyn_cfg.get("enabled", True)
        self.time_horizon_s: float = dyn_cfg.get("time_horizon_s", 2.5)
        self.conflict_dist_thresh: float = dyn_cfg.get("conflict_distance_threshold", 45.0)
        self.min_conf: float = dyn_cfg.get("min_conflict_confidence", 0.40)

    def evaluate_conflicts(
        self,
        spatial_entities: List,
        candidates: List,
        frame_w: int = 640,
        frame_h: int = 480,
        grid_rows: int = 12,
        grid_cols: int = 20,
        fps: float = 30.0,
    ) -> Dict[str, List[DynamicConflict]]:
        """
        Evaluate prospective trajectory intersections for all candidate corridors.

        Returns
        -------
        Dict[str, List[DynamicConflict]] — mapping corridor_direction -> list of active conflicts
        """
        conflicts_by_dir: Dict[str, List[DynamicConflict]] = {
            cand.direction: [] for cand in candidates
        }

        if not self.enabled or not spatial_entities or not candidates:
            return conflicts_by_dir

        eff_fps = max(float(fps), 10.0)
        horizon_frames = int(self.time_horizon_s * eff_fps)
        cw = frame_w // max(grid_cols, 1)
        ch = frame_h // max(grid_rows, 1)

        # Filter entities with meaningful dynamic motion
        dynamic_entities = [
            e for e in spatial_entities
            if e.motion_direction in ("approaching", "crossing", "parallel")
            and e.tracking_confidence >= self.min_conf
        ]

        for entity in dynamic_entities:
            vx, vy = entity.velocity
            x0, y0 = entity.ground_position
            speed = math.hypot(vx, vy)

            # Skip effectively stationary entities
            if speed < 1.0:
                continue

            for cand in candidates:
                if cand.is_stop or not cand.points:
                    continue

                # Corridor polygon / centerline pixels
                cand_pixels: List[Tuple[float, float]] = [
                    (c * cw + cw / 2.0, r * ch + ch / 2.0)
                    for r, c in cand.points
                ]
                corridor_half_w = max((cand.width * cw) / 2.0, 15.0)
                safety_margin = corridor_half_w + self.conflict_dist_thresh

                # Project entity forward in time steps (every 2 frames)
                conflict_found = False
                for step in range(2, horizon_frames + 1, 2):
                    pred_x = x0 + step * vx
                    pred_y = y0 + step * vy

                    # Bounds check
                    if pred_x < -50 or pred_x > frame_w + 50 or pred_y < -50 or pred_y > frame_h + 50:
                        break

                    # Distance from projected point to any corridor point
                    for cpx, cpy in cand_pixels:
                        dist = math.hypot(pred_x - cpx, pred_y - cpy)
                        if dist < safety_margin:
                            time_to_conflict = round(step / eff_fps, 2)
                            # Higher confidence if entity has high tracking confidence and close approach
                            base_conf = entity.tracking_confidence
                            temporal_decay = max(0.2, 1.0 - (step / horizon_frames) * 0.5)
                            conf_val = round(float(base_conf * temporal_decay), 3)

                            conflict = DynamicConflict(
                                track_id=entity.track_id,
                                class_name=entity.class_name,
                                motion_direction=entity.motion_direction,
                                corridor_direction=cand.direction,
                                time_to_conflict_s=time_to_conflict,
                                conflict_confidence=conf_val,
                                predicted_crossing_pos=[round(pred_x, 1), round(pred_y, 1)],
                                is_imminent=(time_to_conflict < 1.5),
                            )
                            conflicts_by_dir[cand.direction].append(conflict)
                            conflict_found = True
                            break

                    if conflict_found:
                        break

        # Attach dynamic conflict penalty score directly to PathCandidate instances
        for cand in candidates:
            c_list = conflicts_by_dir.get(cand.direction, [])
            if c_list:
                max_conf = max(c.conflict_confidence for c in c_list)
                cand.dynamic_conflict_score = round(max_conf, 3)
            else:
                cand.dynamic_conflict_score = 0.0

        return conflicts_by_dir
