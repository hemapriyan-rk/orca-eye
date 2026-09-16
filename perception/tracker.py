"""
ORCA EYE — Stage 3: Centroid Tracker  (Phase 1 Enhanced)
======================================================
Provides temporal continuity across frames using centroid + IoU matching.
Features:
  - Explicit TrackState lifecycle: TENTATIVE -> CONFIRMED -> TEMPORARILY_LOST -> REACQUIRED -> TERMINATED
  - Ground-plane contact tracking (bottom_center and history)
  - Multi-frame smoothed velocity regression and acceleration calculation
  - Semantic motion classification (stationary, approaching, receding, crossing, parallel, uncertain)
  - Full backward compatibility with existing TrackedObject data contract

Inputs  : List[Detection] from detector
Outputs : List[TrackedObject]
"""

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track State Constants
# ---------------------------------------------------------------------------

STATE_TENTATIVE        = "TENTATIVE"
STATE_CONFIRMED        = "CONFIRMED"
STATE_TEMPORARILY_LOST = "TEMPORARILY_LOST"
STATE_REACQUIRED       = "REACQUIRED"
STATE_TERMINATED       = "TERMINATED"

class TrackState:
    TENTATIVE = STATE_TENTATIVE
    CONFIRMED = STATE_CONFIRMED
    TEMPORARILY_LOST = STATE_TEMPORARILY_LOST
    REACQUIRED = STATE_REACQUIRED
    TERMINATED = STATE_TERMINATED

MOTION_STATIONARY  = "stationary"
MOTION_APPROACHING = "approaching"
MOTION_RECEDING    = "receding"
MOTION_CROSSING    = "crossing"
MOTION_PARALLEL    = "parallel"
MOTION_UNCERTAIN   = "uncertain"


# ---------------------------------------------------------------------------
# Data Contract — TrackedObject & Track
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """
    A temporally tracked object with continuity, kinematic state, and motion pattern.
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]                                    # [x1, y1, x2, y2]
    center: List[float] = field(default_factory=list)  # [cx, cy]
    area: float = 0.0
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [vx, vy]
    age: int = 1                                       # total frames since first seen
    last_seen: int = 1                                 # frame_id of last detection
    is_stable: bool = True                             # True if age >= min_track_age
    disappeared: int = 0                               # consecutive frames without detection match
    # History for analysis
    center_history: List[List[float]] = field(default_factory=list)
    # Phase 1 additions:
    bottom_center: List[float] = field(default_factory=list)
    bottom_center_history: List[List[float]] = field(default_factory=list)
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0])
    area_history: List[float] = field(default_factory=list)
    state: str = STATE_TENTATIVE
    motion_state: str = MOTION_UNCERTAIN
    missed_frames: int = 0
    last_update_timestamp: float = field(default_factory=time.time)
    track_state: Optional[str] = None

    def __post_init__(self) -> None:
        if self.track_state is not None:
            self.state = self.track_state
        elif self.state:
            self.track_state = self.state
        if not self.bottom_center and len(self.bbox) == 4:
            x1, y1, x2, y2 = self.bbox
            self.bottom_center = [float((x1 + x2) / 2.0), float(y2)]
        if not self.center_history and self.center:
            self.center_history = [[float(self.center[0]), float(self.center[1])]]
        if not self.bottom_center_history and self.bottom_center:
            self.bottom_center_history = [[float(self.bottom_center[0]), float(self.bottom_center[1])]]
        if not self.area_history and self.area:
            self.area_history = [float(self.area)]
        self.missed_frames = self.disappeared


# Canonical Alias
Track = TrackedObject


# ---------------------------------------------------------------------------
# Centroid Tracker
# ---------------------------------------------------------------------------

class CentroidTracker:
    """
    Centroid + IoU matching tracker with kinematic estimation.
    """

    def __init__(self, cfg: dict) -> None:
        self.max_disappeared: int = cfg.get("max_disappeared", 10)
        self.max_distance: float = cfg.get("max_distance", 150)
        self.iou_weight: float = cfg.get("iou_weight", 0.6)
        self.centroid_weight: float = cfg.get("centroid_weight", 0.4)
        self.velocity_alpha: float = cfg.get("velocity_alpha", 0.3)
        self.min_track_age: int = cfg.get("min_track_age", 2)

        # Phase 1 parameters
        self.history_window: int = cfg.get("history_window", 5)
        self.stationary_thresh: float = cfg.get("stationary_velocity_thresh", 1.5)
        self.approaching_area_rate_thresh: float = cfg.get("approaching_area_rate_thresh", 0.04)
        self.crossing_ratio_thresh: float = cfg.get("crossing_ratio_thresh", 2.0)

        self._next_id: int = 0
        self._tracks: Dict[int, TrackedObject] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List,
        frame_id: int,
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections for this frame.
        """
        # If no detections, age all tracks and cull expired
        if len(detections) == 0:
            for tid in list(self._tracks.keys()):
                trk = self._tracks[tid]
                trk.disappeared += 1
                trk.missed_frames = trk.disappeared
                trk.state = STATE_TEMPORARILY_LOST
                if trk.disappeared > self.max_disappeared:
                    trk.state = STATE_TERMINATED
                    logger.debug("Track %d expired.", tid)
                    del self._tracks[tid]
            return list(self._tracks.values())

        # No existing tracks -> register all
        if len(self._tracks) == 0:
            for det in detections:
                self._register(det, frame_id)
            return list(self._tracks.values())

        # Match existing tracks to detections
        track_ids = list(self._tracks.keys())
        trk_centers = np.array(
            [[t.center[0], t.center[1]] for t in self._tracks.values()], dtype=float
        )
        trk_bboxes = np.array([t.bbox for t in self._tracks.values()], dtype=float)

        det_centers = np.array([[d.center[0], d.center[1]] for d in detections], dtype=float)
        det_bboxes = np.array([d.bbox for d in detections], dtype=float)

        # Build cost matrix
        D_dist = np.linalg.norm(trk_centers[:, np.newaxis] - det_centers[np.newaxis, :], axis=2)
        D_dist_norm = np.clip(D_dist / max(self.max_distance, 1.0), 0.0, 1.0)

        ious = np.zeros((len(self._tracks), len(detections)), dtype=float)
        for i, tb in enumerate(trk_bboxes):
            for j, db in enumerate(det_bboxes):
                ious[i, j] = self._compute_iou(tb, db)
        D_iou = 1.0 - ious

        C = self.centroid_weight * D_dist_norm + self.iou_weight * D_iou

        # Greedy bipartite matching
        matched_trk: set = set()
        matched_det: set = set()

        if C.size > 0:
            flat_indices = np.argsort(C.ravel())
            for idx in flat_indices:
                t_idx, d_idx = divmod(idx, len(detections))
                if t_idx in matched_trk or d_idx in matched_det:
                    continue
                if D_dist[t_idx, d_idx] > self.max_distance and ious[t_idx, d_idx] < 0.10:
                    continue
                tid = track_ids[t_idx]
                self._update_track(self._tracks[tid], detections[d_idx], frame_id)
                matched_trk.add(t_idx)
                matched_det.add(d_idx)

        # Unmatched tracks -> age and flag temporarily lost
        for i, tid in enumerate(track_ids):
            if i not in matched_trk:
                trk = self._tracks[tid]
                trk.disappeared += 1
                trk.missed_frames = trk.disappeared
                trk.state = STATE_TEMPORARILY_LOST
                if trk.disappeared > self.max_disappeared:
                    trk.state = STATE_TERMINATED
                    logger.debug("Track %d expired.", tid)
                    del self._tracks[tid]

        # Unmatched detections -> register as new
        for j, det in enumerate(detections):
            if j not in matched_det:
                self._register(det, frame_id)

        return list(self._tracks.values())

    def get_tracks(self) -> List[TrackedObject]:
        return list(self._tracks.values())

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self._tracks.get(track_id)

    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def to_dict_list(self) -> List[dict]:
        return [
            {
                "track_id": t.track_id,
                "class_id": t.class_id,
                "class_name": t.class_name,
                "confidence": round(t.confidence, 4),
                "bbox": t.bbox,
                "center": [round(c, 1) for c in t.center],
                "bottom_center": [round(bc, 1) for bc in t.bottom_center],
                "velocity": [round(v, 2) for v in t.velocity],
                "acceleration": [round(a, 2) for a in t.acceleration],
                "age": t.age,
                "is_stable": t.is_stable,
                "state": t.state,
                "motion_state": t.motion_state,
                "disappeared": t.disappeared,
            }
            for t in self._tracks.values()
        ]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _register(self, det, frame_id: int) -> None:
        tid = self._next_id
        self._next_id += 1
        cx, cy = float(det.center[0]), float(det.center[1])
        x1, y1, x2, y2 = det.bbox
        bcx, bcy = float((x1 + x2) / 2.0), float(y2)
        initial_state = STATE_CONFIRMED if self.min_track_age <= 1 else STATE_TENTATIVE

        trk = TrackedObject(
            track_id=tid,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=list(det.bbox),
            center=[cx, cy],
            bottom_center=[bcx, bcy],
            area=float(det.area),
            velocity=[0.0, 0.0],
            acceleration=[0.0, 0.0],
            age=1,
            last_seen=frame_id,
            is_stable=(self.min_track_age <= 1),
            disappeared=0,
            missed_frames=0,
            center_history=[[cx, cy]],
            bottom_center_history=[[bcx, bcy]],
            area_history=[float(det.area)],
            state=initial_state,
            motion_state=MOTION_UNCERTAIN,
            last_update_timestamp=time.time(),
        )
        self._tracks[tid] = trk
        det.track_id = tid

    def _update_track(self, trk: TrackedObject, det, frame_id: int) -> None:
        prev_center = list(trk.center)
        prev_vel = list(trk.velocity)
        prev_state = trk.state

        new_cx = float(det.center[0])
        new_cy = float(det.center[1])
        x1, y1, x2, y2 = det.bbox
        new_bcx = float((x1 + x2) / 2.0)
        new_bcy = float(y2)

        raw_vx = new_cx - prev_center[0]
        raw_vy = new_cy - prev_center[1]

        # Multi-frame velocity smoothing
        alpha = self.velocity_alpha
        vx = alpha * raw_vx + (1 - alpha) * prev_vel[0]
        vy = alpha * raw_vy + (1 - alpha) * prev_vel[1]

        ax = vx - prev_vel[0]
        ay = vy - prev_vel[1]

        trk.center = [new_cx, new_cy]
        trk.bottom_center = [new_bcx, new_bcy]
        trk.bbox = list(det.bbox)
        trk.confidence = det.confidence
        trk.area = float(det.area)
        trk.velocity = [vx, vy]
        trk.acceleration = [ax, ay]
        trk.age += 1
        trk.last_seen = frame_id
        trk.is_stable = (trk.age >= self.min_track_age)

        # Transition track state
        if prev_state == STATE_TEMPORARILY_LOST:
            trk.state = STATE_REACQUIRED
        elif trk.is_stable:
            trk.state = STATE_CONFIRMED
        else:
            trk.state = STATE_TENTATIVE

        trk.disappeared = 0
        trk.missed_frames = 0
        trk.last_update_timestamp = time.time()

        # Update histories
        trk.center_history.append([new_cx, new_cy])
        trk.bottom_center_history.append([new_bcx, new_bcy])
        trk.area_history.append(float(det.area))

        if len(trk.center_history) > self.history_window:
            trk.center_history.pop(0)
        if len(trk.bottom_center_history) > self.history_window:
            trk.bottom_center_history.pop(0)
        if len(trk.area_history) > self.history_window:
            trk.area_history.pop(0)

        # Classify motion pattern
        trk.motion_state = self._classify_motion(trk)
        det.track_id = trk.track_id

    def _classify_motion(self, trk: TrackedObject) -> str:
        """
        Classify kinematic pattern: stationary, approaching, receding, crossing, parallel, uncertain.
        """
        if trk.age < 2 or len(trk.center_history) < 2:
            return MOTION_UNCERTAIN

        vx, vy = trk.velocity
        speed = math.hypot(vx, vy)

        if speed < self.stationary_thresh:
            return MOTION_STATIONARY

        abs_vx = abs(vx)
        abs_vy = abs(vy)

        # Laterally crossing test: |vx| dominates |vy|
        if abs_vx > self.crossing_ratio_thresh * max(abs_vy, 0.5) and abs_vx >= 1.5:
            return MOTION_CROSSING

        # Approaching test: area expanding or moving downward toward camera
        area_rate = 0.0
        if len(trk.area_history) >= 2 and trk.area_history[-2] > 0:
            area_rate = (trk.area_history[-1] - trk.area_history[-2]) / trk.area_history[-2]

        if area_rate > self.approaching_area_rate_thresh or vy > 2.0:
            return MOTION_APPROACHING

        if vy < -2.0:
            return MOTION_RECEDING

        return MOTION_PARALLEL

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH

        areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        unionArea = areaA + areaB - interArea

        return interArea / unionArea if unionArea > 0 else 0.0

    def to_dict_list(self, tracks: Optional[List] = None) -> List[dict]:
        """Serialize tracked objects for logging."""
        if tracks is None:
            tracks = list(self._tracks.values())
        return [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "bbox": [round(v, 1) for v in t.bbox],
                "center": [round(v, 1) for v in t.center],
                "velocity": [round(v, 2) for v in t.velocity],
                "motion_state": getattr(t, "motion_state", "uncertain"),
                "state": getattr(t, "state", "CONFIRMED"),
                "age": t.age,
                "is_stable": t.is_stable,
            }
            for t in tracks
        ]
