"""
ORCA EYE — Stage 11: Failure Detection Subsystem
==================================================
Automatically flags frames exhibiting failure modes and saves
JPEG snapshots + JSON sidecars to failure_cases/.

This is NOT optional. The research contribution will emerge from these.

Failure conditions checked every frame:
  1.  No valid path exists
  2.  Rapid path change (direction changed in < rapid_change_window frames)
  3.  Selected path has low clearance
  4.  Detection confidence dropped below threshold
  5.  Free-space estimation unstable (high uncertainty)
  6.  Depth inconsistency (large frame-to-frame change)
  7.  Sudden object appearance (new track, not previously seen)
  8.  Object disappearance (track dropped)
  9.  Repeated direction changes (oscillation)
  10. Left/right decision oscillation
  11. Unknown regions dominate the scene
  12. Perception/planner disagreement

Each failure record contains:
  - failure_codes   : list of triggered conditions
  - frame snapshot  : JPEG
  - full state JSON : all pipeline values at time of failure
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure Codes
# ---------------------------------------------------------------------------

class FailureCode:
    NO_VALID_PATH = "NO_VALID_PATH"
    RAPID_PATH_CHANGE = "RAPID_PATH_CHANGE"
    LOW_CLEARANCE = "LOW_CLEARANCE"
    LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"
    FREESPACE_UNSTABLE = "FREESPACE_UNSTABLE"
    DEPTH_INCONSISTENCY = "DEPTH_INCONSISTENCY"
    SUDDEN_APPEARANCE = "SUDDEN_APPEARANCE"
    SUDDEN_DISAPPEARANCE = "SUDDEN_DISAPPEARANCE"
    DIRECTION_OSCILLATION = "DIRECTION_OSCILLATION"
    LR_OSCILLATION = "LR_OSCILLATION"
    UNKNOWN_DOMINANCE = "UNKNOWN_DOMINANCE"
    PERCEPTION_PLANNER_DISAGREEMENT = "PERCEPTION_PLANNER_DISAGREEMENT"


@dataclass
class FailureRecord:
    """Complete failure event record."""
    frame_id: int
    timestamp: float
    failure_codes: List[str]
    severity: str           # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    # State snapshot
    command: str = ""
    score: float = 0.0
    clearance: float = 0.0
    uncertainty: float = 0.0
    num_detections: int = 0
    num_tracks: int = 0
    freespace_stats: Dict = field(default_factory=dict)
    depth_stats: Dict = field(default_factory=dict)
    stability_metrics: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Failure Detector
# ---------------------------------------------------------------------------

class FailureDetector:
    """
    Inspects per-frame pipeline state and flags failure conditions.

    Configuration keys (from cfg['failure_detection']):
      low_clearance_threshold
      low_detection_confidence
      high_uncertainty_threshold
      depth_inconsistency_threshold
      rapid_change_window
      unknown_dominance_threshold
      save_snapshots
    """

    def __init__(self, cfg: dict) -> None:
        fd_cfg = cfg.get("failure_detection", {})
        self.low_clearance_thresh: float = fd_cfg.get("low_clearance_threshold", 0.15)
        self.low_conf_thresh: float = fd_cfg.get("low_detection_confidence", 0.30)
        self.high_unc_thresh: float = fd_cfg.get("high_uncertainty_threshold", 0.60)
        self.depth_incons_thresh: float = fd_cfg.get("depth_inconsistency_threshold", 0.25)
        self.rapid_change_window: int = fd_cfg.get("rapid_change_window", 5)
        self.unknown_dom_thresh: float = fd_cfg.get("unknown_dominance_threshold", 0.55)
        self.save_snapshots: bool = fd_cfg.get("save_snapshots", True)

        self.failure_dir = Path(cfg.get("logging", {}).get("failure_dir", "failure_cases"))
        self.failure_dir.mkdir(parents=True, exist_ok=True)

        # State history
        self._direction_history: Deque[str] = deque(maxlen=20)
        self._track_ids_prev: Set[int] = set()
        self._prev_depth_mean: Optional[float] = None
        self._prev_command: Optional[str] = None

        # Counters
        self.total_failures: int = 0
        _all_codes = [
            FailureCode.NO_VALID_PATH,
            FailureCode.RAPID_PATH_CHANGE,
            FailureCode.LOW_CLEARANCE,
            FailureCode.LOW_DETECTION_CONFIDENCE,
            FailureCode.FREESPACE_UNSTABLE,
            FailureCode.DEPTH_INCONSISTENCY,
            FailureCode.SUDDEN_APPEARANCE,
            FailureCode.SUDDEN_DISAPPEARANCE,
            FailureCode.DIRECTION_OSCILLATION,
            FailureCode.LR_OSCILLATION,
            FailureCode.UNKNOWN_DOMINANCE,
            FailureCode.PERCEPTION_PLANNER_DISAGREEMENT,
        ]
        self.failures_by_code: Dict[str, int] = {code: 0 for code in _all_codes}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        frame_id: int,
        frame: np.ndarray,
        decision,
        detection_result,
        tracks: List,
        freespace_result,
        depth_result,
        stability_metrics: Dict,
        candidates: List,
    ) -> Optional[FailureRecord]:
        """
        Check all failure conditions for the current frame.

        Returns a FailureRecord if any condition is triggered, else None.
        Also saves snapshot to disk if configured.
        """
        codes: List[str] = []

        # --- 1. No valid path
        if decision.command in ("STOP",) and decision.reason != "STOP is best":
            # STOP due to no safe path (not deliberate)
            if "no" in decision.reason.lower() or "score" in decision.reason.lower() \
               or "clearance" in decision.reason.lower():
                codes.append(FailureCode.NO_VALID_PATH)

        # --- 2. Rapid path change
        cmd = decision.command
        self._direction_history.append(cmd)
        hist = list(self._direction_history)
        if len(hist) >= self.rapid_change_window:
            recent = hist[-self.rapid_change_window:]
            changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
            if changes >= self.rapid_change_window - 1:
                codes.append(FailureCode.RAPID_PATH_CHANGE)

        # --- 3. Low clearance
        if decision.clearance < self.low_clearance_thresh and not decision.is_stop:
            codes.append(FailureCode.LOW_CLEARANCE)

        # --- 4. Low detection confidence
        if (detection_result.is_low_confidence and
                detection_result.num_detections > 0 and
                detection_result.mean_confidence < self.low_conf_thresh):
            codes.append(FailureCode.LOW_DETECTION_CONFIDENCE)

        # --- 5. Free-space unstable
        if freespace_result and freespace_result.uncertainty > self.high_unc_thresh:
            codes.append(FailureCode.FREESPACE_UNSTABLE)

        # --- 6. Depth inconsistency
        if depth_result and depth_result.is_valid:
            curr_depth_mean = depth_result.statistics.get("mean", 0.5)
            if self._prev_depth_mean is not None:
                delta = abs(curr_depth_mean - self._prev_depth_mean)
                if delta > self.depth_incons_thresh:
                    codes.append(FailureCode.DEPTH_INCONSISTENCY)
            self._prev_depth_mean = curr_depth_mean

        # --- 7. Sudden appearance
        current_track_ids = {t.track_id for t in tracks}
        new_ids = current_track_ids - self._track_ids_prev
        appeared = [t for t in tracks if t.track_id in new_ids and t.age == 1]
        if appeared:
            codes.append(FailureCode.SUDDEN_APPEARANCE)

        # --- 8. Sudden disappearance
        disappeared_ids = self._track_ids_prev - current_track_ids
        if disappeared_ids:
            codes.append(FailureCode.SUDDEN_DISAPPEARANCE)

        self._track_ids_prev = current_track_ids

        # --- 9. Direction oscillation (from stability metrics)
        if stability_metrics.get("is_oscillating", False):
            codes.append(FailureCode.DIRECTION_OSCILLATION)

        # --- 10. Left/Right oscillation specifically
        if len(hist) >= 4:
            lr_hist = [c for c in hist[-6:] if c in ("LEFT", "RIGHT", "SLIGHT_LEFT", "SLIGHT_RIGHT")]
            if len(lr_hist) >= 4:
                lefts = sum(1 for c in lr_hist if "LEFT" in c)
                rights = sum(1 for c in lr_hist if "RIGHT" in c)
                if lefts >= 2 and rights >= 2:
                    codes.append(FailureCode.LR_OSCILLATION)

        # --- 11. Unknown region dominance
        if freespace_result:
            unk_frac = freespace_result.statistics.get("unknown_fraction", 0.0)
            if unk_frac > self.unknown_dom_thresh:
                codes.append(FailureCode.UNKNOWN_DOMINANCE)

        # --- 12. Perception/planner disagreement
        # If free-space says lots of FREE but planner says STOP
        if freespace_result and decision.is_stop:
            free_frac = freespace_result.statistics.get("free_fraction", 0.0)
            if free_frac > 0.5:  # lots of free space but planner stopped
                codes.append(FailureCode.PERCEPTION_PLANNER_DISAGREEMENT)

        if not codes:
            return None

        # --- Determine severity
        severity = self._compute_severity(codes)

        record = FailureRecord(
            frame_id=frame_id,
            timestamp=time.time(),
            failure_codes=codes,
            severity=severity,
            command=decision.command,
            score=decision.score,
            clearance=decision.clearance,
            uncertainty=decision.uncertainty,
            num_detections=detection_result.num_detections,
            num_tracks=len(tracks),
            freespace_stats=freespace_result.statistics if freespace_result else {},
            depth_stats=depth_result.statistics if depth_result else {},
            stability_metrics=stability_metrics,
        )

        self.total_failures += 1
        for code in codes:
            self.failures_by_code[code] = self.failures_by_code.get(code, 0) + 1

        if self.save_snapshots:
            self._save_snapshot(frame_id, frame, record)

        logger.warning(
            "Frame %d: failures=%s severity=%s", frame_id, codes, severity
        )
        return record

    def get_summary(self) -> Dict:
        """Return aggregate failure statistics."""
        return {
            "total_failures": self.total_failures,
            "failures_by_code": dict(self.failures_by_code),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_severity(self, codes: List[str]) -> str:
        if FailureCode.NO_VALID_PATH in codes or FailureCode.DIRECTION_OSCILLATION in codes:
            return "CRITICAL"
        if len(codes) >= 3:
            return "HIGH"
        if len(codes) == 2:
            return "MEDIUM"
        return "LOW"

    def _save_snapshot(
        self,
        frame_id: int,
        frame: np.ndarray,
        record: FailureRecord,
    ) -> None:
        """Save JPEG frame + JSON sidecar to failure_cases/."""
        base_name = f"frame_{frame_id:06d}_{record.severity}"
        jpg_path = self.failure_dir / f"{base_name}.jpg"
        json_path = self.failure_dir / f"{base_name}.json"

        try:
            cv2.imwrite(str(jpg_path), frame)
        except Exception as exc:
            logger.warning("Failed to save failure snapshot: %s", exc)
            return

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "frame_id": record.frame_id,
                    "timestamp": record.timestamp,
                    "failure_codes": record.failure_codes,
                    "severity": record.severity,
                    "command": record.command,
                    "score": record.score,
                    "clearance": record.clearance,
                    "uncertainty": record.uncertainty,
                    "num_detections": record.num_detections,
                    "num_tracks": record.num_tracks,
                    "freespace_stats": record.freespace_stats,
                    "depth_stats": record.depth_stats,
                    "stability_metrics": record.stability_metrics,
                }, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save failure JSON: %s", exc)
