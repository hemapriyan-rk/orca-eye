"""
ORCA EYE — Stage 8: Navigation Decision Maker
===============================================
Selects the safest candidate path and emits a navigation command.

Safety rules:
  - If best non-STOP score < min_go_score → STOP
  - If best path clearance < min_clearance → STOP
  - If uncertainty > max_uncertainty → CAUTION
  - If imminent dynamic collision (TTC < threshold) → STOP or CAUTION
  - Hysteresis: current direction requires beating by margin to switch

Also tracks stability metrics:
  N_switch  = direction changes per unit time
  T_stable  = average duration of a stable direction

Inputs  : List[PathCandidate] (scored, sorted), spatial_entities, dynamic_conflicts
Outputs : NavigationState (formerly NavigationDecision)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardized Reason Codes
# ---------------------------------------------------------------------------

class NavigationReason:
    CLEAR_PATH = "CLEAR_PATH"
    STATIC_OBSTACLE = "STATIC_OBSTACLE"
    DYNAMIC_OBSTACLE = "DYNAMIC_OBSTACLE"
    WALL_COLLISION = "WALL_COLLISION"
    WALL_PROXIMITY = "WALL_PROXIMITY"
    LOW_CLEARANCE = "LOW_CLEARANCE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    CORRIDOR_BLOCKED = "CORRIDOR_BLOCKED"
    LOW_PERCEPTUAL_SUPPORT = "LOW_PERCEPTUAL_SUPPORT"
    HYSTERESIS_HOLD = "HYSTERESIS_HOLD"
    NO_CANDIDATES = "NO_CANDIDATES"


# ---------------------------------------------------------------------------
# Data Contract — NavigationState (and NavigationDecision alias)
# ---------------------------------------------------------------------------

@dataclass
class NavigationState:
    """
    Canonical output of the decision module for a single frame.
    Exposes full situational context for logging, audio guidance, and failure analysis.
    """
    command: str                   # e.g. "STRAIGHT", "STOP", "CAUTION"
    selected_path_direction: str   # direction of the chosen path candidate
    score: float                   # score of selected path
    clearance: float               # clearance of selected path
    uncertainty: float             # uncertainty of selected path
    reason: str                    # human-readable explanation / reason code
    decision: str = "GO"           # "STOP" | "CAUTION" | "GO" | "SLOW" | "TURN"
    decision_confidence: float = 0.8
    nearest_obstacle_distance: float = 999.0
    minimum_path_clearance: float = 0.0
    dynamic_conflict: bool = False
    time_to_possible_conflict: Optional[float] = None
    free_space_confidence: float = 0.8
    spatial_uncertainty: float = 0.1
    active_track_ids: List[int] = field(default_factory=list)
    relevant_track_ids: List[int] = field(default_factory=list)
    recommended_direction: str = ""
    all_scores: List[dict] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)
    wall_warning: Optional[str] = None   # None | "LEFT" | "RIGHT" | "FRONTAL"
    is_stop: bool = False
    is_caution: bool = False
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    # Stage A: Single-Camera Scientific Validation & Perceptual Support Hierarchy
    critical_entities: List[int] = field(default_factory=list)      # E_k
    camera_support: float = 1.0                                      # Support(c_primary, k)
    rho_fov: dict = field(default_factory=dict)                      # entity_id -> rho_fov
    responsible_camera: str = "PRIMARY_CAM"
    responsibility_state: str = "PRIMARY"
    prediction_horizon: float = 2.0
    admissibility_status: str = "ADMISSIBLE"
    corridor_support_records: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.recommended_direction:
            self.recommended_direction = self.selected_path_direction or self.command
        if self.minimum_path_clearance == 0.0 and self.clearance != 0.0:
            self.minimum_path_clearance = self.clearance
        if not self.candidates and self.all_scores:
            self.candidates = self.all_scores
        if not self.decision or self.decision == "GO":
            if self.is_stop or self.command == "STOP":
                self.decision = "STOP"
            elif self.is_caution or self.command == "CAUTION" or self.wall_warning:
                self.decision = "CAUTION"
            elif self.command in ("LEFT", "RIGHT", "SLIGHT_LEFT", "SLIGHT_RIGHT"):
                self.decision = "TURN"
            else:
                self.decision = "GO"

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 4),
            "frame_id": self.frame_id,
            "command": self.command,
            "decision": self.decision,
            "recommended_direction": self.recommended_direction,
            "decision_confidence": round(self.decision_confidence, 4),
            "score": round(self.score, 4),
            "clearance": round(self.clearance, 4),
            "uncertainty": round(self.uncertainty, 4),
            "nearest_obstacle_distance": round(self.nearest_obstacle_distance, 3) if self.nearest_obstacle_distance < 900 else None,
            "minimum_path_clearance": round(self.minimum_path_clearance, 4),
            "dynamic_conflict": self.dynamic_conflict,
            "time_to_possible_conflict": round(self.time_to_possible_conflict, 2) if self.time_to_possible_conflict is not None else None,
            "wall_warning": self.wall_warning,
            "spatial_uncertainty": round(self.spatial_uncertainty, 4),
            "active_track_ids": self.active_track_ids,
            "relevant_track_ids": self.relevant_track_ids,
            "reason": self.reason,
            "is_stop": self.is_stop,
            "is_caution": self.is_caution,
            "all_scores": self.all_scores,
            # Stage A fields
            "critical_entities": self.critical_entities,
            "camera_support": round(self.camera_support, 4),
            "rho_fov": self.rho_fov,
            "responsible_camera": self.responsible_camera,
            "responsibility_state": self.responsibility_state,
            "prediction_horizon": self.prediction_horizon,
            "admissibility_status": self.admissibility_status,
            "corridor_support_records": self.corridor_support_records,
        }



# 100% backward compatibility alias
NavigationDecision = NavigationState


# ---------------------------------------------------------------------------
# Decision Maker
# ---------------------------------------------------------------------------

class DecisionMaker:
    """
    Selects the best safe path and emits navigation commands with hysteresis.

    Configuration keys (from cfg['safety']):
      min_go_score          : minimum score to issue GO (vs STOP)
      min_clearance         : minimum clearance to issue GO
      max_uncertainty       : maximum uncertainty before CAUTION
      direction_hysteresis  : margin required to switch direction
      stability_window      : frames used for N_switch / T_stable metrics
      max_direction_changes : changes per window before flagging oscillation
    """

    COMMAND_PRIORITY = {
        "STRAIGHT": 0,
        "SLIGHT_LEFT": 1,
        "SLIGHT_RIGHT": 1,
        "LEFT": 2,
        "RIGHT": 2,
        "SLOW": 3,
        "CAUTION": 4,
        "STOP": 5,
    }

    def __init__(self, cfg: dict) -> None:
        self.min_go_score: float = cfg.get("min_go_score", 0.35)
        self.min_clearance: float = cfg.get("min_clearance", 0.20)
        self.max_uncertainty: float = cfg.get("max_uncertainty", 0.65)
        self.hysteresis: float = cfg.get("direction_hysteresis", 0.08)
        self.stability_window: int = cfg.get("stability_window", 10)
        self.max_changes: int = cfg.get("max_direction_changes", 4)

        self._prev_decision: Optional[NavigationState] = None
        self._prev_command: str = "STRAIGHT"
        self._prev_score: float = 0.0
        self._decision_history: Deque[str] = deque(maxlen=self.stability_window)
        self._switch_times: Deque[float] = deque(maxlen=50)
        self._current_direction_start: float = time.time()

        # Metrics
        self.n_switch: int = 0          # total lifetime direction changes
        self.t_stable_history: List[float] = []  # durations of stable directions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        candidates: List,  # List[PathCandidate] sorted descending by score
        frame_id: int = 0,
        spatial_entities: Optional[List] = None,
        dynamic_conflicts: Optional[List] = None,
        wall_proximity: Optional[dict] = None,
        corridor_support: Optional[dict] = None,
        primary_camera_id: str = "PRIMARY_CAM",
    ) -> NavigationState:
        """
        Select the safest candidate and emit a navigation command.
        Enforces physical safety constraints and perceptual-support admissibility.

        Parameters
        ----------
        candidates       : List[PathCandidate] sorted descending by score
        frame_id         : int
        spatial_entities : Optional[List[SpatialEntity]]
        dynamic_conflicts: Optional[List[DynamicConflict]]
        wall_proximity   : Optional[dict]
        corridor_support : Optional[Dict[str, CorridorSupportRecord]]
        primary_camera_id: str

        Returns
        -------
        NavigationState
        """
        # Context extraction
        active_tracks = [
            e.track_id for e in (spatial_entities or [])
            if getattr(e, "track_id", -1) >= 0
        ]
        nearest_dist = min(
            [float(e.relative_distance) for e in (spatial_entities or [])
             if getattr(e, "relative_distance", None) is not None],
            default=999.0
        )

        has_conflict = False
        min_ttc: Optional[float] = None
        conflict_tracks: List[int] = []
        if dynamic_conflicts:
            for c in dynamic_conflicts:
                if getattr(c, "has_conflict", False):
                    has_conflict = True
                    tid = getattr(c, "track_id", -1)
                    if tid >= 0 and tid not in conflict_tracks:
                        conflict_tracks.append(tid)
                    ttc = getattr(c, "time_to_conflict", None)
                    if ttc is not None:
                        min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)

        if not candidates:
            return self._make_stop_decision(
                "No candidates generated",
                frame_id,
                reason_code=NavigationReason.NO_CANDIDATES,
                active_tracks=active_tracks,
                nearest_dist=nearest_dist,
            )

        # Separate STOP from directional candidates
        stop_candidate = next((c for c in candidates if c.is_stop), None)
        directional = [c for c in candidates if not c.is_stop]

        # --- Check overall scene uncertainty
        if directional:
            max_uncertainty = max(c.uncertainty for c in directional)
            if max_uncertainty > self.max_uncertainty:
                return self._make_caution_decision(
                    f"Scene uncertainty {max_uncertainty:.2f} > {self.max_uncertainty:.2f}",
                    max_uncertainty,
                    frame_id,
                    reason_code=NavigationReason.HIGH_UNCERTAINTY,
                    active_tracks=active_tracks,
                    nearest_dist=nearest_dist,
                )

        # --- Imminent wall / static barrier collision check (< 1.0m ahead)
        if wall_proximity and wall_proximity.get("is_frontal_collision", False):
            min_free = wall_proximity.get("min_frontal_free", 0.0)
            return self._make_stop_decision(
                f"Solid obstacle/wall directly ahead (<1.0m, clearance {min_free:.2f})",
                frame_id,
                reason_code=NavigationReason.WALL_COLLISION,
                active_tracks=active_tracks,
                conflict_tracks=conflict_tracks,
                nearest_dist=min(nearest_dist, 0.8),
            )

        # --- Imminent dynamic collision check on current heading
        if has_conflict and min_ttc is not None and min_ttc < 0.8:
            return self._make_stop_decision(
                f"Imminent collision detected with moving agent (TTC {min_ttc:.1f}s)",
                frame_id,
                reason_code=NavigationReason.DYNAMIC_OBSTACLE,
                active_tracks=active_tracks,
                conflict_tracks=conflict_tracks,
                nearest_dist=nearest_dist,
                dynamic_conflict=True,
                ttc=min_ttc,
            )

        # --- Filter candidates by Perceptual Support Admissibility Gate
        admissible_directional = []
        low_support_rejected = []
        if corridor_support:
            for c in directional:
                sup_rec = corridor_support.get(c.direction)
                if sup_rec is not None and not sup_rec.is_admissible and sup_rec.status_label == "INADMISSIBLE_LOW_SUPPORT":
                    low_support_rejected.append(c)
                else:
                    admissible_directional.append(c)
        else:
            admissible_directional = directional

        # --- Find best candidate (prioritizing perceptually admissible paths)
        best = admissible_directional[0] if admissible_directional else (directional[0] if directional else None)

        if best is None:
            return self._make_stop_decision(
                "No directional candidates",
                frame_id,
                reason_code=NavigationReason.NO_CANDIDATES,
                active_tracks=active_tracks,
                nearest_dist=nearest_dist,
            )

        # If best candidate is physically blocked
        if best.score < self.min_go_score:
            return self._make_stop_decision(
                f"Best score {best.score:.3f} < min_go {self.min_go_score:.3f}",
                frame_id,
                reason_code=NavigationReason.CORRIDOR_BLOCKED,
                active_tracks=active_tracks,
                nearest_dist=nearest_dist,
            )

        if best.clearance < self.min_clearance:
            return self._make_stop_decision(
                f"Best clearance {best.clearance:.3f} < min {self.min_clearance:.3f}",
                frame_id,
                reason_code=NavigationReason.LOW_CLEARANCE,
                active_tracks=active_tracks,
                nearest_dist=nearest_dist,
            )

        # If all directional candidates failed perceptual support, issue cautious stop/warning
        if not admissible_directional and low_support_rejected:
            return self._make_stop_decision(
                f"Perceptual observation of critical entities expiring across all corridors",
                frame_id,
                reason_code=NavigationReason.LOW_PERCEPTUAL_SUPPORT,
                active_tracks=active_tracks,
                nearest_dist=nearest_dist,
            )

        # --- Apply hysteresis: only switch if improvement exceeds margin
        hysteresis_held = False
        if self._prev_command not in ("STOP", "CAUTION") and self._prev_decision is not None:
            prev_dir_candidate = next(
                (c for c in admissible_directional if c.direction == self._prev_command),
                None
            )
            if prev_dir_candidate is not None:
                improvement = best.score - prev_dir_candidate.score
                if improvement < self.hysteresis and not best.is_stop:
                    best = prev_dir_candidate
                    hysteresis_held = True
                    logger.debug(
                        "Hysteresis: keeping %s (improvement %.3f < %.3f)",
                        self._prev_command, improvement, self.hysteresis
                    )

        # --- Build decision state
        all_scores = [
            {
                "direction": c.direction,
                "score": round(c.score, 4),
                "clearance": round(c.clearance, 4),
                "dynamic_conflict": round(getattr(c, "dynamic_conflict_score", 0.0), 3),
            }
            for c in candidates
        ]

        wall_warn = wall_proximity.get("lateral_warning") if wall_proximity else None

        # Determine decision category
        if has_conflict and min_ttc is not None and min_ttc < 1.8:
            decision_type = "SLOW"
        elif wall_warn:
            decision_type = "CAUTION"
        elif best.direction in ("LEFT", "RIGHT", "SLIGHT_LEFT", "SLIGHT_RIGHT"):
            decision_type = "TURN"
        else:
            decision_type = "GO"

        reason_tag = (
            f"[{NavigationReason.HYSTERESIS_HOLD}] " if hysteresis_held
            else f"[{NavigationReason.CLEAR_PATH}] "
        )
        wall_tag = f"[{NavigationReason.WALL_PROXIMITY}: {wall_warn}] " if wall_warn else ""
        reason_str = (
            f"{wall_tag}{reason_tag}Best scored path: {best.direction} "
            f"(score={best.score:.3f}, clearance={best.clearance:.3f})"
        )

        # Extract Stage A corridor support metadata
        best_support_rec = corridor_support.get(best.direction) if corridor_support else None
        crit_entities = best_support_rec.critical_entities if best_support_rec else []
        cam_support = best_support_rec.perceptual_support if best_support_rec else 1.0
        rho_dict = best_support_rec.entity_rho_fov if best_support_rec else {}
        admiss_status = best_support_rec.status_label if best_support_rec else "ADMISSIBLE"

        state = NavigationState(
            command=best.direction,
            selected_path_direction=best.direction,
            recommended_direction=best.direction,
            score=best.score,
            clearance=best.clearance,
            uncertainty=best.uncertainty,
            reason=reason_str,
            decision=decision_type,
            decision_confidence=round(max(0.1, min(1.0, best.score)), 4),
            nearest_obstacle_distance=nearest_dist,
            minimum_path_clearance=best.clearance,
            dynamic_conflict=has_conflict,
            time_to_possible_conflict=min_ttc,
            wall_warning=wall_warn,
            free_space_confidence=round(getattr(best, "free_space_score", 1.0 - best.risk), 4),
            spatial_uncertainty=best.uncertainty,
            active_track_ids=active_tracks,
            relevant_track_ids=conflict_tracks,
            all_scores=all_scores,
            candidates=all_scores,
            is_stop=False,
            is_caution=False,
            timestamp=time.time(),
            frame_id=frame_id,
            # Stage A fields
            critical_entities=crit_entities,
            camera_support=cam_support,
            rho_fov=rho_dict,
            responsible_camera=primary_camera_id,
            responsibility_state="PRIMARY",
            prediction_horizon=2.0,
            admissibility_status=admiss_status,
            corridor_support_records={
                d: {
                    "support": rec.perceptual_support,
                    "admissible": rec.is_admissible,
                    "status": rec.status_label,
                }
                for d, rec in (corridor_support or {}).items()
            },
        )

        self._update_stability(state)
        return state

    def get_stability_metrics(self) -> dict:
        """Return current stability metrics for logging."""
        window_changes = sum(
            1 for i in range(1, len(self._decision_history))
            if list(self._decision_history)[i] != list(self._decision_history)[i - 1]
        )
        t_stable_mean = (
            float(sum(self.t_stable_history) / len(self.t_stable_history))
            if self.t_stable_history else 0.0
        )
        return {
            "n_switch_total": self.n_switch,
            "n_switch_window": window_changes,
            "t_stable_mean_s": round(t_stable_mean, 3),
            "current_direction": self._prev_command,
            "is_oscillating": window_changes >= self.max_changes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_stop_decision(
        self,
        reason: str,
        frame_id: int,
        reason_code: str = NavigationReason.STATIC_OBSTACLE,
        active_tracks: Optional[List[int]] = None,
        conflict_tracks: Optional[List[int]] = None,
        nearest_dist: float = 999.0,
        dynamic_conflict: bool = False,
        ttc: Optional[float] = None,
    ) -> NavigationState:
        full_reason = f"[{reason_code}] {reason}"
        d = NavigationState(
            command="STOP",
            selected_path_direction="STOP",
            recommended_direction="STOP",
            score=0.0,
            clearance=0.0,
            uncertainty=1.0,
            reason=full_reason,
            decision="STOP",
            decision_confidence=1.0,
            nearest_obstacle_distance=nearest_dist,
            minimum_path_clearance=0.0,
            dynamic_conflict=dynamic_conflict,
            time_to_possible_conflict=ttc,
            active_track_ids=active_tracks or [],
            relevant_track_ids=conflict_tracks or [],
            is_stop=True,
            is_caution=False,
            timestamp=time.time(),
            frame_id=frame_id,
        )
        self._update_stability(d)
        logger.debug("STOP issued: %s", full_reason)
        return d

    def _make_caution_decision(
        self,
        reason: str,
        uncertainty: float,
        frame_id: int,
        reason_code: str = NavigationReason.HIGH_UNCERTAINTY,
        active_tracks: Optional[List[int]] = None,
        nearest_dist: float = 999.0,
    ) -> NavigationState:
        full_reason = f"[{reason_code}] {reason}"
        d = NavigationState(
            command="CAUTION",
            selected_path_direction="CAUTION",
            recommended_direction="CAUTION",
            score=0.0,
            clearance=0.0,
            uncertainty=uncertainty,
            reason=full_reason,
            decision="CAUTION",
            decision_confidence=0.5,
            nearest_obstacle_distance=nearest_dist,
            minimum_path_clearance=0.0,
            active_track_ids=active_tracks or [],
            is_stop=False,
            is_caution=True,
            timestamp=time.time(),
            frame_id=frame_id,
        )
        self._update_stability(d)
        logger.debug("CAUTION issued: %s", full_reason)
        return d

    def _update_stability(self, decision: NavigationState) -> None:
        """Update direction-change tracking and stability metrics."""
        cmd = decision.command
        now = time.time()

        if cmd != self._prev_command:
            duration = now - self._current_direction_start
            self.t_stable_history.append(duration)
            if len(self.t_stable_history) > 200:
                self.t_stable_history.pop(0)

            self.n_switch += 1
            self._switch_times.append(now)
            self._current_direction_start = now
            logger.debug(
                "Direction change: %s → %s (after %.2fs)",
                self._prev_command, cmd, duration
            )

        self._decision_history.append(cmd)
        self._prev_command = cmd
        self._prev_score = decision.score
        self._prev_decision = decision

    def to_dict(self, decision: NavigationState) -> dict:
        """Serialize a NavigationState for logging."""
        return decision.to_dict()
