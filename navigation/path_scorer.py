"""
ORCA EYE — Stage 7: Path Scorer  (3D-aware)
============================================
Implements the weighted scoring function:

  Score(P_i) = w_c·C + w_f·F + w_p·P + w_depth·Depth - w_r·R - w_u·U - w_d·DC

  C     = obstacle clearance (min free_prob)
  F     = free-space availability
  P     = forward progress
  Depth = depth clearance (0=near/blocked, 1=far/open) — 3D awareness
  R     = estimated collision risk
  U     = uncertainty penalty
  DC    = dynamic conflict penalty

All weights are read from config.yaml. Nothing is hard-coded here.

Inputs  : List[PathCandidate], config
Outputs : List[PathCandidate] with .score populated, sorted descending
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class PathScorer:
    """
    Scores each PathCandidate using the weighted linear formula.

    The STOP candidate receives a fixed base score and is always kept
    as a safe fallback regardless of other scores.

    Configuration keys (from cfg['scoring']):
      w_c              : clearance weight
      w_f              : free-space weight
      w_p              : progress weight
      w_r              : risk penalty weight
      w_u              : uncertainty penalty weight
      stop_base_score  : fixed score for the STOP candidate
    """

    def __init__(self, cfg: dict) -> None:
        self.w_c:     float = cfg.get("w_c",     0.28)
        self.w_f:     float = cfg.get("w_f",     0.22)
        self.w_p:     float = cfg.get("w_p",     0.22)
        self.w_r:     float = cfg.get("w_r",     0.15)
        self.w_u:     float = cfg.get("w_u",     0.05)
        self.w_d:     float = cfg.get("w_d",     0.18)
        self.w_depth: float = cfg.get("w_depth", 0.15)   # 3D depth clearance reward
        self.stop_base_score: float = cfg.get("stop_base_score", 0.10)

        logger.info(
            "PathScorer weights — clearance:%.2f free:%.2f progress:%.2f "
            "risk:%.2f uncertainty:%.2f dyn_conflict:%.2f depth:%.2f stop:%.2f",
            self.w_c, self.w_f, self.w_p, self.w_r, self.w_u,
            self.w_d, self.w_depth, self.stop_base_score,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, candidates: List) -> List:
        """
        Score all candidates and return them sorted by score descending.

        Parameters
        ----------
        candidates : List[PathCandidate]

        Returns
        -------
        List[PathCandidate] — sorted best-first, scores populated
        """
        for candidate in candidates:
            if candidate.is_stop:
                candidate.score = self.stop_base_score
                candidate.total_score = self.stop_base_score
            else:
                candidate.score = self._compute_score(candidate)
                candidate.total_score = candidate.score

        # Sort: best score first
        candidates.sort(key=lambda c: c.score, reverse=True)

        logger.debug(
            "Path scores: %s",
            [(c.direction, round(c.score, 3)) for c in candidates]
        )
        return candidates

    def score_one(self, candidate) -> float:
        """Score a single PathCandidate and return its score."""
        if candidate.is_stop:
            candidate.score = self.stop_base_score
            candidate.total_score = self.stop_base_score
            return self.stop_base_score
        score = self._compute_score(candidate)
        candidate.score = score
        candidate.total_score = score
        return score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, p) -> float:
        """
        Weighted scoring formula including 3D depth clearance and dynamic conflict.

        Score(P) = w_c·C + w_f·F + w_p·P + w_depth·Depth - w_r·R - w_u·U - w_d·D
        """
        C = float(p.clearance if p.clearance > 0.0 else p.min_clearance)
        F = float(p.free_space_score if p.free_space_score > 0.0 else (1.0 - p.risk))
        P = float(p.progress) * (1.0 - 0.3 * p.curvature)
        R = float(p.risk)
        U = float(p.uncertainty)
        D = float(getattr(p, "dynamic_conflict_score", 0.0))
        # 3D depth clearance: min depth along path (0=near/blocked, 1=far/open)
        Depth = float(getattr(p, "depth_clearance", 0.5))
        # Blend with 3D metric corridor clearance if available (rewards open corridors)
        c3d = float(getattr(p, "corridor_3d_clearance_m", 0.0))
        if c3d > 0.0:
            Depth = 0.35 * Depth + 0.65 * min(1.0, c3d / 5.0)

        score = (
            self.w_c     * C
            + self.w_f   * F
            + self.w_p   * P
            + self.w_depth * Depth     # reward paths that go through open 3D space
            - self.w_r   * R
            - self.w_u   * U
            - self.w_d   * D
        )

        score = max(-0.1, min(1.0, round(score, 6)))
        p.score = score
        p.total_score = score
        return score

    def to_dict_list(self, candidates: List) -> List[dict]:
        """Serialize scored candidates for logging."""
        return [
            {
                "direction":       c.direction,
                "angle_deg":       c.angle_deg,
                "score":           round(c.score, 4),
                "clearance":       round(c.clearance, 4),
                "depth_clearance": round(getattr(c, "depth_clearance", 0.0), 4),
                "dynamic_conflict": round(getattr(c, "dynamic_conflict_score", 0.0), 3),
                "progress":        round(c.progress, 4),
                "risk":            round(c.risk, 4),
                "uncertainty":     round(c.uncertainty, 4),
                "is_stop":         c.is_stop,
            }
            for c in candidates
        ]
