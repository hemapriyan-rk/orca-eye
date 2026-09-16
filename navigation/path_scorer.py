"""
ORCA EYE — Stage 7: Path Scorer
=================================
Implements the transparent baseline scoring function:

  Score(P_i) = w_c·C_i + w_f·F_i + w_p·P_i - w_r·R_i - w_u·U_i

  C_i = obstacle clearance
  F_i = free-space availability
  P_i = forward progress
  R_i = estimated collision risk
  U_i = uncertainty penalty

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
        self.w_c: float = cfg.get("w_c", 0.30)
        self.w_f: float = cfg.get("w_f", 0.25)
        self.w_p: float = cfg.get("w_p", 0.25)
        self.w_r: float = cfg.get("w_r", 0.15)
        self.w_u: float = cfg.get("w_u", 0.05)
        self.w_d: float = cfg.get("w_d", 0.20)
        self.stop_base_score: float = cfg.get("stop_base_score", 0.10)

        logger.info(
            "PathScorer weights — clearance:%.2f free:%.2f progress:%.2f "
            "risk:%.2f uncertainty:%.2f dyn_conflict:%.2f stop:%.2f",
            self.w_c, self.w_f, self.w_p, self.w_r, self.w_u, self.w_d, self.stop_base_score
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
        Weighted scoring formula including dynamic conflict penalty.

        Score(P_i) = w_c·C + w_f·F + w_p·P - w_r·R - w_u·U - w_d·D
        """
        # Clearance: min free_prob along path
        C = float(p.clearance if p.clearance > 0.0 else p.min_clearance)

        # Free-space: free_space_score or 1 - risk
        F = float(p.free_space_score if p.free_space_score > 0.0 else (1.0 - p.risk))

        # Progress: fraction of planned depth reached, scaled down by curvature
        P = float(p.progress) * (1.0 - 0.3 * p.curvature)

        # Risk: mean occupancy along path (collision probability proxy)
        R = float(p.risk)

        # Uncertainty: mean epistemic uncertainty
        U = float(p.uncertainty)

        # Dynamic conflict: prospective collision with moving agents
        D = float(getattr(p, "dynamic_conflict_score", 0.0))

        score = (
            self.w_c * C
            + self.w_f * F
            + self.w_p * P
            - self.w_r * R
            - self.w_u * U
            - self.w_d * D
        )

        # Clamp to [-0.1, 1.0] — can go slightly negative in worst case
        score = max(-0.1, min(1.0, score))
        score = round(score, 6)
        p.score = score
        p.total_score = score
        return score

    def to_dict_list(self, candidates: List) -> List[dict]:
        """Serialize scored candidates for logging."""
        return [
            {
                "direction": c.direction,
                "angle_deg": c.angle_deg,
                "score": round(c.score, 4),
                "clearance": round(c.clearance, 4),
                "dynamic_conflict": round(getattr(c, "dynamic_conflict_score", 0.0), 3),
                "progress": round(c.progress, 4),
                "risk": round(c.risk, 4),
                "uncertainty": round(c.uncertainty, 4),
                "is_stop": c.is_stop,
            }
            for c in candidates
        ]
