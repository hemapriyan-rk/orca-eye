"""
ORCA EYE — Stage A: Perceptual-Support Gate & Corridor Admissibility
===================================================================
Evaluates whether candidate corridors possess sufficient perceptual support
to be safely navigated by the user:

  Support(c, k) = 1.0                      if E_k is empty
                  min_{e in E_k} rho_FOV(c, e, t, H)  otherwise

Corridor Admissibility Gate:
  Admissible(k) <=> SafetyScore(k) >= tau_safe  and  Support(c_primary, k) >= tau_cam

Defaults:
  tau_safe = 0.35
  tau_cam  = 0.40

If a corridor is physically open but perception cannot maintain observation
of the critical entities needed to verify safe traversal, the corridor is
declared INADMISSIBLE.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from navigation.critical_region import CriticalRegion
from navigation.rho_fov import RhoFOVPrediction


@dataclass
class CorridorSupportRecord:
    """
    Perceptual support assessment and admissibility verdict for a corridor.
    """
    corridor_direction: str
    safety_score: float
    perceptual_support: float
    is_admissible: bool
    status_label: str                                   # "ADMISSIBLE" | "INADMISSIBLE_LOW_SAFETY" | "INADMISSIBLE_LOW_SUPPORT"
    critical_entities: List[int] = field(default_factory=list)
    entity_rho_fov: Dict[int, float] = field(default_factory=dict)
    reason: str = ""


class PerceptualSupportGate:
    """
    Computes corridor-level perceptual support and enforces the admissibility gate.
    """

    def __init__(
        self,
        tau_safe: float = 0.35,
        tau_cam: float = 0.40,
    ) -> None:
        self.tau_safe = tau_safe
        self.tau_cam = tau_cam

    def evaluate_corridors(
        self,
        candidates: List,  # List[PathCandidate]
        critical_regions: Dict[str, CriticalRegion],
        rho_predictions: Dict[int, RhoFOVPrediction],  # entity_id -> RhoFOVPrediction
    ) -> Dict[str, CorridorSupportRecord]:
        """
        Evaluate admissibility for each candidate corridor based on safety and perceptual support.
        """
        results: Dict[str, CorridorSupportRecord] = {}

        for cand in candidates:
            direction = cand.direction
            # Candidate safety score (e.g. from clearance or score)
            safety = float(getattr(cand, "total_score", getattr(cand, "score", cand.clearance)))

            # STOP is a safety maneuver; perceptual support is intrinsically 1.0
            if cand.is_stop:
                results[direction] = CorridorSupportRecord(
                    corridor_direction=direction,
                    safety_score=1.0,
                    perceptual_support=1.0,
                    is_admissible=True,
                    status_label="ADMISSIBLE",
                    critical_entities=[],
                    entity_rho_fov={},
                    reason="Stop fallback is always admissible",
                )
                continue

            crit_region = critical_regions.get(direction)
            crit_entity_ids = crit_region.critical_entities if crit_region else []

            entity_rhos: Dict[int, float] = {}
            if not crit_entity_ids:
                # E_k is empty -> full support
                support_score = 1.0
            else:
                rhos = []
                for eid in crit_entity_ids:
                    pred = rho_predictions.get(eid)
                    rho_val = pred.rho_fov if pred else 0.50
                    entity_rhos[eid] = rho_val
                    rhos.append(rho_val)
                support_score = float(min(rhos)) if rhos else 1.0

            # Evaluate Admissibility Gate
            passed_safety = safety >= self.tau_safe
            passed_support = support_score >= self.tau_cam

            if passed_safety and passed_support:
                is_admissible = True
                status_label = "ADMISSIBLE"
                reason = f"Safe (score={safety:.2f}) and perceptually supported (rho={support_score:.2f})"
            elif not passed_safety:
                is_admissible = False
                status_label = "INADMISSIBLE_LOW_SAFETY"
                reason = f"Physical clearance below safety threshold ({safety:.2f} < {self.tau_safe:.2f})"
            else:
                is_admissible = False
                status_label = "INADMISSIBLE_LOW_SUPPORT"
                reason = f"Perceptual observation of critical entities expiring ({support_score:.2f} < {self.tau_cam:.2f})"

            results[direction] = CorridorSupportRecord(
                corridor_direction=direction,
                safety_score=round(safety, 3),
                perceptual_support=round(support_score, 3),
                is_admissible=is_admissible,
                status_label=status_label,
                critical_entities=crit_entity_ids,
                entity_rho_fov=entity_rhos,
                reason=reason,
            )

        return results
