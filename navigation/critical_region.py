"""
ORCA EYE — Stage A: Navigation-Critical Region & Critical Entity Set (E_k)
========================================================================
Defines the spatial-temporal safety corridor envelope S(G_k) for each candidate
maneuver G_k across the execution horizon H in [1.5s, 2.5s].

Identifies the critical entity set E_k:
  E_k = { e in Entities : predicted footprint of e intersects S(G_k) during [t, t + H] }

This couples perception directly to navigation: an entity is only critical if
it poses an observational or physical hazard to the specific maneuver G_k.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from navigation.threat_model import ThreatLevel, ThreatModel, ThreatRecord


@dataclass
class CriticalRegion:
    """
    Representation of the safety envelope S(G_k) and critical entity set E_k
    for candidate corridor G_k.
    """
    corridor_direction: str
    corridor_angle_deg: float
    critical_entities: List[int] = field(default_factory=list)  # Entity IDs in E_k
    threat_records: Dict[int, ThreatRecord] = field(default_factory=dict)
    max_threat_level: ThreatLevel = ThreatLevel.NONE
    min_ttc: Optional[float] = None
    reason: str = "Clear corridor"


class CriticalRegionExtractor:
    """
    Extracts critical entities E_k for all candidate corridors.
    """

    def __init__(
        self,
        corridor_width_m: float = 1.0,
        horizon_s: float = 2.0,
        user_walk_speed_m_s: float = 1.0,
        d_safe: float = 0.8,
    ) -> None:
        self.corridor_width_m = corridor_width_m
        self.horizon_s = horizon_s
        self.user_walk_speed = user_walk_speed_m_s
        self.d_safe = d_safe
        self.threat_model = ThreatModel(
            d_safe=d_safe,
            horizon_s=horizon_s,
            user_walk_speed_m_s=user_walk_speed_m_s,
        )

    def extract_critical_regions(
        self,
        candidates: List,  # List[PathCandidate]
        spatial_entities: List,  # List[SpatialEntity]
    ) -> Dict[str, CriticalRegion]:
        """
        Extract CriticalRegion for each candidate path corridor.

        Returns
        -------
        Dict[direction_str, CriticalRegion]
        """
        results: Dict[str, CriticalRegion] = {}

        for cand in candidates:
            direction = cand.direction
            angle_deg = getattr(cand, "angle_deg", 0.0)

            if cand.is_stop:
                # STOP candidate critical region: immediate proximity only (radius d_safe)
                crit = self._extract_stop_critical_region(spatial_entities)
                results[direction] = crit
                continue

            crit_entities: List[int] = []
            threat_records: Dict[int, ThreatRecord] = {}
            max_threat = ThreatLevel.NONE
            min_ttc: Optional[float] = None
            reasons: List[str] = []

            for entity in spatial_entities:
                record = self.threat_model.evaluate_entity_corridor(
                    entity=entity,
                    corridor_direction=direction,
                    corridor_angle_deg=angle_deg,
                )

                # Entity is admitted to E_k if it has a trajectory conflict or corridor overlap
                is_critical = False
                if record.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.MEDIUM):
                    is_critical = True
                elif record.corridor_overlap_m > 0.15:
                    is_critical = True

                if is_critical:
                    track_id = record.entity_id
                    crit_entities.append(track_id)
                    threat_records[track_id] = record

                    if record.ttc_seconds is not None:
                        if min_ttc is None or record.ttc_seconds < min_ttc:
                            min_ttc = record.ttc_seconds

                    # Update max threat level
                    if record.threat_level == ThreatLevel.CRITICAL:
                        max_threat = ThreatLevel.CRITICAL
                    elif record.threat_level == ThreatLevel.MEDIUM and max_threat != ThreatLevel.CRITICAL:
                        max_threat = ThreatLevel.MEDIUM
                    elif record.threat_level == ThreatLevel.LOW and max_threat == ThreatLevel.NONE:
                        max_threat = ThreatLevel.LOW

                    reasons.append(f"ID {track_id}: {record.reason}")

            reason_str = "; ".join(reasons) if reasons else "No critical entities in envelope"

            results[direction] = CriticalRegion(
                corridor_direction=direction,
                corridor_angle_deg=angle_deg,
                critical_entities=crit_entities,
                threat_records=threat_records,
                max_threat_level=max_threat,
                min_ttc=min_ttc,
                reason=reason_str,
            )

        return results

    def _extract_stop_critical_region(self, spatial_entities: List) -> CriticalRegion:
        """For STOP, only entities directly colliding within 1.0m are critical."""
        crit_entities: List[int] = []
        threat_records: Dict[int, ThreatRecord] = {}
        max_threat = ThreatLevel.NONE

        for entity in spatial_entities:
            dist_norm = float(getattr(entity, "relative_distance", 0.5))
            dist_m = dist_norm * 6.0
            if dist_m <= 1.2:
                t_id = int(getattr(entity, "track_id", 0))
                rec = ThreatRecord(
                    entity_id=t_id,
                    corridor_direction="STOP",
                    threat_level=ThreatLevel.CRITICAL if dist_m < 0.8 else ThreatLevel.MEDIUM,
                    ttc_seconds=0.0 if dist_m < 0.8 else 1.0,
                    closest_approach_m=dist_m,
                    confidence=float(getattr(entity, "tracking_confidence", 1.0)),
                    reason="Entity in immediate proximity bubble",
                )
                crit_entities.append(t_id)
                threat_records[t_id] = rec
                max_threat = ThreatLevel.CRITICAL

        return CriticalRegion(
            corridor_direction="STOP",
            corridor_angle_deg=0.0,
            critical_entities=crit_entities,
            threat_records=threat_records,
            max_threat_level=max_threat,
            reason="Stop safety boundary",
        )
