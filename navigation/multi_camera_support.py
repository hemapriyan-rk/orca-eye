"""
ORCA EYE — Stage B: Multi-Camera Corridor Perceptual Support Gate
=================================================================
Extends single-camera perceptual support to multi-camera arrays:

  Stage A (Single Camera):
    Support_A(k) = min_{e in E_k} rho_FOV(PRIMARY, e, t, H)

  Stage B (Multi-Camera):
    Support_B(k) = min_{e in E_k} max_{c in C} rho_FOV(c, e, t, H)

Concept:
  Every critical entity in corridor envelope E_k must have at least ONE camera
  capable of maintaining sufficient future observation support over horizon H.

Admissibility Gate:
  Admissible(k) <=> Safety(k) >= tau_safe AND Support_B(k) >= tau_cam
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from navigation.camera_observation import MultiCameraEntityState
from navigation.camera_support import CorridorSupportRecord
from navigation.critical_region import CriticalRegion
from navigation.path_generator import PathCandidate


@dataclass
class MultiCameraCorridorSupportRecord:
    """Multi-camera perceptual support verdict for a single candidate corridor."""
    corridor_direction: str
    safety_score: float
    multi_cam_support: float
    single_cam_support: float
    is_admissible: bool
    status_label: str                                   # "ADMISSIBLE" | "INADMISSIBLE_LOW_SAFETY" | "INADMISSIBLE_LOW_SUPPORT"
    critical_entities: List[int] = field(default_factory=list)
    entity_best_camera: Dict[int, str] = field(default_factory=dict)
    entity_camera_rhos: Dict[int, Dict[str, float]] = field(default_factory=dict)
    reason: str = ""

    def to_legacy_corridor_support_record(self) -> CorridorSupportRecord:
        """Converts to CorridorSupportRecord for 100% backward compatibility with DecisionMaker."""
        return CorridorSupportRecord(
            corridor_direction=self.corridor_direction,
            safety_score=self.safety_score,
            perceptual_support=self.multi_cam_support,
            is_admissible=self.is_admissible,
            status_label=self.status_label,
            critical_entities=self.critical_entities,
            entity_rho_fov={eid: max(rhos.values()) if rhos else 1.0 for eid, rhos in self.entity_camera_rhos.items()},
            reason=self.reason,
        )


class MultiCameraSupportGate:
    """
    Computes multi-camera corridor support and enforces the Stage-B admissibility gate.
    """

    def __init__(
        self,
        tau_safe: float = 0.35,
        tau_cam: float = 0.40,
        primary_camera_id: str = "CAM0",
    ) -> None:
        self.tau_safe = tau_safe
        self.tau_cam = tau_cam
        self.primary_camera_id = primary_camera_id

    def evaluate_corridors(
        self,
        candidates: List[PathCandidate],
        critical_regions: Dict[str, CriticalRegion],
        entities: Dict[int, MultiCameraEntityState],  # global_entity_id -> MultiCameraEntityState
        use_multi_cam: bool = True,                   # True: Stage-B max_c rho; False: Stage-A CAM0 only
    ) -> Dict[str, MultiCameraCorridorSupportRecord]:
        """
        Calculates corridor support and determines admissibility.
        """
        results: Dict[str, MultiCameraCorridorSupportRecord] = {}

        for cand in candidates:
            direction = cand.direction
            safety = float(getattr(cand, "total_score", getattr(cand, "score", cand.clearance)))

            # STOP maneuver is always admissible
            if cand.is_stop:
                rec = MultiCameraCorridorSupportRecord(
                    corridor_direction=direction,
                    safety_score=1.0,
                    multi_cam_support=1.0,
                    single_cam_support=1.0,
                    is_admissible=True,
                    status_label="ADMISSIBLE",
                    critical_entities=[],
                    entity_best_camera={},
                    entity_camera_rhos={},
                    reason="Stop fallback is unconditionally admissible",
                )
                results[direction] = rec
                continue

            crit_region = critical_regions.get(direction)
            crit_entity_ids = crit_region.critical_entities if crit_region else []

            # If no critical entities intersect corridor envelope, support is 1.0
            if not crit_entity_ids:
                is_adm = (safety >= self.tau_safe)
                status = "ADMISSIBLE" if is_adm else "INADMISSIBLE_LOW_SAFETY"
                rec = MultiCameraCorridorSupportRecord(
                    corridor_direction=direction,
                    safety_score=safety,
                    multi_cam_support=1.0,
                    single_cam_support=1.0,
                    is_admissible=is_adm,
                    status_label=status,
                    critical_entities=[],
                    entity_best_camera={},
                    entity_camera_rhos={},
                    reason="No critical entities in corridor envelope",
                )
                results[direction] = rec
                continue

            # Evaluate each critical entity
            multi_cam_rhos = []
            single_cam_rhos = []
            best_cams: Dict[int, str] = {}
            camera_rhos_map: Dict[int, Dict[str, float]] = {}

            for eid in crit_entity_ids:
                ent = entities.get(eid)
                if ent is None or not ent.observations:
                    # Entity unobserved or missing: default to 1.0
                    multi_cam_rhos.append(1.0)
                    single_cam_rhos.append(1.0)
                    continue

                # Single camera rho (primary)
                prim_obs = ent.observations.get(self.primary_camera_id)
                rho_prim = prim_obs.rho_fov if prim_obs else 0.0
                single_cam_rhos.append(rho_prim)

                # Multi-camera rhos
                rhos = {cam_id: obs.rho_fov for cam_id, obs in ent.observations.items()}
                camera_rhos_map[eid] = rhos

                best_cam = max(rhos, key=rhos.get) if rhos else self.primary_camera_id
                best_rho = rhos[best_cam] if rhos else 0.0

                best_cams[eid] = best_cam
                multi_cam_rhos.append(best_rho)

            single_support = min(single_cam_rhos) if single_cam_rhos else 1.0
            multi_support = min(multi_cam_rhos) if multi_cam_rhos else 1.0

            effective_support = multi_support if use_multi_cam else single_support

            # Admissibility conjunction
            is_safe = safety >= self.tau_safe
            is_supported = effective_support >= self.tau_cam
            is_admissible = is_safe and is_supported

            if not is_safe:
                status = "INADMISSIBLE_LOW_SAFETY"
                reason = f"Safety score ({safety:.2f}) < tau_safe ({self.tau_safe:.2f})"
            elif not is_supported:
                status = "INADMISSIBLE_LOW_SUPPORT"
                reason = f"Perceptual support ({effective_support:.2f}) < tau_cam ({self.tau_cam:.2f})"
            else:
                status = "ADMISSIBLE"
                reason = f"Admissible (safety={safety:.2f}, support={effective_support:.2f})"

            rec = MultiCameraCorridorSupportRecord(
                corridor_direction=direction,
                safety_score=safety,
                multi_cam_support=multi_support,
                single_cam_support=single_support,
                is_admissible=is_admissible,
                status_label=status,
                critical_entities=crit_entity_ids,
                entity_best_camera=best_cams,
                entity_camera_rhos=camera_rhos_map,
                reason=reason,
            )
            results[direction] = rec

        return results
