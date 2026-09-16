"""
ORCA EYE — Stage A: Camera Responsibility State Machine & Abstraction
=====================================================================
Defines camera responsibility states:
  PRIMARY   : Directly responsible for perceiving the selected navigation corridor
  SUPPORT   : Assigned to track lateral critical entities or flanking threats
  RESERVED  : Earmarked for impending maneuver transition or high-risk zone
  STANDBY   : Low-power idle state or inactive sensor

In Stage A (single camera):
  The single physical camera serves as PRIMARY.
  Virtual / abstraction hooks are established so that Stage B (multi-camera array)
  can seamlessly deploy secondary cameras without altering navigation logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CameraRole(str, Enum):
    PRIMARY = "PRIMARY"
    SUPPORT = "SUPPORT"
    RESERVED = "RESERVED"
    STANDBY = "STANDBY"


@dataclass
class CameraAssignment:
    """
    Current responsibility assignment and target focus of a camera.
    """
    camera_id: str
    role: CameraRole
    is_physical: bool = True
    active: bool = True
    assigned_corridor: Optional[str] = None
    target_entities: List[int] = field(default_factory=list)
    support_score: float = 1.0
    status_notes: str = ""


class CameraResponsibilityManager:
    """
    Manages camera responsibility assignments.
    Stage A enforces single-camera PRIMARY operation with Stage B virtual stubs.
    """

    def __init__(self, primary_camera_id: str = "PRIMARY_CAM") -> None:
        self.primary_cam_id = primary_camera_id
        self.assignments: Dict[str, CameraAssignment] = {
            self.primary_cam_id: CameraAssignment(
                camera_id=self.primary_cam_id,
                role=CameraRole.PRIMARY,
                is_physical=True,
                active=True,
                assigned_corridor="STRAIGHT",
                status_notes="Active frontal navigation perception",
            )
        }
        # Virtual stubs for Stage B multi-camera preparedness
        self.virtual_cams = ["SUPPORT_LEFT", "SUPPORT_RIGHT"]
        for vcam in self.virtual_cams:
            self.assignments[vcam] = CameraAssignment(
                camera_id=vcam,
                role=CameraRole.STANDBY,
                is_physical=False,
                active=False,
                status_notes="Stage B placeholder",
            )

    def update_responsibilities(
        self,
        selected_corridor: str,
        critical_entities: List[int],
        support_score: float = 1.0,
    ) -> Dict[str, CameraAssignment]:
        """
        Update camera responsibility assignments based on current navigation decision.
        """
        # Primary camera always covers the selected corridor in Stage A
        primary = self.assignments[self.primary_cam_id]
        primary.assigned_corridor = selected_corridor
        primary.target_entities = list(critical_entities)
        primary.support_score = round(support_score, 3)
        primary.status_notes = f"Tracking {len(critical_entities)} critical entities for corridor {selected_corridor}"

        return self.assignments

    def get_primary_assignment(self) -> CameraAssignment:
        return self.assignments[self.primary_cam_id]
