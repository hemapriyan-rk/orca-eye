"""
ORCA EYE — Stage B: Predictive Camera Handover & Responsibility Manager
======================================================================
Implements proactive responsibility transfer:
  PRIMARY -> PRE_ARM -> TRANSFER -> SECONDARY -> RELEASED

Handover rule:
  HandoverReady(e, c0, c1) = [rho_0 < tau_release]
                           and [rho_1 > tau_acquire]
                           and [A_01 > tau_assoc]

Tracks:
  - Handover Lead Time: T_lead = t_actual_loss - t_transfer (T_lead > 0 is predictive)
  - Observation Gap: G = t_secondary_acq - t_primary_loss (G <= 0 is continuous)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from navigation.camera_observation import CameraObservation, MultiCameraEntityState
from navigation.cross_camera_association import CrossCameraAssociator


class HandoverState(str, Enum):
    PRIMARY = "PRIMARY"
    PRE_ARM = "PRE_ARM"
    TRANSFER = "TRANSFER"
    SECONDARY = "SECONDARY"
    RELEASED = "RELEASED"


class HandoverMode(str, Enum):
    B0_SINGLE_CAM = "B0_SINGLE_CAM"
    B1_REACTIVE = "B1_REACTIVE"
    B2_PREDICTIVE = "B2_PREDICTIVE"


@dataclass
class HandoverEvent:
    """Detailed event log for every state transition in the handover controller."""
    entity_id: int
    timestamp: float
    previous_state: str
    new_state: str
    trigger_reason: str
    rho_primary: float
    rho_secondary: float
    association_confidence: float
    handover_lead_time_s: Optional[float] = None
    observation_gap_s: Optional[float] = None


class PredictiveHandoverManager:
    """
    Manages multi-camera entity responsibility and executes proactive handovers.
    """

    def __init__(
        self,
        primary_camera_id: str = "CAM0",
        secondary_camera_id: str = "CAM1",
        tau_release: float = 0.30,
        tau_acquire: float = 0.25,
        tau_assoc: float = 0.70,
        min_dwell_time_s: float = 0.50,
        mode: HandoverMode = HandoverMode.B2_PREDICTIVE,
    ) -> None:
        self.primary_camera_id = primary_camera_id
        self.secondary_camera_id = secondary_camera_id
        self.tau_release = tau_release
        self.tau_acquire = tau_acquire
        self.tau_assoc = tau_assoc
        self.min_dwell_time_s = min_dwell_time_s
        self.mode = mode

        self.associator = CrossCameraAssociator(tau_assoc=tau_assoc)
        self.handover_events: List[HandoverEvent] = []

    def update_entity_handover(
        self,
        entity: MultiCameraEntityState,
        association_confidence: float = 0.0,
        current_time: Optional[float] = None,
    ) -> MultiCameraEntityState:
        """
        Executes the handover state machine for an individual entity.
        """
        now = time.time() if current_time is None else current_time
        if entity.state_enter_timestamp is None:
            entity.state_enter_timestamp = now

        obs_prim = entity.observations.get(self.primary_camera_id)
        obs_sec = entity.observations.get(self.secondary_camera_id)

        rho_prim = obs_prim.rho_fov if obs_prim else 0.0
        rho_sec = obs_sec.rho_fov if obs_sec else 0.0
        entity.association_confidence = association_confidence

        # Track actual loss and acquisition times
        if obs_prim and not obs_prim.in_fov and entity.primary_loss_timestamp is None:
            entity.primary_loss_timestamp = now
        elif obs_prim and obs_prim.in_fov and entity.primary_loss_timestamp is not None:
            # Reacquired by primary
            pass

        if obs_sec and obs_sec.in_fov and entity.secondary_acq_timestamp is None:
            entity.secondary_acq_timestamp = now

        dwell_time = max(0.0, now - entity.state_enter_timestamp)

        # -------------------------------------------------------------
        # Mode B0: Single Camera Baseline (No Handover)
        # -------------------------------------------------------------
        if self.mode == HandoverMode.B0_SINGLE_CAM:
            entity.responsible_camera_id = self.primary_camera_id
            entity.responsibility_state = HandoverState.PRIMARY.value
            return entity

        # -------------------------------------------------------------
        # Mode B1: Reactive Handover (Transfer ONLY after primary loss)
        # -------------------------------------------------------------
        if self.mode == HandoverMode.B1_REACTIVE:
            primary_lost = (obs_prim is None or not obs_prim.in_fov)
            sec_visible = (obs_sec is not None and obs_sec.in_fov)

            if entity.responsibility_state == HandoverState.PRIMARY.value:
                if primary_lost and sec_visible:
                    # Reactive transfer
                    self._transition_state(
                        entity=entity,
                        new_state=HandoverState.SECONDARY.value,
                        responsible_cam=self.secondary_camera_id,
                        now=now,
                        reason="Reactive handover: Primary FOV lost, secondary acquired",
                        rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
                    )
            elif entity.responsibility_state == HandoverState.SECONDARY.value:
                # If primary reacquires and secondary loses
                if (obs_sec is None or not obs_sec.in_fov) and (obs_prim is not None and obs_prim.in_fov):
                    self._transition_state(
                        entity=entity,
                        new_state=HandoverState.PRIMARY.value,
                        responsible_cam=self.primary_camera_id,
                        now=now,
                        reason="Reactive handover: Secondary lost, primary reacquired",
                        rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
                    )
            return entity

        # -------------------------------------------------------------
        # Mode B2: Predictive Handover (Proactive Pre-Arm & Transfer)
        # -------------------------------------------------------------
        current_state = entity.responsibility_state

        if current_state == HandoverState.PRIMARY.value:
            # Check for Pre-Arm Condition:
            # Primary survival declining toward tau_release while secondary has sufficient support
            is_pre_arm = (rho_prim < self.tau_release) and (rho_sec > self.tau_acquire)
            if is_pre_arm:
                entity.pre_arm_timestamp = now
                self._transition_state(
                    entity=entity,
                    new_state=HandoverState.PRE_ARM.value,
                    responsible_cam=self.primary_camera_id,  # Still primary, but secondary pre-armed
                    now=now,
                    reason=f"Predictive Pre-Arm: rho_prim ({rho_prim:.2f}) < {self.tau_release:.2f} and rho_sec ({rho_sec:.2f}) > {self.tau_acquire:.2f}",
                    rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
                )

        elif current_state == HandoverState.PRE_ARM.value:
            # Check for Transfer Condition:
            # Secondary confirmed with sufficient association confidence
            can_transfer = (association_confidence >= self.tau_assoc) and (rho_sec >= self.tau_acquire)
            if can_transfer and dwell_time >= 0.10:  # Brief confirm window
                entity.transfer_timestamp = now
                self._transition_state(
                    entity=entity,
                    new_state=HandoverState.TRANSFER.value,
                    responsible_cam=self.secondary_camera_id,
                    now=now,
                    reason=f"Predictive Transfer: Association {association_confidence:.2f} >= {self.tau_assoc:.2f}",
                    rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
                )
            elif rho_prim >= self.tau_release + 0.10:
                # Threat aborted / entity turned back into center (with hysteresis)
                self._transition_state(
                    entity=entity,
                    new_state=HandoverState.PRIMARY.value,
                    responsible_cam=self.primary_camera_id,
                    now=now,
                    reason="Pre-Arm aborted: Primary support recovered",
                    rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
                )

        elif current_state == HandoverState.TRANSFER.value:
            # Finalize into SECONDARY
            self._transition_state(
                entity=entity,
                new_state=HandoverState.SECONDARY.value,
                responsible_cam=self.secondary_camera_id,
                now=now,
                reason="Handover complete: Secondary active, primary released",
                rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
            )

        elif current_state == HandoverState.SECONDARY.value:
            # Check if entity returns to primary coverage
            can_return = (rho_prim > self.tau_acquire) and (rho_sec < self.tau_release)
            if can_return and dwell_time >= self.min_dwell_time_s and association_confidence >= self.tau_assoc:
                self._transition_state(
                    entity=entity,
                    new_state=HandoverState.PRIMARY.value,
                    responsible_cam=self.primary_camera_id,
                    now=now,
                    reason=f"Reverse Handover: Primary reacquired with rho_prim ({rho_prim:.2f}) > {self.tau_acquire:.2f}",
                    rho_p=rho_prim, rho_s=rho_sec, assoc=association_confidence,
                )

        return entity

    def _transition_state(
        self,
        entity: MultiCameraEntityState,
        new_state: str,
        responsible_cam: str,
        now: float,
        reason: str,
        rho_p: float,
        rho_s: float,
        assoc: float,
    ) -> None:
        prev_state = entity.responsibility_state
        entity.responsibility_state = new_state
        entity.responsible_camera_id = responsible_cam
        entity.state_enter_timestamp = now

        lead_time = None
        gap = None
        if new_state in (HandoverState.TRANSFER.value, HandoverState.SECONDARY.value):
            # Compute lead time if actual primary loss timestamp is known or anticipated
            if entity.primary_loss_timestamp is not None and entity.transfer_timestamp is not None:
                lead_time = entity.primary_loss_timestamp - entity.transfer_timestamp
                entity.handover_lead_time_s = lead_time

            if entity.secondary_acq_timestamp is not None and entity.primary_loss_timestamp is not None:
                gap = entity.secondary_acq_timestamp - entity.primary_loss_timestamp
                entity.observation_gap_s = gap

        event = HandoverEvent(
            entity_id=entity.global_entity_id,
            timestamp=now,
            previous_state=prev_state,
            new_state=new_state,
            trigger_reason=reason,
            rho_primary=round(rho_p, 4),
            rho_secondary=round(rho_s, 4),
            association_confidence=round(assoc, 4),
            handover_lead_time_s=round(lead_time, 4) if lead_time is not None else None,
            observation_gap_s=round(gap, 4) if gap is not None else None,
        )
        self.handover_events.append(event)
