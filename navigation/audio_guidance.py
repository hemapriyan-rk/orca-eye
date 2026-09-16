"""
ORCA EYE — Stage 9: Audio Guidance and Instruction Generation
============================================================
Translates NavigationState, SpatialEntities, and DynamicConflicts into
concise, low-latency spoken instructions for assistive navigation.

Priority Arbitration:
  - Priority 1: Emergency STOP (preempts all, repeats if hazard persists)
  - Priority 2: CAUTION dynamic conflict ("Caution, person approaching from left")
  - Priority 3: Direction change ("Turn slight right", "Clear path ahead")
  - Priority 4: Status reminder (periodic confirmation every N seconds)

Temporal Suppression:
  - Strict cooldown (default 2.5s) between non-emergency instructions
  - Direction changes require 2 consecutive frames before triggering speech
  - Emergency STOP preempts immediately and repeats every 1.5s if hazard persists
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from navigation.decision import NavigationState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract — AudioInstruction
# ---------------------------------------------------------------------------

@dataclass
class AudioInstruction:
    """
    Spoken instruction payload emitted by InstructionGenerator.
    """
    priority: int                  # 1=STOP, 2=CAUTION, 3=TURN/GO, 4=REMINDER
    text: str                      # natural speech phrase
    category: str                  # "EMERGENCY_STOP", "DYNAMIC_CONFLICT", "DIRECTION_CHANGE", "REMINDER"
    preempt: bool = False          # True for Priority 1 emergency instructions
    speech_rate: int = 155         # words per minute
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Instruction Generator
# ---------------------------------------------------------------------------

class InstructionGenerator:
    """
    Arbitrates and generates audio instructions with temporal suppression.
    """

    PHRASES = {
        "STRAIGHT": "Continue straight.",
        "SLIGHT_LEFT": "Bear slightly left.",
        "SLIGHT_RIGHT": "Bear slightly right.",
        "LEFT": "Turn left.",
        "RIGHT": "Turn right.",
        "CAUTION": "Caution.",
        "STOP": "Stop.",
    }

    def __init__(self, cfg: dict) -> None:
        audio_cfg = cfg.get("audio", {})
        self.enabled: bool = audio_cfg.get("enabled", False)
        self.cooldown_seconds: float = audio_cfg.get("cooldown_seconds", 2.5)
        self.reminder_seconds: float = audio_cfg.get("reminder_seconds", 8.0)
        self.speech_rate: int = audio_cfg.get("speech_rate", 155)

        self._last_instruction_time: float = 0.0
        self._last_stop_time: float = 0.0
        self._last_instruction_text: str = ""
        self._last_confirmed_command: str = "STRAIGHT"

        # Direction change confirmation hysteresis (prevent jitter thrashing)
        self._candidate_command: str = "STRAIGHT"
        self._candidate_frame_count: int = 0
        self._required_confirmation_frames: int = 2

        # Asynchronous TTS worker setup
        self._tts_queue: Optional[queue.Queue] = None
        self._tts_thread: Optional[threading.Thread] = None
        if self.enabled:
            self._init_tts_worker()

    def _init_tts_worker(self) -> None:
        """Start a background daemon thread for non-blocking speech playback."""
        try:
            import pyttsx3
            self._tts_queue = queue.Queue(maxsize=5)

            def _worker():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.speech_rate)
                    while True:
                        item = self._tts_queue.get()
                        if item is None:
                            break
                        try:
                            engine.say(item)
                            engine.runAndWait()
                        except Exception as e:
                            logger.error("TTS playback error: %s", e)
                        finally:
                            self._tts_queue.task_done()
                except Exception as e:
                    logger.warning("Failed to initialize pyttsx3 speech engine: %s", e)

            self._tts_thread = threading.Thread(target=_worker, daemon=True)
            self._tts_thread.start()
            logger.info("Non-blocking TTS background worker initialized.")
        except ImportError:
            logger.warning("pyttsx3 is not installed. Spoken audio is disabled.")
            self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_instruction(
        self,
        nav_state: NavigationState,
        spatial_entities: Optional[List] = None,
        dynamic_conflicts: Optional[List] = None,
    ) -> Optional[AudioInstruction]:
        """
        Evaluate current navigation and dynamic state to emit the highest priority
        instruction that satisfies temporal suppression criteria.
        """
        now = time.time()

        # --------------------------------------------------------------
        # Priority 1: Emergency STOP
        # --------------------------------------------------------------
        if nav_state.is_stop or nav_state.decision == "STOP":
            # Allow repeating STOP every 1.5s if hazard persists
            if (now - self._last_stop_time) >= 1.5:
                phrase = "Stop! Path obstructed."
                if "WALL_COLLISION" in nav_state.reason:
                    phrase = "Stop! Wall directly ahead."
                elif nav_state.dynamic_conflict:
                    phrase = "Stop! Moving obstacle ahead."
                elif "CORRIDOR_BLOCKED" in nav_state.reason:
                    phrase = "Stop! Path is blocked."
                elif "LOW_CLEARANCE" in nav_state.reason:
                    phrase = "Stop! Insufficient clearance."

                instr = AudioInstruction(
                    priority=1,
                    text=phrase,
                    category="EMERGENCY_STOP",
                    preempt=True,
                    speech_rate=self.speech_rate,
                    timestamp=now,
                )
                self._last_stop_time = now
                self._last_instruction_time = now
                self._last_instruction_text = phrase
                self._last_confirmed_command = "STOP"
                return instr
            return None

        cooldown_passed = (
            self._last_instruction_time == 0.0
            or (now - self._last_instruction_time) >= self.cooldown_seconds
        )

        # --------------------------------------------------------------
        # Priority 2: Wall Proximity or Dynamic Conflict Warning
        # --------------------------------------------------------------
        if getattr(nav_state, "wall_warning", None) and cooldown_passed:
            phrase = f"Caution, wall close on {nav_state.wall_warning.lower()}."
            instr = AudioInstruction(
                priority=2,
                text=phrase,
                category="WALL_PROXIMITY",
                preempt=False,
                speech_rate=self.speech_rate,
                timestamp=now,
            )
            self._last_instruction_time = now
            self._last_instruction_text = phrase
            return instr

        if dynamic_conflicts:
            active_conflicts = [
                c for c in dynamic_conflicts
                if getattr(c, "has_conflict", False) and getattr(c, "time_to_conflict", 99.0) <= 2.2
            ]
            if active_conflicts and cooldown_passed:
                # Find most imminent conflict
                most_urgent = min(active_conflicts, key=lambda c: getattr(c, "time_to_conflict", 99.0))
                entity_label = "object"
                if spatial_entities:
                    matching = next(
                        (e for e in spatial_entities if getattr(e, "track_id", -1) == getattr(most_urgent, "track_id", -2)),
                        None
                    )
                    if matching:
                        entity_label = matching.label

                phrase = f"Caution, {entity_label} approaching."
                instr = AudioInstruction(
                    priority=2,
                    text=phrase,
                    category="DYNAMIC_CONFLICT",
                    preempt=False,
                    speech_rate=self.speech_rate,
                    timestamp=now,
                )
                self._last_instruction_time = now
                self._last_instruction_text = phrase
                return instr

        # --------------------------------------------------------------
        # Priority 3: Direction Change / Safe Path Selection
        # --------------------------------------------------------------
        current_cmd = nav_state.command
        if current_cmd != self._last_confirmed_command:
            if current_cmd == self._candidate_command:
                self._candidate_frame_count += 1
            else:
                self._candidate_command = current_cmd
                self._candidate_frame_count = 1

            if (
                self._candidate_frame_count >= self._required_confirmation_frames
                and cooldown_passed
            ):
                phrase = self.PHRASES.get(current_cmd, f"Proceed {current_cmd.lower()}.")
                instr = AudioInstruction(
                    priority=3,
                    text=phrase,
                    category="DIRECTION_CHANGE",
                    preempt=False,
                    speech_rate=self.speech_rate,
                    timestamp=now,
                )
                self._last_confirmed_command = current_cmd
                self._last_instruction_time = now
                self._last_instruction_text = phrase
                return instr
        else:
            self._candidate_frame_count = 0

        # --------------------------------------------------------------
        # Priority 4: Status / Reassurance Reminder
        # --------------------------------------------------------------
        if self._last_instruction_time > 0.0 and (now - self._last_instruction_time) >= self.reminder_seconds:
            if current_cmd == "STRAIGHT":
                phrase = "Continuing straight. Path is clear."
            else:
                phrase = f"Continuing {current_cmd.lower()}."

            instr = AudioInstruction(
                priority=4,
                text=phrase,
                category="REMINDER",
                preempt=False,
                speech_rate=self.speech_rate,
                timestamp=now,
            )
            self._last_instruction_time = now
            self._last_instruction_text = phrase
            return instr

        return None

    def speak(self, instruction: AudioInstruction) -> None:
        """Enqueue speech instruction for non-blocking playback."""
        if not self.enabled or self._tts_queue is None:
            return

        try:
            if instruction.preempt:
                while not self._tts_queue.empty():
                    try:
                        self._tts_queue.get_nowait()
                        self._tts_queue.task_done()
                    except queue.Empty:
                        break
            self._tts_queue.put_nowait(instruction.text)
        except queue.Full:
            logger.debug("TTS queue full; dropping audio instruction.")
