"""
ORCA EYE — Stage 12b: Scenario Test Suite
==========================================
Defines 15 test scenarios (A–O) and a runner that executes them
against recorded video clips or synthetic test frames.

The purpose is to expose weaknesses, NOT to demonstrate successes.

Scenarios:
  A. Open corridor
  B. Single static obstacle
  C. Obstacle on left
  D. Obstacle on right
  E. Obstacle directly ahead
  F. Narrow passage
  G. Multiple obstacles
  H. Moving pedestrian
  I. Crossing pedestrian
  J. Partial occlusion
  K. Low-light scene
  L. Strong illumination / glare
  M. Crowded environment
  N. Detection instability
  O. Large unknown region

Each scenario logs results to logs/scenarios/<scenario_id>/.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario Definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Single test scenario descriptor."""
    scenario_id: str           # e.g. "A", "B", …
    name: str                  # human-readable name
    description: str           # what the scenario tests
    expected_behavior: str     # what a correct system should do
    failure_indicators: List[str]  # signals that indicate failure
    video_path: Optional[str] = None   # path to test video (None = skip if missing)
    synthetic_generator: Optional[Callable] = None  # optional synthetic frame generator


@dataclass
class ScenarioResult:
    """Results of running a scenario."""
    scenario_id: str
    name: str
    frames_processed: int = 0
    commands_issued: Dict[str, int] = field(default_factory=dict)
    failure_events: List[str] = field(default_factory=list)
    mean_score: float = 0.0
    min_clearance: float = 1.0
    mean_fps: float = 0.0
    notes: str = ""
    passed: Optional[bool] = None  # None = no ground truth


# ---------------------------------------------------------------------------
# Scenario Registry
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id="A",
        name="Open Corridor",
        description="Wide open walkway with no obstacles ahead.",
        expected_behavior="STRAIGHT continuously. High score. High clearance.",
        failure_indicators=["STOP when path is clear", "excessive direction changes"],
    ),
    Scenario(
        scenario_id="B",
        name="Single Static Obstacle",
        description="One stationary object (e.g. chair) directly ahead.",
        expected_behavior="Detour LEFT or RIGHT. Clearance maintained.",
        failure_indicators=["STRAIGHT into obstacle", "STOP with clear detour available"],
    ),
    Scenario(
        scenario_id="C",
        name="Obstacle on Left",
        description="Obstacle positioned on the left side of the walkway.",
        expected_behavior="RIGHT or SLIGHT_RIGHT command issued.",
        failure_indicators=["LEFT command", "STOP when right path is clear"],
    ),
    Scenario(
        scenario_id="D",
        name="Obstacle on Right",
        description="Obstacle positioned on the right side of the walkway.",
        expected_behavior="LEFT or SLIGHT_LEFT command issued.",
        failure_indicators=["RIGHT command", "STOP when left path is clear"],
    ),
    Scenario(
        scenario_id="E",
        name="Obstacle Directly Ahead",
        description="Large obstacle blocking the full forward path.",
        expected_behavior="STOP if no detour; otherwise detour command.",
        failure_indicators=["STRAIGHT through obstacle"],
    ),
    Scenario(
        scenario_id="F",
        name="Narrow Passage",
        description="Corridor narrowed by obstacles on both sides.",
        expected_behavior="STRAIGHT with low clearance but passable. STOP if too narrow.",
        failure_indicators=["STRAIGHT through sub-minimum-width passage"],
    ),
    Scenario(
        scenario_id="G",
        name="Multiple Obstacles",
        description="Several objects distributed across the scene.",
        expected_behavior="Correct detour. No STOP if at least one path is clear.",
        failure_indicators=["STOP when a valid path exists", "oscillation"],
    ),
    Scenario(
        scenario_id="H",
        name="Moving Pedestrian",
        description="A pedestrian walking in the same direction (away from camera).",
        expected_behavior="Track pedestrian. STRAIGHT with reduced speed signal. No collision.",
        failure_indicators=["STOP repeatedly", "path change oscillation"],
    ),
    Scenario(
        scenario_id="I",
        name="Crossing Pedestrian",
        description="Pedestrian crosses the path laterally.",
        expected_behavior="React to crossing: SLIGHT_LEFT or SLIGHT_RIGHT to avoid.",
        failure_indicators=["STRAIGHT through crossing pedestrian"],
    ),
    Scenario(
        scenario_id="J",
        name="Partial Occlusion",
        description="An obstacle is partially hidden behind another object.",
        expected_behavior="CAUTION or conservative path. Avoid hidden obstacle region.",
        failure_indicators=["confident STRAIGHT into occluded region"],
    ),
    Scenario(
        scenario_id="K",
        name="Low-Light Scene",
        description="Scene captured in dim or near-dark conditions.",
        expected_behavior="CAUTION due to high uncertainty. Conservative path or STOP.",
        failure_indicators=["confident direction in zero-visibility conditions"],
    ),
    Scenario(
        scenario_id="L",
        name="Strong Illumination / Glare",
        description="Direct sunlight or blown-out regions in frame.",
        expected_behavior="High uncertainty in glare region. Conservative path.",
        failure_indicators=["free-space classified as walkable in glare zones"],
    ),
    Scenario(
        scenario_id="M",
        name="Crowded Environment",
        description="Multiple people moving in various directions simultaneously.",
        expected_behavior="STOP or very conservative path. High uncertainty.",
        failure_indicators=["confident STRAIGHT through crowd", "LR oscillation"],
    ),
    Scenario(
        scenario_id="N",
        name="Detection Instability",
        description="Scene where YOLO detections flicker or disappear frame-by-frame.",
        expected_behavior="Tracker provides continuity. Path remains stable despite flickering.",
        failure_indicators=["rapid direction change each frame", "SUDDEN_DISAPPEARANCE failures"],
    ),
    Scenario(
        scenario_id="O",
        name="Large Unknown Region",
        description="Scene with heavy visual clutter or textureless surfaces.",
        expected_behavior="UNKNOWN_DOMINANCE flagged. CAUTION or STOP issued conservatively.",
        failure_indicators=["confident navigation through large unknown regions"],
    ),
]


# ---------------------------------------------------------------------------
# Scenario Runner
# ---------------------------------------------------------------------------

class ScenarioRunner:
    """
    Runs the 15-scenario test suite against video clips.

    Usage:
        runner = ScenarioRunner(pipeline_fn, cfg)
        results = runner.run_all()
        runner.save_results(results)
    """

    def __init__(
        self,
        pipeline_fn: Callable,  # callable(video_path) -> List[FrameResult]
        cfg: dict,
    ) -> None:
        """
        Parameters
        ----------
        pipeline_fn : Callable
            A function that accepts a video path string and returns
            a list of per-frame result dicts (command, score, clearance, etc.)
        cfg : dict
            Full system config.
        """
        self.pipeline_fn = pipeline_fn
        self.scenarios_dir = Path(cfg.get("evaluation", {}).get("scenarios_dir", "data/scenarios"))
        self.results_dir = Path(cfg.get("evaluation", {}).get("results_dir", "logs/scenarios"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> List[ScenarioResult]:
        """Run all 15 scenarios. Skip if video not found."""
        results: List[ScenarioResult] = []
        for scenario in SCENARIOS:
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario. Returns ScenarioResult."""
        logger.info("Running Scenario %s: %s", scenario.scenario_id, scenario.name)

        # Find video
        video_path = None
        if scenario.video_path:
            video_path = str(self.scenarios_dir / scenario.video_path)
        else:
            # Try to find by convention: scenarios/scenario_A.mp4
            candidate = self.scenarios_dir / f"scenario_{scenario.scenario_id}.mp4"
            if candidate.exists():
                video_path = str(candidate)

        if video_path is None:
            logger.warning(
                "Scenario %s: no video found at %s — skipping",
                scenario.scenario_id, self.scenarios_dir
            )
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                notes=f"SKIPPED — no video at {self.scenarios_dir}/scenario_{scenario.scenario_id}.mp4",
            )

        # Run pipeline
        try:
            frame_results = self.pipeline_fn(video_path)
        except Exception as exc:
            logger.error("Scenario %s failed: %s", scenario.scenario_id, exc)
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                notes=f"ERROR: {exc}",
            )

        if not frame_results:
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                notes="Pipeline returned no frames.",
            )

        # Aggregate results
        commands: Dict[str, int] = {}
        scores: List[float] = []
        clearances: List[float] = []
        fpss: List[float] = []
        failure_events: List[str] = []

        for fr in frame_results:
            cmd = fr.get("command", "UNKNOWN")
            commands[cmd] = commands.get(cmd, 0) + 1
            scores.append(fr.get("score", 0.0))
            clearances.append(fr.get("clearance", 0.0))
            fpss.append(fr.get("fps", 0.0))
            failure_events.extend(fr.get("failure_codes", []))

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            name=scenario.name,
            frames_processed=len(frame_results),
            commands_issued=commands,
            failure_events=failure_events,
            mean_score=float(sum(scores) / len(scores)) if scores else 0.0,
            min_clearance=float(min(clearances)) if clearances else 0.0,
            mean_fps=float(sum(fpss) / len(fpss)) if fpss else 0.0,
        )
        return result

    def save_results(self, results: List[ScenarioResult]) -> None:
        """Save all scenario results to JSON + Markdown."""
        # JSON
        json_path = self.results_dir / "scenario_results.json"
        json_data = []
        for r in results:
            json_data.append({
                "scenario_id": r.scenario_id,
                "name": r.name,
                "frames_processed": r.frames_processed,
                "commands_issued": r.commands_issued,
                "failure_events": r.failure_events[:20],  # cap for file size
                "mean_score": round(r.mean_score, 4),
                "min_clearance": round(r.min_clearance, 4),
                "mean_fps": round(r.mean_fps, 2),
                "notes": r.notes,
                "passed": r.passed,
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # Markdown
        md_path = self.results_dir / "scenario_results.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# ORCA EYE — Scenario Test Results\n\n")
            f.write("> RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE\n\n")
            f.write("| ID | Name | Frames | Score | MinClear | FPS | Failures | Notes |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in results:
                fail_count = len(r.failure_events)
                f.write(
                    f"| {r.scenario_id} | {r.name} | {r.frames_processed} | "
                    f"{r.mean_score:.3f} | {r.min_clearance:.3f} | {r.mean_fps:.1f} | "
                    f"{fail_count} | {r.notes[:40]} |\n"
                )

        logger.info("Scenario results saved to %s", self.results_dir)


def list_scenarios() -> None:
    """Print all scenarios to stdout. Useful for CLI inspection."""
    print("\nORCA EYE — Test Scenarios\n" + "=" * 50)
    for s in SCENARIOS:
        print(f"\n[{s.scenario_id}] {s.name}")
        print(f"  Description : {s.description}")
        print(f"  Expected    : {s.expected_behavior}")
        print(f"  Failures    : {', '.join(s.failure_indicators)}")


# ===========================================================================
# Stage A: Controlled Navigation Scenario Suite (Synthetic & Deterministic)
# ===========================================================================

from enum import Enum


class ControlledScenarioType(str, Enum):
    CLEAR_PATH = "CLEAR_PATH"
    CENTRAL_STATIC_OBSTACLE = "CENTRAL_STATIC_OBSTACLE"
    LEFT_BLOCKED_RIGHT_OPEN = "LEFT_BLOCKED_RIGHT_OPEN"
    RIGHT_BLOCKED_LEFT_OPEN = "RIGHT_BLOCKED_LEFT_OPEN"
    BOTH_SIDES_BLOCKED = "BOTH_SIDES_BLOCKED"
    CROSSING_PEDESTRIAN = "CROSSING_PEDESTRIAN"
    BLIND_UNCERTAINTY = "BLIND_UNCERTAINTY"


@dataclass
class ControlledScenarioSpec:
    scenario_type: ControlledScenarioType
    name: str
    description: str
    expected_command: str
    admissible_commands: List[str]


CONTROLLED_SUITE: Dict[ControlledScenarioType, ControlledScenarioSpec] = {
    ControlledScenarioType.CLEAR_PATH: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.CLEAR_PATH,
        name="Clear Path",
        description="Unobstructed straight corridor ahead with full visibility.",
        expected_command="STRAIGHT",
        admissible_commands=["STRAIGHT", "CONTINUE_STRAIGHT"],
    ),
    ControlledScenarioType.CENTRAL_STATIC_OBSTACLE: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.CENTRAL_STATIC_OBSTACLE,
        name="Central Static Obstacle",
        description="Stationary obstacle centered at x=0, y=2.0m requiring lateral veer.",
        expected_command="SLIGHT_RIGHT",
        admissible_commands=["SLIGHT_RIGHT", "SLIGHT_LEFT", "RIGHT", "LEFT", "STRAIGHT"],
    ),
    ControlledScenarioType.LEFT_BLOCKED_RIGHT_OPEN: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.LEFT_BLOCKED_RIGHT_OPEN,
        name="Left Blocked / Right Open",
        description="Left side blocked by wall or obstacle, right side clear.",
        expected_command="SLIGHT_RIGHT",
        admissible_commands=["SLIGHT_RIGHT", "RIGHT", "STRAIGHT"],
    ),
    ControlledScenarioType.RIGHT_BLOCKED_LEFT_OPEN: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.RIGHT_BLOCKED_LEFT_OPEN,
        name="Right Blocked / Left Open",
        description="Right side blocked by wall or obstacle, left side clear.",
        expected_command="SLIGHT_LEFT",
        admissible_commands=["SLIGHT_LEFT", "LEFT", "STRAIGHT"],
    ),
    ControlledScenarioType.BOTH_SIDES_BLOCKED: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.BOTH_SIDES_BLOCKED,
        name="Both Sides Blocked / Impassable",
        description="Wall or obstacles completely blocking forward progress across all corridors.",
        expected_command="STOP",
        admissible_commands=["STOP"],
    ),
    ControlledScenarioType.CROSSING_PEDESTRIAN: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.CROSSING_PEDESTRIAN,
        name="Crossing Pedestrian",
        description="Pedestrian crossing laterally into wearer's forward corridor with imminent TTC.",
        expected_command="STOP",
        admissible_commands=["STOP", "CAUTION", "SLIGHT_LEFT", "SLIGHT_RIGHT"],
    ),
    ControlledScenarioType.BLIND_UNCERTAINTY: ControlledScenarioSpec(
        scenario_type=ControlledScenarioType.BLIND_UNCERTAINTY,
        name="Blind Uncertainty",
        description="Sensor dropout or dense occlusion with high epistemic uncertainty.",
        expected_command="STOP",
        admissible_commands=["STOP", "CAUTION"],
    ),
}


def create_synthetic_spatial_state(
    scenario_type: ControlledScenarioType,
    grid_rows: int = 12,
    grid_cols: int = 20,
):
    """
    Construct deterministic synthetic SpatialMap and SpatialEntity inputs
    for a given ControlledScenarioType to enable regression testing.
    """
    from navigation.spatial_map import SpatialMap, GridCell
    from navigation.spatial_entity import SpatialEntity

    cfg = {"grid_rows": grid_rows, "grid_cols": grid_cols}
    smap = SpatialMap(cfg, frame_width=640, frame_height=480)

    # Initialize default clear map
    for r in range(grid_rows):
        for c in range(grid_cols):
            smap.grid[r][c].occupancy = 0.05
            smap.grid[r][c].free_prob = 0.90
            smap.grid[r][c].uncertainty = 0.10
            smap.grid[r][c].depth_mean = 0.80

    mid_c = grid_cols // 2
    entities: List[SpatialEntity] = []

    if scenario_type == ControlledScenarioType.CLEAR_PATH:
        # All clear, no entities
        pass

    elif scenario_type == ControlledScenarioType.CENTRAL_STATIC_OBSTACLE:
        # Obstacle in center rows, leaving flanks open for SLIGHT_RIGHT / SLIGHT_LEFT detour
        for r in range(grid_rows // 2, grid_rows - 3):
            for c in range(mid_c - 1, mid_c + 2):
                smap.grid[r][c].occupancy = 0.95
                smap.grid[r][c].free_prob = 0.05
        entities.append(SpatialEntity(
            track_id=1,
            class_id=56,  # chair
            class_name="chair",
            confidence=0.92,
            bbox=[280, 200, 360, 380],
            image_position=[320.0, 290.0],
            ground_position=[320.0, 380.0],
            relative_bearing=0.0,
            relative_distance=0.35,
            velocity=[0.0, 0.0],
            motion_direction="stationary",
            tracking_confidence=0.95,
            depth_confidence=0.90,
            depth_valid=True,
            navigation_relevance=0.85,
        ))

    elif scenario_type == ControlledScenarioType.LEFT_BLOCKED_RIGHT_OPEN:
        # Left side blocked, and forward straight path blocked ahead, right flank open
        for r in range(grid_rows - 1):
            for c in range(0, mid_c + 1):
                smap.grid[r][c].occupancy = 0.95
                smap.grid[r][c].free_prob = 0.05
        # Bottom row (wearer position)
        for c in range(0, mid_c):
            smap.grid[grid_rows - 1][c].occupancy = 0.95
            smap.grid[grid_rows - 1][c].free_prob = 0.05

    elif scenario_type == ControlledScenarioType.RIGHT_BLOCKED_LEFT_OPEN:
        # Right side blocked, and forward straight path blocked ahead, left flank open
        for r in range(grid_rows - 1):
            for c in range(mid_c, grid_cols):
                smap.grid[r][c].occupancy = 0.95
                smap.grid[r][c].free_prob = 0.05
        # Bottom row (wearer position)
        for c in range(mid_c + 1, grid_cols):
            smap.grid[grid_rows - 1][c].occupancy = 0.95
            smap.grid[grid_rows - 1][c].free_prob = 0.05

    elif scenario_type == ControlledScenarioType.BOTH_SIDES_BLOCKED:
        # Forward completely blocked
        for r in range(grid_rows // 3, grid_rows):
            for c in range(grid_cols):
                smap.grid[r][c].occupancy = 0.98
                smap.grid[r][c].free_prob = 0.02

    elif scenario_type == ControlledScenarioType.CROSSING_PEDESTRIAN:
        # Clear static map, but fast crossing pedestrian heading into center corridor
        entities.append(SpatialEntity(
            track_id=10,
            class_id=0,  # person
            class_name="person",
            confidence=0.95,
            bbox=[100, 180, 180, 420],
            image_position=[140.0, 300.0],
            ground_position=[140.0, 420.0],
            relative_bearing=-20.0,
            relative_distance=0.25,
            velocity=[15.0, 0.0],  # fast lateral velocity across screen
            motion_direction="crossing",
            tracking_confidence=0.98,
            depth_confidence=0.90,
            depth_valid=True,
            navigation_relevance=0.95,
        ))

    elif scenario_type == ControlledScenarioType.BLIND_UNCERTAINTY:
        # High uncertainty everywhere
        for r in range(grid_rows):
            for c in range(grid_cols):
                smap.grid[r][c].occupancy = 0.50
                smap.grid[r][c].free_prob = 0.20
                smap.grid[r][c].uncertainty = 0.95

    return smap, entities

