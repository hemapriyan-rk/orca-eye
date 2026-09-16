"""
ORCA EYE — Stage 6: Candidate Path Generator
=============================================
Generates a set of candidate walking corridors from the spatial map.

Each candidate is an arc trajectory through the grid, parameterized
by a lateral angle offset from straight ahead.

Inputs  : SpatialMap, frame dimensions, config
Outputs : List[PathCandidate]

PathCandidate exposes all intermediate values for logging and research.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract — PathCandidate
# ---------------------------------------------------------------------------

@dataclass
class PathCandidate:
    """
    A single candidate walking trajectory.

    All fields are exposed for logging, scoring, and failure analysis.
    """
    direction: str               # "LEFT" | "SLIGHT_LEFT" | "STRAIGHT" | etc.
    angle_deg: float             # lateral angle offset (negative=left)
    points: List[Tuple[int, int]] = field(default_factory=list)
                                 # list of (row, col) grid cells along path
    clearance: float = 0.0       # minimum free_prob along path [0, 1]
    width: float = 0.0           # effective corridor width in grid cells
    progress: float = 0.0        # forward progress metric [0, 1]
    curvature: float = 0.0       # |angle| normalized to [0, 1]
    uncertainty: float = 1.0     # mean uncertainty along path [0, 1]
    risk: float = 1.0            # mean occupancy along path [0, 1]
    score: float = 0.0           # filled by PathScorer
    is_stop: bool = False        # True for the STOP candidate
    # Phase 1 additions:
    min_clearance: float = 0.0
    avg_clearance: float = 0.0
    free_space_score: float = 0.0
    dynamic_conflict_score: float = 0.0
    heading: float = 0.0
    total_score: float = 0.0

    def __post_init__(self) -> None:
        if self.heading == 0.0 and self.angle_deg != 0.0:
            self.heading = self.angle_deg
        if self.min_clearance == 0.0 and self.clearance != 0.0:
            self.min_clearance = self.clearance
        if self.total_score == 0.0 and self.score != 0.0:
            self.total_score = self.score


# ---------------------------------------------------------------------------
# Path Generator
# ---------------------------------------------------------------------------

class PathGenerator:
    """
    Generates candidate trajectories through the navigation grid.

    Candidates are defined by angle offsets from straight ahead.
    For each angle, an arc of grid cells is sampled along the path direction.
    The STOP candidate is always included.

    Configuration keys (from cfg['path_generation']):
      candidate_angles_deg : list of angles (deg) from straight (0 = forward)
      path_depth           : number of grid rows to project forward
      min_corridor_width   : minimum width (grid cells) to mark as viable
      include_stop         : always include STOP
    """

    _DIRECTION_NAMES = {
        -30.0: "LEFT",
        -15.0: "SLIGHT_LEFT",
         0.0:  "STRAIGHT",
        15.0:  "SLIGHT_RIGHT",
        30.0:  "RIGHT",
    }

    def __init__(self, cfg: dict, grid_rows: int, grid_cols: int) -> None:
        self.angles: List[float] = cfg.get("candidate_angles_deg", [-30.0, -15.0, 0.0, 15.0, 30.0])
        self.path_depth: int = cfg.get("path_depth", 8)
        self.min_corridor_width: int = cfg.get("min_corridor_width", 2)
        self.include_stop: bool = cfg.get("include_stop", True)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._mid_col = grid_cols // 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, spatial_map) -> List[PathCandidate]:
        """
        Generate all candidate paths from the current spatial map.

        Parameters
        ----------
        spatial_map : SpatialMap

        Returns
        -------
        List[PathCandidate]
        """
        candidates: List[PathCandidate] = []

        for angle_deg in self.angles:
            direction = self._angle_to_direction(angle_deg)
            candidate = self._generate_arc(spatial_map, angle_deg, direction)
            candidates.append(candidate)

        if self.include_stop:
            candidates.append(self._make_stop_candidate())

        logger.debug(
            "Generated %d path candidates: %s",
            len(candidates),
            [c.direction for c in candidates]
        )
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_arc(
        self,
        spatial_map,
        angle_deg: float,
        direction: str,
    ) -> PathCandidate:
        """
        Sample a curved path through the grid at the given lateral angle.

        The path starts from the bottom-center of the grid (user position)
        and projects forward (toward smaller row indices) with a lateral drift
        proportional to tan(angle).

        Parameters
        ----------
        spatial_map : SpatialMap
        angle_deg   : float  — positive = right, negative = left
        direction   : str    — human-readable name
        """
        # Convert angle to column drift per row step
        angle_rad = math.radians(angle_deg)
        col_drift = math.tan(angle_rad)  # cols shifted per row step

        # Start from bottom-center row
        start_row = self.grid_rows - 1
        start_col = float(self._mid_col)

        path_cells: List[Tuple[int, int]] = []
        free_probs: List[float] = []
        uncertainties: List[float] = []
        occupancies: List[float] = []

        for step in range(self.path_depth):
            row = start_row - step
            col = int(round(start_col + step * col_drift))

            # Clamp to grid bounds
            row = max(0, min(self.grid_rows - 1, row))
            col = max(0, min(self.grid_cols - 1, col))

            cell = spatial_map.get_cell(row, col)
            if cell is None:
                break

            path_cells.append((row, col))
            free_probs.append(cell.free_prob)
            uncertainties.append(cell.uncertainty)
            occupancies.append(cell.occupancy)

        if not path_cells:
            return PathCandidate(
                direction=direction,
                angle_deg=angle_deg,
                points=[],
                clearance=0.0,
                width=0.0,
                progress=0.0,
                curvature=abs(angle_deg) / 45.0,
                uncertainty=1.0,
                risk=1.0,
            )

        min_clearance = float(min(free_probs)) if free_probs else 0.0
        avg_clearance = float(np.mean(free_probs)) if free_probs else 0.0
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 1.0
        mean_risk = float(np.mean(occupancies)) if occupancies else 1.0
        progress = len(path_cells) / self.path_depth  # fraction of intended depth reached
        curvature = abs(angle_deg) / 45.0  # normalize to [0, 1] over ±45°

        # Corridor width: sample adjacent columns at midpoint
        mid_step = len(path_cells) // 2
        width = self._measure_width(spatial_map, path_cells, mid_step)

        return PathCandidate(
            direction=direction,
            angle_deg=angle_deg,
            points=path_cells,
            clearance=min_clearance,
            width=width,
            progress=progress,
            curvature=curvature,
            uncertainty=mean_uncertainty,
            risk=mean_risk,
            min_clearance=min_clearance,
            avg_clearance=avg_clearance,
            free_space_score=avg_clearance,
            heading=angle_deg,
        )

    def _measure_width(
        self,
        spatial_map,
        path_cells: List[Tuple[int, int]],
        step_idx: int,
    ) -> float:
        """
        Measure the lateral walkable corridor width at a given step.
        Expands left and right from the path cell until hitting an obstacle.
        Returns width in grid cells.
        """
        if step_idx >= len(path_cells):
            return 0.0

        row, col = path_cells[step_idx]
        left_width = 0
        right_width = 0

        # Expand left
        for dc in range(1, self.grid_cols):
            c = col - dc
            if c < 0:
                break
            cell = spatial_map.get_cell(row, c)
            if cell is None or cell.free_prob < 0.4:
                break
            left_width += 1

        # Expand right
        for dc in range(1, self.grid_cols):
            c = col + dc
            if c >= self.grid_cols:
                break
            cell = spatial_map.get_cell(row, c)
            if cell is None or cell.free_prob < 0.4:
                break
            right_width += 1

        return float(left_width + right_width + 1)  # +1 for the path cell itself

    def _make_stop_candidate(self) -> PathCandidate:
        """Return the STOP pseudo-candidate."""
        return PathCandidate(
            direction="STOP",
            angle_deg=0.0,
            points=[],
            clearance=1.0,   # STOP is always "safe" in terms of clearance
            width=0.0,
            progress=0.0,    # No forward progress
            curvature=0.0,
            uncertainty=0.0,
            risk=0.0,
            is_stop=True,
        )

    def _angle_to_direction(self, angle_deg: float) -> str:
        """Map an angle to a human-readable direction string."""
        # Round to nearest known angle
        known = min(self._DIRECTION_NAMES.keys(), key=lambda a: abs(a - angle_deg))
        if abs(known - angle_deg) < 5.0:
            return self._DIRECTION_NAMES[known]
        return f"ARC_{angle_deg:+.0f}deg"
