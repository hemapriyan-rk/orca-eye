"""
ORCA EYE — Stage 6: A* 3D Voxel Path Generator
================================================
Replaces fixed arc sampling with A* graph search on a 3D voxel grid:

    Grid: rows(12) × cols(20) × depth_bands(4)
    Total nodes: 960  →  A* finds path in < 1ms

Depth bands  (from config depth_band_thresholds = [0.25, 0.45, 0.70]):
    0 = VERY_NEAR  depth < 0.25   (high cost — likely collision)
    1 = NEAR       depth < 0.45   (medium cost)
    2 = MID        depth < 0.70   (low cost)
    3 = FAR        depth >= 0.70  (free — preferred)

For each candidate direction (angle → goal column), one A* search runs
and returns the optimal safe path through 3D space.

Node cost:
    g(n) = w_occ   * occ[r,c]
          + w_depth * depth_band_penalty(band)
          + w_unc   * unc[r,c]
          + w_curv  * |col - mid_col| / max_col    (prefer straight)

Heuristic (admissible):
    h(n) = rows_remaining  (Manhattan forward-progress estimate)

Output: List[PathCandidate] — same contract as arc generator.
        New fields: depth_profile, depth_clearance, path_3d.

Falls back to arc sampling if use_astar=False in config.

Inputs  : SpatialMap
Outputs : List[PathCandidate]
"""

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract — PathCandidate
# ---------------------------------------------------------------------------

@dataclass
class PathCandidate:
    """
    A single candidate walking trajectory.
    All fields exposed for logging, scoring, and failure analysis.
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
    # Phase 1 fields
    min_clearance: float = 0.0
    avg_clearance: float = 0.0
    free_space_score: float = 0.0
    dynamic_conflict_score: float = 0.0
    heading: float = 0.0
    total_score: float = 0.0
    # 3D depth fields (A* output)
    depth_profile: List[float] = field(default_factory=list)  # depth per step
    depth_clearance: float = 0.0    # min depth along path (0=near, 1=far)
    path_3d: List[Tuple[int, int, int]] = field(default_factory=list)  # (r,c,band)
    corridor_3d_clearance_m: float = 0.0  # physical clearance distance in 3D ground space (meters)

    def __post_init__(self) -> None:
        if self.heading == 0.0 and self.angle_deg != 0.0:
            self.heading = self.angle_deg
        if self.min_clearance == 0.0 and self.clearance != 0.0:
            self.min_clearance = self.clearance
        if self.total_score == 0.0 and self.score != 0.0:
            self.total_score = self.score


# ---------------------------------------------------------------------------
# Depth band helpers
# ---------------------------------------------------------------------------

_DEPTH_BAND_LABELS = ["VERY_NEAR", "NEAR", "MID", "FAR"]

# Cost multiplier per depth band (0=very near = highest cost)
_DEPTH_BAND_COST = [1.0, 0.6, 0.2, 0.0]


def _depth_to_band(depth_val: float, thresholds: List[float]) -> int:
    """Map a [0,1] depth value to a band index (0=very near, N=far)."""
    for i, t in enumerate(thresholds):
        if depth_val < t:
            return i
    return len(thresholds)  # FAR


# ---------------------------------------------------------------------------
# A* on 3D voxel grid
# ---------------------------------------------------------------------------

class _AStarSolver:
    """
    A* search on a (rows × cols × depth_bands) grid.

    State  : (row, col, depth_band)
    Start  : (rows-1, mid_col, band_of(mid_col))
    Goal   : row == 0 AND col in goal_col_range
    """

    def __init__(
        self,
        occ_arr:   np.ndarray,   # (rows, cols) float32
        free_arr:  np.ndarray,   # (rows, cols) float32
        depth_arr: np.ndarray,   # (rows, cols) float32
        unc_arr:   np.ndarray,   # (rows, cols) float32
        thresholds: List[float],
        w_occ: float,
        w_depth: float,
        w_unc: float,
        w_curv: float,
        mid_col: int,
    ) -> None:
        self.occ   = occ_arr
        self.free  = free_arr
        self.depth = depth_arr
        self.unc   = unc_arr
        self.rows, self.cols = occ_arr.shape
        self.n_bands   = len(thresholds) + 1
        self.thresholds = thresholds
        self.w_occ   = w_occ
        self.w_depth = w_depth
        self.w_unc   = w_unc
        self.w_curv  = w_curv
        self.mid_col = mid_col

    def _node_cost(self, r: int, c: int) -> float:
        band = _depth_to_band(float(self.depth[r, c]), self.thresholds)
        # Floor rows directly near feet (bottom 2 rows of grid) are naturally close
        # and should not be penalized as obstacles.
        if r >= self.rows - 2:
            depth_pen = 0.0
        else:
            depth_pen = _DEPTH_BAND_COST[min(band, len(_DEPTH_BAND_COST) - 1)]
        curv_pen  = abs(c - self.mid_col) / max(self.cols - 1, 1)
        return (
            self.w_occ   * float(self.occ[r, c])
            + self.w_depth * depth_pen
            + self.w_unc   * float(self.unc[r, c])
            + self.w_curv  * curv_pen
        )

    def _heuristic(self, r: int, goal_col: int) -> float:
        # Forward rows remaining + lateral distance to goal column
        return float(r) + 0.5 * abs(self.mid_col - goal_col) / max(self.cols - 1, 1)

    def search(
        self,
        goal_col: int,
        max_depth: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Run A* from (rows-1, mid_col) toward (0, goal_col).
        Returns list of (row, col, depth_band) nodes — empty on failure.
        """
        start_r = self.rows - 1
        start_c = self.mid_col
        start_b = _depth_to_band(float(self.depth[start_r, start_c]), self.thresholds)

        # (f_cost, g_cost, row, col, band, parent_key)
        start_state = (start_r, start_c, start_b)
        g_costs: Dict[Tuple, float] = {start_state: 0.0}
        parents: Dict[Tuple, Optional[Tuple]] = {start_state: None}

        h0 = self._heuristic(start_r, goal_col)
        open_heap = [(h0, 0.0, start_r, start_c, start_b)]

        best_goal = None
        best_g    = float("inf")

        steps = 0
        while open_heap and steps < 4096:
            steps += 1
            f, g, r, c, b = heapq.heappop(open_heap)
            state = (r, c, b)

            if g > g_costs.get(state, float("inf")) + 1e-6:
                continue  # stale entry

            # Goal: reached row 0 OR max_depth rows forward, within ±2 cols of goal
            rows_forward = (self.rows - 1) - r
            if rows_forward >= max_depth or (r == 0):
                if abs(c - goal_col) <= 2:
                    if g < best_g:
                        best_g    = g
                        best_goal = state
                    continue

            # Expand neighbours: forward (+row toward 0), ±1 col, same/adjacent band
            for dr, dc in [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                nb = _depth_to_band(float(self.depth[nr, nc]), self.thresholds)
                nstate = (nr, nc, nb)
                edge_cost = 1.0 + self._node_cost(nr, nc)
                ng = g + edge_cost
                if ng < g_costs.get(nstate, float("inf")):
                    g_costs[nstate] = ng
                    parents[nstate] = state
                    h = self._heuristic(nr, goal_col)
                    heapq.heappush(open_heap, (ng + h, ng, nr, nc, nb))

        if best_goal is None:
            return []

        # Reconstruct path
        path = []
        node = best_goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Path Generator
# ---------------------------------------------------------------------------

class PathGenerator:
    """
    Generates candidate walking trajectories via A* on a 3D voxel grid.

    One A* search per candidate direction (goal column).
    Falls back to arc sampling if use_astar=False.
    """

    _DIRECTION_NAMES = {
        -30.0: "LEFT",
        -15.0: "SLIGHT_LEFT",
         0.0:  "STRAIGHT",
        15.0:  "SLIGHT_RIGHT",
        30.0:  "RIGHT",
    }

    def __init__(self, cfg: dict, grid_rows: int, grid_cols: int) -> None:
        self.angles: List[float]   = cfg.get("candidate_angles_deg", [-30.0,-15.0,0.0,15.0,30.0])
        self.path_depth: int       = cfg.get("path_depth", 8)
        self.min_corridor_width: int = cfg.get("min_corridor_width", 2)
        self.include_stop: bool    = cfg.get("include_stop", True)
        self.use_astar: bool       = cfg.get("use_astar", True)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._mid_col  = grid_cols // 2

        # Depth bands
        self._depth_thresholds: List[float] = cfg.get("depth_band_thresholds", [0.25, 0.45, 0.70])

        # A* weights
        self._w_occ   = float(cfg.get("astar_w_occ",   3.0))
        self._w_depth = float(cfg.get("astar_w_depth",  2.0))
        self._w_unc   = float(cfg.get("astar_w_unc",    0.5))
        self._w_curv  = float(cfg.get("astar_w_curv",   0.3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, spatial_map, geometry_3d_result: Optional[Any] = None) -> List[PathCandidate]:
        """Generate candidate paths from current spatial map and optional 3D geometry."""
        candidates: List[PathCandidate] = []

        for angle_deg in self.angles:
            direction = self._angle_to_direction(angle_deg)
            if self.use_astar:
                candidate = self._generate_astar(spatial_map, angle_deg, direction)
            else:
                candidate = self._generate_arc(spatial_map, angle_deg, direction)

            # Attach 3D metric clearance if available
            if geometry_3d_result is not None:
                prof = geometry_3d_result.corridors.get(direction)
                if prof is not None:
                    candidate.corridor_3d_clearance_m = prof.clearance_m

            candidates.append(candidate)

        if self.include_stop:
            candidates.append(self._make_stop_candidate())

        return candidates

    # ------------------------------------------------------------------
    # A* path generation
    # ------------------------------------------------------------------

    def _generate_astar(
        self,
        spatial_map,
        angle_deg: float,
        direction: str,
    ) -> PathCandidate:
        """Run A* toward the goal column implied by angle_deg."""
        # Goal column: angle shifts us left/right from mid
        angle_rad = math.radians(angle_deg)
        col_offset = int(round(math.tan(angle_rad) * self.path_depth))
        goal_col = max(0, min(self.grid_cols - 1, self._mid_col + col_offset))

        solver = _AStarSolver(
            occ_arr=spatial_map.occ_arr,
            free_arr=spatial_map.free_arr,
            depth_arr=spatial_map.depth_arr,
            unc_arr=spatial_map.unc_arr,
            thresholds=self._depth_thresholds,
            w_occ=self._w_occ,
            w_depth=self._w_depth,
            w_unc=self._w_unc,
            w_curv=self._w_curv,
            mid_col=self._mid_col,
        )

        path_3d = solver.search(goal_col=goal_col, max_depth=self.path_depth)

        if not path_3d:
            return PathCandidate(
                direction=direction, angle_deg=angle_deg,
                points=[], clearance=0.0, width=0.0, progress=0.0,
                curvature=abs(angle_deg) / 45.0,
                uncertainty=1.0, risk=1.0,
                depth_clearance=0.0,
            )

        points_2d   = [(r, c) for r, c, _ in path_3d]
        free_probs  = [float(spatial_map.free_arr[r, c]) for r, c, _ in path_3d]
        uncertainties = [float(spatial_map.unc_arr[r, c]) for r, c, _ in path_3d]
        occupancies = [float(spatial_map.occ_arr[r, c]) for r, c, _ in path_3d]
        depth_vals  = [float(spatial_map.depth_arr[r, c]) for r, c, _ in path_3d]

        min_clearance = float(min(free_probs))
        avg_clearance = float(np.mean(free_probs))
        depth_clearance = float(min(depth_vals))
        mean_unc   = float(np.mean(uncertainties))
        mean_risk  = float(np.mean(occupancies))
        progress   = len(path_3d) / self.path_depth
        curvature  = abs(angle_deg) / 45.0

        mid_step = len(points_2d) // 2
        width = self._measure_width(spatial_map, points_2d, mid_step)

        return PathCandidate(
            direction=direction,
            angle_deg=angle_deg,
            points=points_2d,
            clearance=min_clearance,
            width=width,
            progress=progress,
            curvature=curvature,
            uncertainty=mean_unc,
            risk=mean_risk,
            min_clearance=min_clearance,
            avg_clearance=avg_clearance,
            free_space_score=avg_clearance,
            heading=angle_deg,
            depth_profile=depth_vals,
            depth_clearance=depth_clearance,
            path_3d=path_3d,
        )

    # ------------------------------------------------------------------
    # Arc fallback (legacy — used when use_astar=False)
    # ------------------------------------------------------------------

    def _generate_arc(
        self,
        spatial_map,
        angle_deg: float,
        direction: str,
    ) -> PathCandidate:
        angle_rad = math.radians(angle_deg)
        col_drift = math.tan(angle_rad)

        start_row = self.grid_rows - 1
        start_col = float(self._mid_col)

        path_cells: List[Tuple[int, int]] = []
        free_probs: List[float] = []
        uncertainties: List[float] = []
        occupancies: List[float] = []
        depth_vals: List[float] = []

        for step in range(self.path_depth):
            row = max(0, min(self.grid_rows - 1, start_row - step))
            col = max(0, min(self.grid_cols - 1, int(round(start_col + step * col_drift))))
            cell = spatial_map.get_cell(row, col)
            if cell is None:
                break
            path_cells.append((row, col))
            free_probs.append(cell.free_prob)
            uncertainties.append(cell.uncertainty)
            occupancies.append(cell.occupancy)
            depth_vals.append(cell.depth_mean)

        if not path_cells:
            return PathCandidate(direction=direction, angle_deg=angle_deg,
                                 curvature=abs(angle_deg)/45.0)

        min_clearance  = float(min(free_probs))
        avg_clearance  = float(np.mean(free_probs))
        depth_clearance = float(min(depth_vals))
        mean_unc  = float(np.mean(uncertainties))
        mean_risk = float(np.mean(occupancies))
        progress  = len(path_cells) / self.path_depth
        curvature = abs(angle_deg) / 45.0
        mid_step  = len(path_cells) // 2
        width     = self._measure_width(spatial_map, path_cells, mid_step)

        return PathCandidate(
            direction=direction, angle_deg=angle_deg, points=path_cells,
            clearance=min_clearance, width=width, progress=progress,
            curvature=curvature, uncertainty=mean_unc, risk=mean_risk,
            min_clearance=min_clearance, avg_clearance=avg_clearance,
            free_space_score=avg_clearance, heading=angle_deg,
            depth_profile=depth_vals, depth_clearance=depth_clearance,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _measure_width(
        self,
        spatial_map,
        path_cells: List[Tuple[int, int]],
        step_idx: int,
    ) -> float:
        if step_idx >= len(path_cells):
            return 0.0
        row, col = path_cells[step_idx]
        left_width = right_width = 0
        for dc in range(1, self.grid_cols):
            c = col - dc
            if c < 0:
                break
            if float(spatial_map.free_arr[row, c]) < 0.4:
                break
            left_width += 1
        for dc in range(1, self.grid_cols):
            c = col + dc
            if c >= self.grid_cols:
                break
            if float(spatial_map.free_arr[row, c]) < 0.4:
                break
            right_width += 1
        return float(left_width + right_width + 1)

    def _make_stop_candidate(self) -> PathCandidate:
        return PathCandidate(
            direction="STOP", angle_deg=0.0, points=[],
            clearance=1.0, width=0.0, progress=0.0,
            curvature=0.0, uncertainty=0.0, risk=0.0,
            is_stop=True, depth_clearance=1.0,
        )

    def _angle_to_direction(self, angle_deg: float) -> str:
        known = min(self._DIRECTION_NAMES.keys(), key=lambda a: abs(a - angle_deg))
        if abs(known - angle_deg) < 5.0:
            return self._DIRECTION_NAMES[known]
        return f"ARC_{angle_deg:+.0f}deg"
