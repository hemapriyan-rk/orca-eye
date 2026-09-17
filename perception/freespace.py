"""
ORCA EYE — Stage 4b: Free-Space Estimation  (GPU-optimised)
============================================================
All pixel-level operations run on CUDA using torch tensors.
Only the final label_map and free_prob_map are transferred back to CPU
(as numpy arrays) for use by the spatial map and renderer.

GPU operations:
  • Geometric prior — tensor broadcast (GPU)
  • Depth masking  — boolean indexing on GPU tensor
  • Gradient       — torch.gradient on GPU tensor
  • Obstacle mask  — scatter/index ops on GPU
  • Threshold      — GPU comparison, no loop
  • CPU transfer   — single .cpu().numpy() per output array

Labels:  FREE=0  OBSTACLE=1  UNKNOWN=2
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

LABEL_FREE     = 0
LABEL_OBSTACLE = 1
LABEL_UNKNOWN  = 2


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class FreeSpaceResult:
    label_map: np.ndarray       # (H, W) uint8: FREE/OBSTACLE/UNKNOWN
    free_prob_map: np.ndarray   # (H, W) float32: probability of FREE
    uncertainty: float
    statistics: Dict[str, float]


# ---------------------------------------------------------------------------
# GPU-Optimised Free-Space Estimator
# ---------------------------------------------------------------------------

class FreeSpaceEstimator:
    """
    Hybrid free-space estimation on CUDA:
      1. Geometric prior (walkable_bottom_fraction)
      2. Depth evidence (far=walkable, near+gradient=obstacle)
      3. YOLO obstacle bboxes

    Configuration keys (from cfg['freespace']):
      walkable_bottom_fraction, obstacle_inflation,
      depth_gradient_threshold, near_depth_threshold,
      free_probability_threshold
    """

    def __init__(self, cfg: dict) -> None:
        self.walkable_frac    : float = cfg.get("walkable_bottom_fraction", 0.55)
        self.obstacle_inflation: float = cfg.get("obstacle_inflation", 1.05)
        self.depth_grad_thresh: float = cfg.get("depth_gradient_threshold", 0.08)
        self.near_depth_thresh: float = cfg.get("near_depth_threshold", 0.30)
        self.free_prob_thresh : float = cfg.get("free_probability_threshold", 0.55)

        # Wall/dustbin heuristic config
        wall_cfg = cfg.get("wall_detection", {})
        self.wall_enabled:         bool  = wall_cfg.get("enabled", True)
        self.wall_near_thresh:     float = wall_cfg.get("near_depth_wall_thresh", 0.35)
        self.wall_min_region_frac: float = wall_cfg.get("min_wall_region_frac", 0.04)
        self.wall_max_gradient:    float = wall_cfg.get("max_wall_gradient", 0.04)

        self._device = None
        self._torch_available = False
        self._init_device()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        depth_result,          # DepthResult
        detections: List,      # List[ObjectState]
    ) -> FreeSpaceResult:
        """
        Estimate free space on GPU.
        Returns FreeSpaceResult with CPU numpy arrays.
        """
        if self._torch_available and self._device is not None:
            return self._estimate_gpu(frame, depth_result, detections)
        return self._estimate_cpu(frame, depth_result, detections)

    def get_column_freespace(
        self,
        label_map: np.ndarray,
        n_cols: int = 20,
    ) -> List[float]:
        H, W = label_map.shape
        col_w = W // n_cols
        return [
            float(np.mean(label_map[:, c * col_w:(c + 1) * col_w] == LABEL_FREE))
            for c in range(n_cols)
        ]

    # ------------------------------------------------------------------
    # GPU path (torch tensors on CUDA)
    # ------------------------------------------------------------------

    def _estimate_gpu(self, frame, depth_result, detections) -> FreeSpaceResult:
        import torch

        H, W = frame.shape[:2]
        dev   = self._device

        # ── 1. Geometric prior ─────────────────────────────────────────
        free_prob = torch.zeros((H, W), dtype=torch.float32, device=dev)
        walkable_top = int(H * (1.0 - self.walkable_frac))
        free_prob[walkable_top:, :] = 0.60
        free_prob[:walkable_top, :] = 0.20

        # ── 2. Depth evidence ──────────────────────────────────────────
        if depth_result.is_valid:
            import torch.nn.functional as F
            # Upload depth to GPU once
            dm = torch.from_numpy(depth_result.depth_map).to(dev)  # (H,W) float32

            # Guard: portrait video can produce (W,H) depth map — resize to (H,W)
            if dm.shape[0] != H or dm.shape[1] != W:
                dm = F.interpolate(
                    dm.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W), mode="bilinear", align_corners=False
                ).squeeze()

            # Far pixels → more free
            far_mask  = (dm > (1.0 - self.near_depth_thresh))
            near_mask = (dm < self.near_depth_thresh)
            free_prob += far_mask.float()  * 0.25

            # Floor protection: near-depth on the floor directly underfoot is normal geometry,
            # not an obstacle. Only penalize near-depth above the floor base (upper 70% of frame).
            non_floor = torch.ones((H, W), dtype=torch.bool, device=dev)
            floor_cutoff_y = int(H * 0.68)
            non_floor[floor_cutoff_y:, :] = False
            free_prob -= (near_mask & non_floor).float() * 0.35

            # Depth gradient (edge detection for obstacles)
            gy, gx = torch.gradient(dm)
            grad_mag = gx.abs() + gy.abs()
            high_grad = (grad_mag > self.depth_grad_thresh)
            free_prob -= high_grad.float() * 0.20

        # ── 2b. Large flat obstacle heuristic (wall / unclassified bin fallback) ───
        obstacle_mask = torch.zeros((H, W), dtype=torch.bool, device=dev)
        if self.wall_enabled and depth_result.is_valid:
            obstacle_mask = self._detect_large_flat_obstacles(
                dm, grad_mag, obstacle_mask, H, W, dev, floor_cutoff_y=floor_cutoff_y
            )

        # ── 3. YOLO obstacle mask ──────────────────────────────────────
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bw = x2 - x1;  bh = y2 - y1
            pad_x = int(bw * (self.obstacle_inflation - 1.0) / 2)
            pad_y = int(bh * (self.obstacle_inflation - 1.0) / 2)
            ix1 = max(0, x1 - pad_x);  iy1 = max(0, y1 - pad_y)
            ix2 = min(W - 1, x2 + pad_x); iy2 = min(H - 1, y2 + pad_y)
            obstacle_mask[iy1:iy2, ix1:ix2] = True

        # Force obstacle prob to -1 (will become OBSTACLE label)
        free_prob[obstacle_mask] = -1.0

        # ── 4. Label assignment ────────────────────────────────────────
        clipped = free_prob.clamp(0.0, 1.0)               # [0,1] for non-obstacle
        label   = torch.full((H, W), LABEL_UNKNOWN,
                             dtype=torch.uint8, device=dev)
        label[clipped >= self.free_prob_thresh] = LABEL_FREE
        label[obstacle_mask]                    = LABEL_OBSTACLE

        # ── 5. Statistics on GPU ───────────────────────────────────────
        total = H * W
        free_count = int((label == LABEL_FREE).sum().item())
        obs_count  = int((label == LABEL_OBSTACLE).sum().item())
        unk_count  = total - free_count - obs_count
        uncertainty = unk_count / total

        # ── 6. Single transfer to CPU ──────────────────────────────────
        label_np   = label.cpu().numpy()
        prob_np    = clipped.cpu().numpy()

        return FreeSpaceResult(
            label_map=label_np,
            free_prob_map=prob_np,
            uncertainty=float(uncertainty),
            statistics={
                "free_fraction":     round(free_count / total, 4),
                "obstacle_fraction": round(obs_count  / total, 4),
                "unknown_fraction":  round(unk_count  / total, 4),
                "uncertainty":       round(uncertainty, 4),
            },
        )

    # ------------------------------------------------------------------
    # Wall / large static obstacle heuristic (GPU-side)
    # ------------------------------------------------------------------

    def _detect_large_flat_obstacles(
        self,
        dm,             # (H,W) float32 GPU tensor — depth map
        grad_mag,       # (H,W) float32 GPU tensor — depth gradient magnitude
        existing_mask,  # (H,W) bool GPU tensor — already-detected obstacles
        H: int, W: int,
        dev,
        floor_cutoff_y: Optional[int] = None,
    ):
        """
        Detect large flat near-depth regions (walls, unclassified dustbins)
        and add them to the obstacle mask.
        Walkable ground floor is excluded via floor_cutoff_y.
        """
        import torch
        import torch.nn.functional as F

        near_mask = (dm < self.wall_near_thresh)                    # near pixels
        flat_mask = (grad_mag < self.wall_max_gradient)             # flat/uniform
        candidate  = (near_mask & flat_mask).float()                # (H, W)

        # Exclude floor plane — smooth flat floor directly in front of camera
        # must never be flagged as a vertical wall obstacle!
        if floor_cutoff_y is not None:
            candidate[floor_cutoff_y:, :] = 0.0
        else:
            candidate[int(H * 0.68):, :] = 0.0

        region_frac = candidate.mean().item()
        if region_frac < self.wall_min_region_frac:
            return existing_mask  # too small — not a wall

        # Morphological dilation: expand candidate region by 15px kernel
        dilated = F.max_pool2d(
            candidate.unsqueeze(0).unsqueeze(0),
            kernel_size=15, stride=1, padding=7
        ).squeeze()
        wall_mask = (dilated > 0.5)

        logger.debug(
            "Wall heuristic: near_frac=%.3f, flat_frac=%.3f, wall_region=%.3f",
            near_mask.float().mean().item(),
            flat_mask.float().mean().item(),
            region_frac,
        )

        return existing_mask | wall_mask

    # ------------------------------------------------------------------
    # CPU fallback (same logic, numpy)
    # ------------------------------------------------------------------

    def _estimate_cpu(self, frame, depth_result, detections) -> FreeSpaceResult:
        H, W = frame.shape[:2]
        free_prob = np.zeros((H, W), dtype=np.float32)

        walkable_top = int(H * (1.0 - self.walkable_frac))
        free_prob[walkable_top:, :] += 0.60
        free_prob[:walkable_top, :] += 0.20

        if depth_result.is_valid:
            dm = depth_result.depth_map
            free_prob += (dm > (1.0 - self.near_depth_thresh)).astype(np.float32) * 0.25
            
            # Floor protection
            non_floor = np.ones((H, W), dtype=bool)
            non_floor[int(H * 0.68):, :] = False
            free_prob -= (dm < self.near_depth_thresh).astype(np.float32) * non_floor.astype(np.float32) * 0.35
            gx = np.abs(np.gradient(dm, axis=1))
            gy = np.abs(np.gradient(dm, axis=0))
            free_prob -= ((gx + gy) > self.depth_grad_thresh).astype(np.float32) * 0.20

        obstacle_mask = np.zeros((H, W), dtype=bool)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bw = x2 - x1;  bh = y2 - y1
            pad_x = int(bw * (self.obstacle_inflation - 1.0) / 2)
            pad_y = int(bh * (self.obstacle_inflation - 1.0) / 2)
            ix1 = max(0, x1 - pad_x);  iy1 = max(0, y1 - pad_y)
            ix2 = min(W - 1, x2 + pad_x); iy2 = min(H - 1, y2 + pad_y)
            obstacle_mask[iy1:iy2, ix1:ix2] = True

        free_prob[obstacle_mask] = -1.0
        clipped = np.clip(free_prob, 0.0, 1.0)
        label   = np.full((H, W), LABEL_UNKNOWN, dtype=np.uint8)
        label[clipped >= self.free_prob_thresh] = LABEL_FREE
        label[obstacle_mask]                    = LABEL_OBSTACLE

        total      = H * W
        free_count = int(np.sum(label == LABEL_FREE))
        obs_count  = int(np.sum(label == LABEL_OBSTACLE))
        unk_count  = total - free_count - obs_count
        uncertainty = unk_count / total

        return FreeSpaceResult(
            label_map=label,
            free_prob_map=clipped,
            uncertainty=float(uncertainty),
            statistics={
                "free_fraction":     round(free_count / total, 4),
                "obstacle_fraction": round(obs_count  / total, 4),
                "unknown_fraction":  round(unk_count  / total, 4),
                "uncertainty":       round(uncertainty, 4),
            },
        )

    # ------------------------------------------------------------------
    # Device init
    # ------------------------------------------------------------------

    def _init_device(self) -> None:
        try:
            import torch
            self._torch_available = True
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info("FreeSpaceEstimator: running on CUDA.")
            else:
                self._device = torch.device("cpu")
                logger.info("FreeSpaceEstimator: CUDA not available, using CPU.")
        except ImportError:
            logger.warning("torch not found — FreeSpaceEstimator will use numpy (CPU).")
            self._torch_available = False
