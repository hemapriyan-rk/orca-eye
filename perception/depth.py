"""
ORCA EYE — Stage 4a: Monocular Depth Estimation  (GPU-optimised)
=================================================================
MiDaS v2.1 small runs entirely on CUDA with FP16 autocast.

GPU optimisations:
  • torch.cuda.amp.autocast  — automatic FP16 for all ops inside inference
  • model on cuda, input tensor on cuda — zero CPU involvement during inference
  • Persistent pre-allocated GPU tensors for interpolation target
  • torch.no_grad()          — disables autograd engine (saves memory + time)
  • 3-pass warm-up           — JIT-compiles CUDA kernels before real data arrives

IMPORTANT: MiDaS output is RELATIVE depth — NOT metric.
           Always treat depth values as ordinal, not as metres.

Inputs  : BGR frame (np.ndarray)
Outputs : DepthResult(depth_map, statistics, is_valid)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class DepthResult:
    """
    Output of depth estimation for one frame.

    depth_map : np.ndarray float32 [H, W], range [0, 1]
                0.0 = closest (nearest), 1.0 = farthest
                Values are RELATIVE — not metric.
    """
    depth_map: np.ndarray       # (H, W) float32, [0, 1]
    raw_depth: np.ndarray       # unnormalised MiDaS output
    inference_ms: float
    statistics: Dict[str, float]
    is_valid: bool

    @property
    def depth_normalized(self) -> np.ndarray:
        return self.depth_map


@dataclass
class ObjectDepthEstimate:
    """
    Robust relative distance estimation for an object ROI.
    0.0 = closest to camera, 1.0 = farthest.
    Values are explicitly RELATIVE, NOT metric meters.
    """
    estimated_distance: float       # relative depth in [0.0, 1.0] (0=nearest)
    distance_confidence: float     # confidence in [0.0, 1.0] from ROI depth variance
    depth_valid: bool              # True if ROI contained valid finite depth pixels
    is_relative: bool = True       # flag distinguishing from metric distance
    raw_depth_median: float = 0.0


# ---------------------------------------------------------------------------
# GPU-Optimised Depth Estimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    """
    MiDaS v2.1 small — full FP16 inference on CUDA.

    Configuration keys (from cfg['depth']):
      model      : "MiDaS_small" | "DPT_Large" | "DPT_Hybrid"
      device     : "cuda"
      half       : true   — enable FP16 autocast
      normalize  : true
      enabled    : true
    """

    _TRANSFORM_MAP = {
        "MiDaS_small": "small",
        "DPT_Large":   "dpt_large",
        "DPT_Hybrid":  "dpt_hybrid",
    }

    def __init__(self, cfg: dict) -> None:
        self.model_name: str  = cfg.get("model", "MiDaS_small")
        self.device_str: str  = cfg.get("device", "cuda")
        self.half: bool       = cfg.get("half", True)
        self.normalize: bool  = cfg.get("normalize", True)
        self.enabled: bool    = cfg.get("enabled", True)
        self.skip_frames: int = cfg.get("skip_frames", 3)
        # Phase 1: Robust ROI depth parameters
        self.roi_inner_ratio: float = cfg.get("roi_inner_ratio", 0.60)
        self.outlier_p_low: float = cfg.get("outlier_percentile_low", 10.0)
        self.outlier_p_high: float = cfg.get("outlier_percentile_high", 90.0)

        self._model: Optional[Any]    = None
        self._transform: Optional[Any] = None
        self._device = None
        self._use_amp: bool = False   # whether torch.amp.autocast is available
        self._last_result: Optional[DepthResult] = None
        self._call_count: int = 0

        # Pre-allocated target output shape (set on first frame)
        self._out_h: int = 0
        self._out_w: int = 0

        if self.enabled:
            self._load_model()
        else:
            logger.info("Depth estimation disabled via config.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray) -> DepthResult:
        """
        Estimate relative depth for a single BGR frame.
        All heavy ops stay on GPU; only the final numpy array comes back to CPU.
        Caches and reuses depth maps according to skip_frames to maintain high FPS.

        Parameters
        ----------
        frame : np.ndarray  BGR uint8 H×W×3

        Returns
        -------
        DepthResult
        """
        if not self.enabled or self._model is None:
            h, w = frame.shape[:2]
            dummy = np.zeros((h, w), dtype=np.float32)
            return DepthResult(
                depth_map=dummy, raw_depth=dummy,
                inference_ms=0.0,
                statistics=self._compute_stats(dummy),
                is_valid=False,
            )

        # Depth changes slowly relative to walking speed; reuse previous frame if skipping
        if self._last_result is not None and self.skip_frames > 1:
            if (self._call_count % self.skip_frames) != 0:
                self._call_count += 1
                return self._last_result

        self._call_count += 1

        import torch

        # Update output shape if needed
        h, w = frame.shape[:2]
        if h != self._out_h or w != self._out_w:
            self._out_h, self._out_w = h, w

        t0 = time.perf_counter()

        # BGR → RGB on CPU (fast); tensor move to GPU
        rgb = frame[:, :, ::-1].copy()            # BGR→RGB, contiguous
        input_batch = self._transform(rgb).to(self._device)  # (1,3,H,W) on GPU

        # FP16 autocast: all matmuls/convs run in FP16 automatically
        ctx = torch.amp.autocast('cuda') if self._use_amp else torch.no_grad()

        with torch.no_grad():
            with ctx if self._use_amp else torch.no_grad():
                raw_pred = self._model(input_batch)           # stays on GPU
                raw_pred = torch.nn.functional.interpolate(
                    raw_pred.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()                                   # (H, W) on GPU

        # Single .cpu() call — only output array crosses PCIe
        raw_np = raw_pred.float().cpu().numpy()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Normalise: MiDaS raw = higher→closer (inverse depth)
        # We invert so 0=nearest, 1=farthest (intuitive)
        if self.normalize:
            d_min, d_max = raw_np.min(), raw_np.max()
            if d_max - d_min > 1e-6:
                norm = (raw_np - d_min) / (d_max - d_min)
                depth_map = (1.0 - norm).astype(np.float32)
            else:
                depth_map = np.zeros_like(raw_np, dtype=np.float32)
        else:
            depth_map = raw_np.astype(np.float32)

        stats = self._compute_stats(depth_map)
        stats["inference_ms"] = round(inference_ms, 2)
        stats["is_relative"]  = True     # explicit: values are NOT metric

        result = DepthResult(
            depth_map=depth_map,
            raw_depth=raw_np,
            inference_ms=inference_ms,
            statistics=stats,
            is_valid=True,
        )
        self._last_result = result
        return result

    def extract_object_depth(self, depth_map: np.ndarray, bbox: List[int]) -> ObjectDepthEstimate:
        """
        Extract robust depth estimate for an object bounding box.
        Selects inner ROI (center 60% width, lower 50% height), rejects outlier percentiles,
        and computes trimmed median. Values are explicitly relative (0=nearest, 1=farthest).
        """
        if depth_map is None or depth_map.size == 0 or len(bbox) != 4:
            return ObjectDepthEstimate(1.0, 0.0, False)

        x1, y1, x2, y2 = bbox
        H, W = depth_map.shape[:2]

        # Clamp to bounds
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            return ObjectDepthEstimate(1.0, 0.0, False)

        # Focus on inner ROI (middle 60% horizontally, lower 60% vertically toward ground contact)
        margin_x = int(bw * ((1.0 - self.roi_inner_ratio) / 2.0))
        roi_x1 = x1 + margin_x
        roi_x2 = max(roi_x1 + 2, x2 - margin_x)
        roi_y1 = y1 + int(bh * 0.25)
        roi_y2 = max(roi_y1 + 2, y2 - int(bh * 0.05))

        roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size < 4:
            roi = depth_map[y1:y2, x1:x2]

        vals = roi.ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 4:
            return ObjectDepthEstimate(1.0, 0.0, False)

        # Outlier rejection (percentile trimming)
        p_low = np.percentile(vals, self.outlier_p_low)
        p_high = np.percentile(vals, self.outlier_p_high)
        trimmed = vals[(vals >= p_low) & (vals <= p_high)]
        if len(trimmed) == 0:
            trimmed = vals

        med = float(np.median(trimmed))
        std = float(np.std(trimmed))
        # Distance confidence is inversely related to depth variance inside ROI
        conf = float(np.clip(1.0 - 2.0 * std, 0.10, 1.0))

        return ObjectDepthEstimate(
            estimated_distance=float(np.clip(med, 0.0, 1.0)),
            distance_confidence=conf,
            depth_valid=True,
            is_relative=True,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import torch

        logger.info(
            "Loading MiDaS: %s  device=%s  half=%s",
            self.model_name, self.device_str, self.half,
        )
        t0 = time.perf_counter()

        # Resolve device — fall back to CPU gracefully
        if self.device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU for depth.")
            self.device_str = "cpu"
            self.half = False

        self._device = torch.device(self.device_str)

        # Check if AMP autocast is supported (CUDA only)
        self._use_amp = (self.half and self._device.type == "cuda")

        try:
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_name,
                trust_repo=True,
            )
            self._model.to(self._device)

            # Convert model weights to FP16 when using autocast
            # (autocast handles activation dtype; weights in FP16 saves VRAM)
            if self._use_amp:
                self._model.half()
                logger.info("MiDaS weights converted to FP16.")

            self._model.eval()

            # Load matching transform
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True,
            )
            tk = self._TRANSFORM_MAP.get(self.model_name, "small")
            if tk == "dpt_large" or tk == "dpt_hybrid":
                self._transform = transforms.dpt_transform
            else:
                self._transform = transforms.small_transform

        except Exception as exc:
            logger.error("MiDaS load failed: %s", exc)
            logger.warning("Depth estimation disabled for this session.")
            self.enabled = False
            self._model = None
            return

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("MiDaS loaded in %.0f ms on %s", elapsed, self._device)

        # Warm-up: force CUDA kernel compilation before real frames arrive
        logger.info("Warming up MiDaS on GPU …")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.estimate(dummy)
        logger.info("MiDaS warm-up complete.")

    @staticmethod
    def _compute_stats(depth_map: np.ndarray) -> Dict[str, float]:
        if depth_map.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "uncertainty": 1.0}
        rng = float(depth_map.max() - depth_map.min())
        return {
            "min": float(depth_map.min()),
            "max": float(depth_map.max()),
            "mean": float(depth_map.mean()),
            "std": float(depth_map.std()),
            "uncertainty": float(depth_map.std() / (rng + 1e-6)),
        }
