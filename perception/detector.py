"""
ORCA EYE — Stage 2: YOLO Object Detection  (Phase 1 Enhanced)
============================================================
All inference runs on CUDA with FP16 (half precision).
Features:
  - Robust Detection dataclass with bottom-center ground contact points
  - Class-specific confidence thresholds
  - Minimum/maximum bounding box dimension validation
  - Non-maximum duplicate suppression
  - Temporal detection confirmation states (NEW -> TENTATIVE -> CONFIRMED)
  - Backward compatibility with ObjectState contracts

Inputs  : BGR frame (np.ndarray)
Outputs : DetectionResult containing List[Detection]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract — Detection & ObjectState (canonical cross-module contract)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Single detected object with explicit ground contact point and temporal confirmation.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]                                  # [x1, y1, x2, y2]
    center: List[int] = field(default_factory=list)  # [cx, cy]
    area: float = 0.0
    frame_id: int = 0
    track_id: Optional[int] = None
    # Phase 1 additions:
    bbox_width: int = 0
    bbox_height: int = 0
    center_x: int = 0
    center_y: int = 0
    bottom_center_x: int = 0
    bottom_center_y: int = 0
    bottom_center: List[int] = field(default_factory=list)  # [bcx, bcy]
    timestamp: float = 0.0
    state: str = "CONFIRMED"                         # "NEW" | "TENTATIVE" | "CONFIRMED"

    def __post_init__(self) -> None:
        if self.bbox and len(self.bbox) == 4:
            x1, y1, x2, y2 = self.bbox
            if not self.center:
                self.center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            if self.area == 0.0:
                self.area = float((x2 - x1) * (y2 - y1))
            self.bbox_width = int(x2 - x1)
            self.bbox_height = int(y2 - y1)
            self.center_x = self.center[0]
            self.center_y = self.center[1]
            self.bottom_center_x = int((x1 + x2) / 2)
            self.bottom_center_y = int(y2)
            self.bottom_center = [self.bottom_center_x, self.bottom_center_y]
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# Backward-compatibility alias
ObjectState = Detection


@dataclass
class DetectionResult:
    """Metadata envelope for a single detection pass."""
    frame_id: int
    timestamp: float
    inference_ms: float
    num_detections: int
    objects: List[Detection]
    mean_confidence: float = 0.0
    is_low_confidence: bool = False


# ---------------------------------------------------------------------------
# GPU-Optimised YOLO Detector
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    YOLOv8 detector running in FP16 on CUDA with robust filtering.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name: str  = cfg.get("model", "yolov8n.pt")
        self.conf_thresh: float = cfg.get("confidence_threshold", 0.40)
        self.nms_thresh: float  = cfg.get("nms_threshold", 0.45)
        self.device_str: str    = cfg.get("device", "cuda")
        self.half: bool         = cfg.get("half", True)
        self.imgsz: int         = cfg.get("imgsz", 320)
        self.target_classes     = cfg.get("target_classes", None)
        self.low_conf_thresh: float = cfg.get("low_confidence_threshold", 0.25)

        # Phase 1 parameters
        self.class_conf_thresholds: Dict[int, float] = cfg.get("class_confidence_thresholds", {})
        self.min_box_dim: int = cfg.get("min_box_dim", 12)
        self.max_box_dim: int = cfg.get("max_box_dim", 640)
        self.confirmation_frames: int = cfg.get("confirmation_frames", 2)
        self.persistence_frames: int = cfg.get("persistence_frames", 3)
        self.duplicate_iou_thresh: float = cfg.get("duplicate_iou_threshold", 0.70)

        # Temporal confirmation state: history of recent detections
        self._recent_detections: List[Dict] = []
        self._frame_counter = 0
        self._model = None
        self._class_names: Dict[int, str] = {}

        self._apply_global_torch_settings()
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> DetectionResult:
        """
        Run FP16 YOLO inference on a single BGR frame with Phase 1 confirmation.
        """
        if frame_id is None:
            self._frame_counter += 1
            frame_id = self._frame_counter

        t0 = time.perf_counter()

        # Run model prediction
        results = self._model.predict(
            source=frame,
            conf=min(self.conf_thresh, 0.30),  # low base pass to let class thresholds filter
            iou=self.nms_thresh,
            imgsz=self.imgsz,
            device=self.device_str,
            half=self.half,
            verbose=False,
            classes=self.target_classes,
            stream=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000.0

        raw_detections: List[Detection] = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            confs      = result.boxes.conf.cpu().numpy()
            cls_ids    = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # 1. Box dimension check
                if w < self.min_box_dim or h < self.min_box_dim:
                    continue
                if w > self.max_box_dim and h > self.max_box_dim:
                    continue

                # 2. Class-specific or global confidence check
                req_conf = self.class_conf_thresholds.get(int(cls_id), self.conf_thresh)
                if conf < req_conf:
                    continue

                det = Detection(
                    class_id=int(cls_id),
                    class_name=self._class_names.get(cls_id, f"cls_{cls_id}"),
                    confidence=float(conf),
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    frame_id=frame_id,
                )
                raw_detections.append(det)

        # 3. Duplicate suppression (NMS on overlapping detections)
        filtered_detections = self._suppress_duplicates(raw_detections, self.duplicate_iou_thresh)

        # 4. Temporal confirmation tracking
        confirmed_detections = self._apply_temporal_confirmation(filtered_detections, frame_id)

        mean_conf = float(np.mean([o.confidence for o in confirmed_detections])) if confirmed_detections else 0.0
        is_low = (mean_conf < self.low_conf_thresh and len(confirmed_detections) > 0)

        return DetectionResult(
            frame_id=frame_id,
            timestamp=time.time(),
            inference_ms=round(inference_ms, 2),
            num_detections=len(confirmed_detections),
            objects=confirmed_detections,
            mean_confidence=mean_conf,
            is_low_confidence=is_low,
        )

    def get_class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    def to_dict(self, result: DetectionResult) -> dict:
        return {
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "inference_ms": round(result.inference_ms, 2),
            "num_detections": result.num_detections,
            "mean_confidence": round(result.mean_confidence, 4),
            "is_low_confidence": result.is_low_confidence,
            "objects": [
                {
                    "class_id": o.class_id,
                    "class_name": o.class_name,
                    "confidence": round(o.confidence, 4),
                    "bbox": o.bbox,
                    "center": o.center,
                    "bottom_center": o.bottom_center,
                    "state": o.state,
                    "area": round(o.area, 1),
                    "track_id": o.track_id,
                }
                for o in result.objects
            ],
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        unionArea = areaA + areaB - interArea

        return interArea / unionArea if unionArea > 0 else 0.0

    def _suppress_duplicates(self, detections: List[Detection], iou_thresh: float) -> List[Detection]:
        """Suppress overlapping duplicate detections, preserving higher confidence."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []

        for det in sorted_dets:
            duplicate = False
            for k in kept:
                if self._compute_iou(det.bbox, k.bbox) > iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept

    def _apply_temporal_confirmation(self, detections: List[Detection], frame_id: int) -> List[Detection]:
        """
        Track detection candidate consistency across frames.
        Transitions: NEW -> TENTATIVE -> CONFIRMED.
        """
        if self.confirmation_frames <= 1:
            for d in detections:
                d.state = "CONFIRMED"
            return detections

        updated_history: List[Dict] = []
        output: List[Detection] = []

        for det in detections:
            matched_hist = None
            best_iou = 0.30

            for h in self._recent_detections:
                iou = self._compute_iou(det.bbox, h["bbox"])
                if iou > best_iou and det.class_id == h["class_id"]:
                    best_iou = iou
                    matched_hist = h

            if matched_hist is not None:
                streak = matched_hist["streak"] + 1
                det.state = "CONFIRMED" if streak >= self.confirmation_frames else "TENTATIVE"
                updated_history.append({
                    "bbox": det.bbox, "class_id": det.class_id,
                    "streak": streak, "last_frame": frame_id,
                })
            else:
                det.state = "NEW"
                updated_history.append({
                    "bbox": det.bbox, "class_id": det.class_id,
                    "streak": 1, "last_frame": frame_id,
                })

            output.append(det)

        self._recent_detections = updated_history
        return output

    @staticmethod
    def _apply_global_torch_settings() -> None:
        """Set cuDNN flags once for the process."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _load_model(self) -> None:
        import torch
        from ultralytics import YOLO

        t0 = time.perf_counter()
        logger.info("Loading YOLO model: %s on %s (half=%s, imgsz=%d)",
                    self.model_name, self.device_str, self.half, self.imgsz)

        self._model = YOLO(self.model_name)

        if self.device_str == "cuda" and torch.cuda.is_available():
            try:
                self._model.fuse()
            except Exception as exc:
                logger.debug("model.fuse() skipped: %s", exc)

        load_ms = (time.perf_counter() - t0) * 1000.0

        if hasattr(self._model, "names") and self._model.names:
            self._class_names = {int(k): str(v) for k, v in self._model.names.items()}

        logger.info("YOLO model loaded in %.1f ms | %d classes",
                    load_ms, len(self._class_names))

        self._warmup()

    def _warmup(self) -> None:
        """Warm up CUDA kernels."""
        if self.device_str != "cuda":
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(3):
                self._model.predict(
                    source=dummy,
                    conf=self.conf_thresh,
                    imgsz=self.imgsz,
                    device=self.device_str,
                    half=self.half,
                    verbose=False,
                )
            torch.cuda.synchronize()
            logger.info("YOLO CUDA warm-up complete.")
        except Exception as exc:
            logger.warning("YOLO warm-up failed: %s", exc)
