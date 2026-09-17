"""
ORCA EYE — Stage 2: YOLO Object Detection  (OIV7 + TensorRT)
=============================================================
Runs YOLOv8n trained on Open Images V7 (601 classes) for navigation-aware
obstacle detection including: Wall, Waste container (dustbin), Door, Stairs,
Handrail — classes absent from COCO-80.

TensorRT Pipeline
-----------------
First run  : exports .pt → .engine (FP16, fixed imgsz) — one-time ~60s
Subsequent : loads .engine directly → ~3–4× faster than PyTorch FP16

Class Filtering
---------------
target_classes=null  → auto-resolve navigation-relevant OIV7 class IDs
                        from model.names using nav_class_names substrings
target_classes=[...] → explicit integer list (legacy / override)

Inputs  : BGR frame (np.ndarray)
Outputs : DetectionResult containing List[Detection]
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract — Detection & ObjectState
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected object with ground contact point and temporal state."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]                                   # [x1, y1, x2, y2]
    center: List[int] = field(default_factory=list)   # [cx, cy]
    area: float = 0.0
    frame_id: int = 0
    track_id: Optional[int] = None
    bbox_width: int = 0
    bbox_height: int = 0
    center_x: int = 0
    center_y: int = 0
    bottom_center_x: int = 0
    bottom_center_y: int = 0
    bottom_center: List[int] = field(default_factory=list)
    timestamp: float = 0.0
    state: str = "CONFIRMED"          # "NEW" | "TENTATIVE" | "CONFIRMED"

    def __post_init__(self) -> None:
        if self.bbox and len(self.bbox) == 4:
            x1, y1, x2, y2 = self.bbox
            if not self.center:
                self.center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            if self.area == 0.0:
                self.area = float((x2 - x1) * (y2 - y1))
            self.bbox_width  = int(x2 - x1)
            self.bbox_height = int(y2 - y1)
            self.center_x    = self.center[0]
            self.center_y    = self.center[1]
            self.bottom_center_x = int((x1 + x2) / 2)
            self.bottom_center_y = int(y2)
            self.bottom_center   = [self.bottom_center_x, self.bottom_center_y]
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
# GPU-Optimised YOLO Detector  (OIV7 + TensorRT)
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    YOLOv8 detector on Open Images V7 (601 classes) with TensorRT export.

    First-run: auto-exports yolov8n-oiv7.pt → yolov8n-oiv7.engine (FP16).
    Subsequent runs: loads .engine directly for maximum throughput.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name: str        = cfg.get("model", "yolov8n-oiv7.pt")
        self.engine_path: str       = cfg.get("engine_path", "yolov8n-oiv7.engine")
        self.use_tensorrt: bool     = cfg.get("use_tensorrt", True)
        self.conf_thresh: float     = cfg.get("confidence_threshold", 0.35)
        self.nms_thresh: float      = cfg.get("nms_threshold", 0.45)
        self.device_str: str        = cfg.get("device", "cuda")
        self.half: bool             = cfg.get("half", True)
        self.imgsz: int             = cfg.get("imgsz", 320)
        self.low_conf_thresh: float = cfg.get("low_confidence_threshold", 0.25)
        self.min_box_dim: int       = cfg.get("min_box_dim", 12)
        self.max_box_dim: int       = cfg.get("max_box_dim", 640)
        self.confirmation_frames: int = cfg.get("confirmation_frames", 1)
        self.persistence_frames: int  = cfg.get("persistence_frames", 3)
        self.duplicate_iou_thresh: float = cfg.get("duplicate_iou_threshold", 0.70)

        # Nav class filter (resolved from model.names after load)
        self._nav_class_names: List[str] = [
            s.lower() for s in cfg.get("nav_class_names", [])
        ]
        self._target_classes: Optional[List[int]] = cfg.get("target_classes", None)
        self._resolved_classes: Optional[List[int]] = None  # set after model load

        # Per-class confidence thresholds (by class name → id resolved post-load)
        self._conf_by_name: Dict[str, float] = {
            k.lower(): float(v)
            for k, v in cfg.get("class_conf_by_name", {}).items()
        }
        self._class_conf_thresholds: Dict[int, float] = {}  # id → thresh (filled post-load)

        self._recent_detections: List[Dict] = []
        self._frame_counter = 0
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._using_engine = False

        self._apply_global_torch_settings()
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> DetectionResult:
        """Run FP16 / TRT inference on a single BGR frame."""
        if frame_id is None:
            self._frame_counter += 1
            frame_id = self._frame_counter

        t0 = time.perf_counter()

        results = self._model.predict(
            source=frame,
            conf=min(self.conf_thresh, 0.25),   # low base; class thresholds filter later
            iou=self.nms_thresh,
            imgsz=self.imgsz,
            device=self.device_str,
            half=self.half and not self._using_engine,  # engine handles half internally
            verbose=False,
            classes=self._resolved_classes,
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
                w = x2 - x1;  h = y2 - y1

                if w < self.min_box_dim or h < self.min_box_dim:
                    continue
                if w > self.max_box_dim and h > self.max_box_dim:
                    continue

                req_conf = self._class_conf_thresholds.get(int(cls_id), self.conf_thresh)
                if conf < req_conf:
                    continue

                det = Detection(
                    class_id=int(cls_id),
                    class_name=self._class_names.get(int(cls_id), f"cls_{cls_id}"),
                    confidence=float(conf),
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    frame_id=frame_id,
                )
                raw_detections.append(det)

        filtered    = self._suppress_duplicates(raw_detections, self.duplicate_iou_thresh)
        confirmed   = self._apply_temporal_confirmation(filtered, frame_id)

        mean_conf = float(np.mean([o.confidence for o in confirmed])) if confirmed else 0.0
        is_low    = mean_conf < self.low_conf_thresh and len(confirmed) > 0

        return DetectionResult(
            frame_id=frame_id,
            timestamp=time.time(),
            inference_ms=round(inference_ms, 2),
            num_detections=len(confirmed),
            objects=confirmed,
            mean_confidence=mean_conf,
            is_low_confidence=is_low,
        )

    def get_class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    def to_dict(self, result: DetectionResult) -> dict:
        return {
            "frame_id":        result.frame_id,
            "timestamp":       result.timestamp,
            "inference_ms":    round(result.inference_ms, 2),
            "num_detections":  result.num_detections,
            "mean_confidence": round(result.mean_confidence, 4),
            "is_low_confidence": result.is_low_confidence,
            "using_tensorrt":  self._using_engine,
            "objects": [
                {
                    "class_id":    o.class_id,
                    "class_name":  o.class_name,
                    "confidence":  round(o.confidence, 4),
                    "bbox":        o.bbox,
                    "center":      o.center,
                    "bottom_center": o.bottom_center,
                    "state":       o.state,
                    "area":        round(o.area, 1),
                    "track_id":    o.track_id,
                }
                for o in result.objects
            ],
        }

    # ------------------------------------------------------------------
    # Internal — Model loading & TensorRT export
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import torch
        from ultralytics import YOLO

        t0 = time.perf_counter()
        engine_path = Path(self.engine_path)

        # ── Try TensorRT engine first ─────────────────────────────────
        if self.use_tensorrt and self.device_str == "cuda" and torch.cuda.is_available():
            if engine_path.exists():
                logger.info("Loading TensorRT engine: %s", engine_path)
                try:
                    self._model = YOLO(str(engine_path))
                    self._using_engine = True
                    logger.info("TensorRT engine loaded in %.1f ms", (time.perf_counter() - t0) * 1000)
                except Exception as exc:
                    logger.warning("TRT engine load failed (%s) — falling back to .pt", exc)
                    self._using_engine = False
                    self._model = None

        # ── Load .pt and export to TRT if needed ─────────────────────
        if self._model is None:
            logger.info("Loading YOLO model: %s (half=%s, imgsz=%d)",
                        self.model_name, self.half, self.imgsz)
            self._model = YOLO(self.model_name)

            if self.device_str == "cuda" and torch.cuda.is_available():
                try:
                    self._model.fuse()
                except Exception as exc:
                    logger.debug("model.fuse() skipped: %s", exc)

            load_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("YOLO .pt loaded in %.1f ms", load_ms)

            # Export to TensorRT (runs once, then .engine is reused)
            if self.use_tensorrt and self.device_str == "cuda" and torch.cuda.is_available():
                if not engine_path.exists():
                    logger.info(
                        "Exporting to TensorRT engine: %s  (one-time, ~60s) …", engine_path
                    )
                    try:
                        self._model.export(
                            format="engine",
                            half=True,
                            imgsz=self.imgsz,
                            device=0,
                        )
                        # Ultralytics saves engine next to the .pt file
                        candidate = Path(self.model_name).with_suffix(".engine")
                        if candidate.exists() and not engine_path.exists():
                            candidate.rename(engine_path)
                        if engine_path.exists():
                            logger.info("TensorRT engine saved: %s — reloading", engine_path)
                            self._model = YOLO(str(engine_path))
                            self._using_engine = True
                        else:
                            logger.warning("TRT engine not found after export — using PyTorch FP16")
                    except Exception as exc:
                        logger.warning("TRT export failed (%s) — using PyTorch FP16", exc)

        # ── Resolve class names ───────────────────────────────────────
        if hasattr(self._model, "names") and self._model.names:
            self._class_names = {int(k): str(v) for k, v in self._model.names.items()}

        logger.info("Detector ready | %d classes | TRT=%s",
                    len(self._class_names), self._using_engine)

        # ── Auto-resolve navigation class IDs from model.names ───────
        self._resolve_nav_classes()
        self._resolve_class_conf_thresholds()
        self._warmup()

    def _resolve_nav_classes(self) -> None:
        """
        Build self._resolved_classes: integer list of OIV7 class IDs matching
        nav_class_names substrings. If target_classes is explicitly set, use that.
        """
        if self._target_classes is not None:
            self._resolved_classes = self._target_classes
            logger.info("Detection: using explicit target_classes (%d ids)",
                        len(self._resolved_classes))
            return

        if not self._nav_class_names or not self._class_names:
            self._resolved_classes = None  # detect all
            logger.info("Detection: no nav_class_names filter — detecting all %d classes",
                        len(self._class_names))
            return

        resolved = []
        for cls_id, cls_name in self._class_names.items():
            cls_lower = cls_name.lower()
            for nav_substr in self._nav_class_names:
                # Use word-boundary matching: "car" must not match "carrot" or "carnivore"
                import re
                pattern = r'(?<![a-z])' + re.escape(nav_substr) + r'(?![a-z])'
                if re.search(pattern, cls_lower):
                    resolved.append(int(cls_id))
                    break

        if resolved:
            self._resolved_classes = sorted(resolved)
            nav_names = [self._class_names[i] for i in self._resolved_classes[:15]]
            logger.info("Detection: resolved %d navigation classes: %s%s",
                        len(self._resolved_classes), nav_names,
                        " ..." if len(self._resolved_classes) > 15 else "")
        else:
            self._resolved_classes = None
            logger.warning("No nav_class_names matched model.names — detecting all classes")

    def _resolve_class_conf_thresholds(self) -> None:
        """Map class-name-keyed confidence thresholds to class ID keys."""
        for cls_id, cls_name in self._class_names.items():
            cls_lower = cls_name.lower()
            for name_key, thresh in self._conf_by_name.items():
                if name_key in cls_lower:
                    self._class_conf_thresholds[int(cls_id)] = thresh
                    break

        logger.debug("Class-specific conf thresholds: %d entries",
                     len(self._class_conf_thresholds))

    # ------------------------------------------------------------------
    # Internal — Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA);    interH = max(0, yB - yA)
        interArea = interW * interH
        areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        unionArea = areaA + areaB - interArea
        return interArea / unionArea if unionArea > 0 else 0.0

    def _suppress_duplicates(self, detections: List[Detection], iou_thresh: float) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in sorted_dets:
            if not any(self._compute_iou(det.bbox, k.bbox) > iou_thresh for k in kept):
                kept.append(det)
        return kept

    def _apply_temporal_confirmation(
        self, detections: List[Detection], frame_id: int
    ) -> List[Detection]:
        """
        Lightweight temporal confirmation.
        confirmation_frames=1 → all detections pass immediately (fastest).
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

    # ------------------------------------------------------------------
    # Internal — GPU setup / warmup
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_global_torch_settings() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _warmup(self) -> None:
        """Warm up CUDA / TRT kernels with 3 dummy frames."""
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
                    half=self.half and not self._using_engine,
                    verbose=False,
                )
            logger.info("Detector warm-up complete (TRT=%s).", self._using_engine)
        except Exception as exc:
            logger.warning("Detector warm-up failed: %s", exc)
