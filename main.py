"""
ORCA EYE — Main Entry Point  (GPU-First)
=========================================
RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE

GPU Strategy
------------
• YOLO detection   : FP16 on CUDA stream A
• MiDaS depth      : FP16 autocast on CUDA stream B  (concurrent with YOLO)
• Free-space        : CUDA tensor ops
• Spatial map, path gen, scoring : CPU (lightweight, <1 ms)
• cudnn.benchmark = True from detector module startup

Usage
-----
  Webcam:           python main.py --source 0
  IP Webcam:        python main.py --source http://10.234.182.115:8080/video
  Video file:       python main.py --source path/to/video.mp4
  Evaluate:         python main.py --source video.mp4 --evaluate
  List scenarios:   python main.py --list-scenarios
  Benchmark:        python main.py --source 0 --benchmark
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orca_eye")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error("config.yaml not found at %s", cfg_path.resolve())
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", cfg_path.resolve())
    return cfg


# ---------------------------------------------------------------------------
# GPU warmup / setup
# ---------------------------------------------------------------------------

def setup_gpu(cfg: dict) -> None:
    """Apply process-wide GPU settings from config."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available — running on CPU.")
            return
        perf = cfg.get("performance", {})
        torch.backends.cudnn.benchmark     = perf.get("cudnn_benchmark", True)
        torch.backends.cudnn.deterministic = perf.get("cudnn_deterministic", False)
        torch.backends.cuda.matmul.allow_tf32 = True   # free TF32 on Ampere
        torch.backends.cudnn.allow_tf32       = True
        # Pre-warm the CUDA context (reduces first-frame jitter)
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        logger.info(
            "GPU: %s  VRAM: %.1f GB  cudnn.benchmark=%s  TF32=enabled",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            torch.backends.cudnn.benchmark,
        )
    except Exception as exc:
        logger.warning("GPU setup error: %s", exc)


# ---------------------------------------------------------------------------
# Parallel GPU inference helper
# ---------------------------------------------------------------------------

_GPU_STREAMS = None
_GPU_EXECUTOR = None


def _get_gpu_streams():
    global _GPU_STREAMS
    if _GPU_STREAMS is None:
        try:
            import torch
            if torch.cuda.is_available():
                _GPU_STREAMS = (torch.cuda.Stream(), torch.cuda.Stream())
        except Exception:
            _GPU_STREAMS = None
    return _GPU_STREAMS


def _get_gpu_executor():
    global _GPU_EXECUTOR
    if _GPU_EXECUTOR is None:
        import concurrent.futures
        _GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="orca_gpu_worker"
        )
    return _GPU_EXECUTOR


def _run_parallel_gpu_inference(detector, depth_estimator, frame):
    """
    Run YOLO and MiDaS truly concurrently on persistent CUDA streams via worker threads.

    Worker threads overlap CPU preparation and GPU kernel execution across
    independent CUDA streams. No explicit synchronize() — CUDA stream ordering
    and the .result() call provide the necessary memory fence.
    """
    streams = _get_gpu_streams()
    if streams is None:
        return detector.detect(frame), depth_estimator.estimate(frame)

    try:
        import torch
        stream_det, stream_depth = streams
        executor = _get_gpu_executor()

        def _run_det():
            with torch.cuda.stream(stream_det):
                return detector.detect(frame)

        def _run_depth():
            with torch.cuda.stream(stream_depth):
                return depth_estimator.estimate(frame)

        fut_det   = executor.submit(_run_det)
        fut_depth = executor.submit(_run_depth)

        # .result() blocks CPU until thread completes; no explicit synchronize needed
        det_result   = fut_det.result()
        depth_result = fut_depth.result()

        return det_result, depth_result

    except Exception:
        return detector.detect(frame), depth_estimator.estimate(frame)


# ---------------------------------------------------------------------------
# FPS printer
# ---------------------------------------------------------------------------

class _FPSStats:
    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: list = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(source, cfg: dict, evaluate: bool = False, max_frames: Optional[int] = None) -> None:
    """
    Full 13-stage ORCA EYE pipeline loop.
    YOLO + MiDaS run on parallel CUDA streams each frame.
    """
    from datetime import datetime

    from perception.camera    import CameraSource
    from perception.detector  import YOLODetector
    from perception.tracker   import CentroidTracker
    from perception.depth     import DepthEstimator
    from perception.freespace import FreeSpaceEstimator
    from perception.calibration import CameraCalibration
    from perception.geometry_3d import Geometry3D
    from navigation.spatial_map    import SpatialMap
    from navigation.path_generator import PathGenerator
    from navigation.path_scorer    import PathScorer
    from navigation.decision       import DecisionMaker
    from navigation.spatial_entity import create_spatial_entities
    from navigation.dynamic_conflict import DynamicConflictEngine
    from navigation.geometry       import EntityKinematics, calculate_bearing
    from navigation.critical_region import CriticalRegionExtractor
    from navigation.rho_fov        import RhoFOVPredictor
    from navigation.camera_support import PerceptualSupportGate
    from navigation.camera_responsibility import CameraResponsibilityManager
    from navigation.audio_guidance import InstructionGenerator
    from visualization.renderer    import Renderer
    from system_logging.logger     import SystemLogger
    from failure_analysis.detector import FailureDetector
    from evaluation.metrics        import MetricsCollector

    logger.info("=" * 60)
    logger.info("ORCA EYE — Stage A: Single-Camera Perceptual Support Pipeline")
    logger.info("RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE")
    logger.info("=" * 60)

    # ── Init all modules ─────────────────────────────────────────────
    camera  = CameraSource(source, cfg.get("camera", {}))
    W, H    = camera.get_resolution()
    logger.info("Camera: %dx%d", W, H)

    # Stage A: Physical Calibration & Geometry
    cam_cfg = cfg.get("camera", {})
    hfov_val = float(cam_cfg.get("hfov_deg", 65.0))
    camera_calib = CameraCalibration.from_fov(frame_width=W, frame_height=H, hfov_deg=hfov_val)
    logger.info("Calibrated Camera: %dx%d, HFOV=%.1f deg, fx=%.1f", W, H, camera_calib.hfov_deg, camera_calib.fx)

    geometry_3d_engine      = Geometry3D(camera_calib)
    detector                = YOLODetector(cfg.get("detection", {}))
    tracker                 = CentroidTracker(cfg.get("tracking", {}))
    depth_estimator         = DepthEstimator(cfg.get("depth", {}))
    fs_estimator            = FreeSpaceEstimator(cfg.get("freespace", {}))
    spatial_map             = SpatialMap(cfg.get("spatial_map", {}), frame_width=W, frame_height=H)
    path_generator          = PathGenerator(cfg.get("path_generation", {}), spatial_map.rows, spatial_map.cols)
    dynamic_conflict_engine = DynamicConflictEngine(cfg)
    scorer                  = PathScorer(cfg.get("scoring", {}))
    decision_maker          = DecisionMaker(cfg.get("safety", {}))
    audio_guidance          = InstructionGenerator(cfg)
    renderer                = Renderer(cfg, W, H)

    # Stage A: Research Navigation-Coupled Perceptual Support
    stage_a_cfg             = cfg.get("stage_a", {})
    rho_cfg                 = stage_a_cfg.get("rho_fov", {})
    safety_cfg              = cfg.get("safety", {})
    critical_extractor      = CriticalRegionExtractor(corridor_width_m=1.0, horizon_s=2.0, user_walk_speed_m_s=1.0)
    rho_predictor           = RhoFOVPredictor(
        half_hfov_rad=camera_calib.half_hfov_rad,
        q_min=float(rho_cfg.get("q_min", 0.20)),
        sigma_br=float(rho_cfg.get("sigma_br", 0.60)),
        num_samples=int(rho_cfg.get("num_samples", 200)),
        horizon_s=float(rho_cfg.get("horizon_s", 2.0)),
        dt_sample_s=float(rho_cfg.get("dt_sample_s", 0.20)),
        use_soft_quality=bool(rho_cfg.get("use_soft_quality", True)),
    )
    support_gate            = PerceptualSupportGate(
        tau_safe=float(safety_cfg.get("tau_safe", 0.35)),
        tau_cam=float(safety_cfg.get("tau_cam", 0.40)),
    )
    camera_manager          = CameraResponsibilityManager(primary_camera_id="PRIMARY_CAM")
    entity_kinematics_map   = {}  # track_id -> EntityKinematics

    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys_logger  = SystemLogger(cfg, session_id=session_id)
    fail_detect = FailureDetector(cfg)

    metrics = MetricsCollector(session_dir=sys_logger.get_session_dir()) \
              if evaluate else None

    fps_stats  = _FPSStats(window=30)
    frame_id   = 0

    # Depth skip: run MiDaS every `depth_skip` frames, reuse last result in between
    depth_skip       = int(cfg.get("depth", {}).get("skip_frames", 4))
    _last_depth      = None
    _depth_frame_ctr = 0

    sys_logger.log_event("session_start", {
        "source": str(source), "resolution": [W, H], "session_id": session_id,
    })
    logger.info("Pipeline running. Press 'q' to quit.")

    try:
        while True:
            t_frame = time.perf_counter()

            # ── Stage 1: Frame acquisition ───────────────────────────
            ok, frame, ts = camera.read()
            if not ok:
                logger.info("Frame acquisition ended.")
                break
            frame_id += 1
            if max_frames and frame_id > max_frames:
                logger.info("Reached max frames (%d). Exiting pipeline loop.", max_frames)
                break

            # ── Stages 2+4a: YOLO every frame, MiDaS every depth_skip frames ─
            t_gpu = time.perf_counter()
            _depth_frame_ctr += 1
            if _depth_frame_ctr % depth_skip == 0 or _last_depth is None:
                # Full parallel inference (YOLO + MiDaS)
                detection_result, depth_result = _run_parallel_gpu_inference(
                    detector, depth_estimator, frame,
                )
                _last_depth = depth_result
            else:
                # YOLO only (TRT — fast); reuse last depth result
                detection_result = detector.detect(frame)
                depth_result = _last_depth
            gpu_ms = (time.perf_counter() - t_gpu) * 1000.0

            # ── Stage 3: Tracker ─────────────────────────────────────
            tracks = tracker.update(detection_result.objects, frame_id=frame_id)
            for obj in detection_result.objects:
                for t in tracks:
                    if abs(t.center[0]-obj.center[0]) < 10 and \
                       abs(t.center[1]-obj.center[1]) < 10:
                        obj.track_id = t.track_id
                        break

            # ── Stage 4a: Unified Spatial Entities ───────────────────
            spatial_entities = create_spatial_entities(
                tracks=tracks,
                depth_map=depth_result.depth_normalized if depth_result else None,
                depth_estimator=depth_estimator,
                frame_w=W,
                frame_h=H,
            )

            # ── Stage 4c: 3D Egocentric Geometry & Corridor Analysis ──
            geometry_3d_result = None
            if depth_result.is_valid and depth_result.depth_map is not None:
                geometry_3d_result = geometry_3d_engine.analyze(
                    depth_result.depth_map, detection_result.objects
                )

            # ── Stage 4b: Free-space (CUDA tensors) ──────────────────
            freespace_result = fs_estimator.estimate(
                frame, depth_result, detection_result.objects
            )

            # ── Stage 5: Spatial map ──────────────────────────────────
            spatial_map.update(freespace_result, depth_result, tracks)
            wall_proximity = spatial_map.evaluate_wall_proximity(geometry_3d_result=geometry_3d_result)

            # ── Stage 6: Path generation ──────────────────────────────
            candidates = path_generator.generate(spatial_map, geometry_3d_result=geometry_3d_result)

            # ── Stage 6b: Dynamic conflict reasoning ─────────────────
            fps_estimate = fps_stats.tick()
            conflicts_by_dir = dynamic_conflict_engine.evaluate_conflicts(
                spatial_entities=spatial_entities,
                candidates=candidates,
                frame_w=W,
                frame_h=H,
                grid_rows=spatial_map.rows,
                grid_cols=spatial_map.cols,
                fps=fps_estimate if fps_estimate > 5.0 else 30.0,
            )
            active_conflicts = [c for c_list in conflicts_by_dir.values() for c in c_list]

            # ── Stage 7: Path scoring (with dynamic conflict penalties) ─
            candidates = scorer.score(candidates)

            # ── Stage 7b: Stage A Perceptual Support & Admissibility ──────
            now_ts = time.time()
            rho_predictions = {}
            for entity in spatial_entities:
                tid = entity.track_id
                # Derive metric egocentric ground coords (meters)
                dist_m = max(0.5, float(getattr(entity, "relative_distance", 0.5)) * 6.0)
                b_rad = math.radians(float(getattr(entity, "relative_bearing", 0.0)))
                gx = dist_m * math.sin(b_rad)
                gy = dist_m * math.cos(b_rad)

                if tid not in entity_kinematics_map:
                    kin = EntityKinematics(
                        track_id=tid, timestamp=now_ts, x=gx, y=gy, bearing_rad=b_rad
                    )
                    entity_kinematics_map[tid] = kin
                else:
                    kin = entity_kinematics_map[tid]
                    kin.update(
                        new_timestamp=now_ts, new_x=gx, new_y=gy,
                        half_hfov_rad=camera_calib.half_hfov_rad,
                    )

                pred = rho_predictor.predict_survival(
                    entity_id=tid,
                    bearing_rad=kin.bearing_rad,
                    bearing_rate=kin.bearing_rate,
                    camera_id="PRIMARY",
                    current_time=now_ts,
                )
                rho_predictions[tid] = pred

            # Extract Navigation-Critical Region S(G_k) & Critical Entities E_k
            critical_regions = critical_extractor.extract_critical_regions(
                candidates=candidates, spatial_entities=spatial_entities
            )

            # Evaluate Perceptual Support & Corridor Admissibility
            corridor_support = support_gate.evaluate_corridors(
                candidates=candidates,
                critical_regions=critical_regions,
                rho_predictions=rho_predictions,
            )

            # ── Stage 8: Decision (NavigationState with Admissibility Gate) ───
            decision = decision_maker.decide(
                candidates,
                frame_id=frame_id,
                spatial_entities=spatial_entities,
                dynamic_conflicts=active_conflicts,
                wall_proximity=wall_proximity,
                corridor_support=corridor_support,
                primary_camera_id="PRIMARY_CAM",
            )
            camera_manager.update_responsibilities(
                selected_corridor=decision.selected_path_direction,
                critical_entities=decision.critical_entities,
                support_score=decision.camera_support,
            )
            stability = decision_maker.get_stability_metrics()

            # ── Stage 8b: Audio guidance generation ───────────────────
            audio_instr = audio_guidance.generate_instruction(
                nav_state=decision,
                spatial_entities=spatial_entities,
                dynamic_conflicts=active_conflicts,
            )
            if audio_instr:
                audio_guidance.speak(audio_instr)

            # ── FPS ───────────────────────────────────────────────────
            fps = fps_estimate
            total_ms = (time.perf_counter() - t_frame) * 1000.0

            # ── Stage 9: Render ───────────────────────────────────────
            composite = renderer.render(
                frame=frame,
                detection_result=detection_result, tracks=tracks,
                freespace_result=freespace_result, depth_result=depth_result,
                spatial_map=spatial_map, candidates=candidates,
                decision=decision, fps=fps, frame_id=frame_id,
                latency_ms=total_ms,
                spatial_entities=spatial_entities,
                dynamic_conflicts=active_conflicts,
                wall_proximity=wall_proximity,
                geometry_3d_result=geometry_3d_result,
            )
            key = renderer.show(composite)
            if key == ord("q"):
                break

            # ── Stage 10: Log ─────────────────────────────────────────
            sys_logger.log_frame(
                frame_id=frame_id, fps=fps, latency_ms=total_ms,
                detection_result=detection_result, tracks=tracks,
                depth_result=depth_result, freespace_result=freespace_result,
                candidates=candidates, decision=decision,
                stability_metrics=stability,
                detector_serializer=detector.to_dict,
                tracker_serializer=tracker.to_dict_list,
                scorer_serializer=scorer.to_dict_list,
                decision_serializer=decision_maker.to_dict,
                extra={
                    "dynamic_conflicts": len(active_conflicts),
                    "audio_instruction": audio_instr.text if audio_instr else None,
                    "critical_entities": decision.critical_entities,
                    "camera_support": round(decision.camera_support, 4),
                    "admissibility_status": decision.admissibility_status,
                    "responsible_camera": decision.responsible_camera,
                },
            )

            # ── Stage 11: Failure detection ───────────────────────────
            fail_detect.check(
                frame_id=frame_id, frame=frame, decision=decision,
                detection_result=detection_result, tracks=tracks,
                freespace_result=freespace_result, depth_result=depth_result,
                stability_metrics=stability, candidates=candidates,
            )

            # ── Stage 12: Metrics ─────────────────────────────────────
            if metrics:
                metrics.update(
                    fps=fps, latency_ms=total_ms,
                    detection_result=detection_result,
                    freespace_result=freespace_result,
                    decision=decision, stability_metrics=stability,
                )

            # ── Console log every 30 frames ───────────────────────────
            if frame_id % 30 == 0:
                try:
                    import torch
                    vram_mb = torch.cuda.memory_allocated() / 1e6
                    vram_str = f"  VRAM: {vram_mb:.0f} MB"
                except Exception:
                    vram_str = ""
                logger.info(
                    "Frame %d | FPS: %.1f | GPU+depth: %.0fms | Total: %.0fms | "
                    "Det: %d | Cmd: %s%s",
                    frame_id, fps, gpu_ms, total_ms,
                    detection_result.num_detections,
                    decision.command, vram_str,
                )

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt.")
    finally:
        camera.release()
        renderer.destroy()
        sys_logger.log_event("session_end", {
            "frame_id": frame_id,
            "failure_summary": fail_detect.get_summary(),
        })
        sys_logger.close()
        if metrics:
            metrics.report()
        # Free CUDA cache
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Session done. %d frames. Failure summary: %s",
                    frame_id, fail_detect.get_summary())


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(source, cfg: dict, n_frames: int = 100) -> None:
    """Measure per-stage latency over n_frames and print a summary table."""
    from perception.camera   import CameraSource
    from perception.detector import YOLODetector
    from perception.depth    import DepthEstimator
    from perception.freespace import FreeSpaceEstimator

    camera = CameraSource(source, cfg.get("camera", {}))
    det    = YOLODetector(cfg.get("detection", {}))
    depth  = DepthEstimator(cfg.get("depth", {}))
    fs     = FreeSpaceEstimator(cfg.get("freespace", {}))

    det_ms_list  = []
    depth_ms_list = []
    fs_ms_list   = []
    total_ms_list = []

    logger.info("Benchmark: %d frames …", n_frames)
    for i in range(n_frames):
        ok, frame, _ = camera.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dr  = det.detect(frame);      det_ms_list.append(dr.inference_ms)
        t1 = time.perf_counter()
        dpr = depth.estimate(frame);  depth_ms_list.append((t1 - t0)*1000 - dr.inference_ms + dpr.inference_ms)
        t2 = time.perf_counter()
        fs.estimate(frame, dpr, dr.objects)
        fs_ms_list.append((time.perf_counter() - t2) * 1000.0)
        total_ms_list.append((time.perf_counter() - t0) * 1000.0)

    import statistics
    def stats(lst):
        return f"{statistics.mean(lst):.1f} ms  (min {min(lst):.1f}  max {max(lst):.1f})"

    print("\n=== ORCA EYE Benchmark ===")
    print(f"  YOLO detect  : {stats(det_ms_list)}")
    print(f"  MiDaS depth  : {stats(depth_ms_list)}")
    print(f"  Free-space   : {stats(fs_ms_list)}")
    print(f"  Total pipeline: {stats(total_ms_list)}")
    print(f"  Throughput   : {1000/statistics.mean(total_ms_list):.1f} FPS  "
          f"(theoretical)")
    camera.release()


# ---------------------------------------------------------------------------
# Stage B: Dual-Camera Pipeline
# ---------------------------------------------------------------------------

def run_dual_pipeline(
    source0,
    source1,
    cfg: dict,
    evaluate: bool = False,
    max_frames: Optional[int] = None,
    save_snapshot: Optional[str] = None,
    loop: bool = False,
) -> None:
    """
    Executes the Dual-Primary Camera Navigation Pipeline:
      CAM0: Left Primary camera (yaw -17.5 deg, FOV [-50 deg, +15 deg])
      CAM1: Right Primary camera (yaw +17.5 deg, FOV [-15 deg, +50 deg])
      Center Overlap: [-15 deg, +15 deg] centered at 0 deg (wearer heading)
    """
    from datetime import datetime
    from perception.calibration import CameraCalibration
    from perception.camera import CameraSource
    from perception.depth import DepthEstimator
    from perception.detector import YOLODetector
    from perception.tracker import CentroidTracker
    from perception.freespace import FreeSpaceEstimator
    from perception.geometry_3d import Geometry3D
    from navigation.spatial_map import SpatialMap
    from navigation.path_generator import PathGenerator
    from navigation.path_scorer import PathScorer
    from navigation.decision import DecisionMaker
    from navigation.spatial_entity import create_spatial_entities
    from navigation.critical_region import CriticalRegionExtractor
    from navigation.rho_fov import RhoFOVPredictor
    from navigation.camera_observation import CameraModel, CameraObservation, MultiCameraEntityState
    from navigation.camera_handover import PredictiveHandoverManager, HandoverMode, HandoverState
    from navigation.multi_camera_support import MultiCameraSupportGate
    from visualization.renderer import Renderer
    from system_logging.logger import SystemLogger
    from failure_analysis.detector import FailureDetector
    from evaluation.metrics import MetricsCollector

    logger.info("=" * 60)
    logger.info("ORCA EYE — Dual-Primary Camera Navigation Pipeline")
    logger.info("  CAM0 (Left Primary) : %s", source0)
    logger.info("  CAM1 (Right Primary): %s", source1)
    logger.info("  Center Overlap Zone : [-15.0 deg, +15.0 deg] at 0 deg")
    logger.info("RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE")
    logger.info("=" * 60)

    cam_cfg0 = dict(cfg.get("camera", {}))
    cam_cfg1 = dict(cfg.get("camera", {}))
    if loop:
        cam_cfg0["loop"] = True
        cam_cfg1["loop"] = True

    cam0 = CameraSource(source0, cam_cfg0)
    cam1 = CameraSource(source1, cam_cfg1)
    W0, H0 = cam0.get_resolution()
    W1, H1 = cam1.get_resolution()
    logger.info("CAM0 Resolution: %dx%d | CAM1 Resolution: %dx%d", W0, H0, W1, H1)

    # Dual Primary Camera Symmetric Geometry
    cam_model0 = CameraModel(camera_id="CAM0", yaw_deg=-17.5, half_hfov_rad=math.radians(32.5))
    cam_model1 = CameraModel(camera_id="CAM1", yaw_deg=17.5, half_hfov_rad=math.radians(32.5))

    camera_calib = CameraCalibration.from_fov(frame_width=W0, frame_height=H0, hfov_deg=65.0)
    geometry_3d_engine = Geometry3D(camera_calib)

    detector = YOLODetector(cfg.get("detection", {}))
    tracker0 = CentroidTracker(cfg.get("tracking", {}))
    tracker1 = CentroidTracker(cfg.get("tracking", {}))
    depth_estimator = DepthEstimator(cfg.get("depth", {}))
    fs_estimator = FreeSpaceEstimator(cfg.get("freespace", {}))
    spatial_map = SpatialMap(cfg.get("spatial_map", {}), frame_width=W0, frame_height=H0)
    path_generator = PathGenerator(cfg.get("path_generation", {}), spatial_map.rows, spatial_map.cols)
    scorer = PathScorer(cfg.get("scoring", {}))

    rho_cfg = cfg.get("stage_a", {}).get("rho_fov", {})
    rho_predictor = RhoFOVPredictor(
        sigma_br=float(rho_cfg.get("sigma_br", 0.60)),
        num_samples=int(rho_cfg.get("num_samples", 200)),
        horizon_s=float(rho_cfg.get("horizon_s", 2.0)),
        dt_sample_s=float(rho_cfg.get("dt_sample_s", 0.20)),
        use_soft_quality=bool(rho_cfg.get("use_soft_quality", True)),
    )

    ho_cfg = cfg.get("dual_camera", {}).get("handover", {})
    handover_mgr = PredictiveHandoverManager(
        primary_camera_id="CAM0",
        secondary_camera_id="CAM1",
        tau_release=float(ho_cfg.get("tau_release", 0.30)),
        tau_acquire=float(ho_cfg.get("tau_acquire", 0.25)),
        tau_assoc=float(ho_cfg.get("tau_assoc", 0.70)),
        min_dwell_time_s=float(ho_cfg.get("min_dwell_time_s", 0.50)),
        mode=HandoverMode.B2_PREDICTIVE,
    )

    safety_cfg = cfg.get("safety", {})
    multi_support_gate = MultiCameraSupportGate(
        tau_safe=float(safety_cfg.get("tau_safe", 0.35)),
        tau_cam=float(safety_cfg.get("tau_cam", 0.40)),
        primary_camera_id="CAM0",
    )
    critical_extractor = CriticalRegionExtractor()
    decision_maker = DecisionMaker(safety_cfg)
    renderer = Renderer(cfg, W0, H0)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys_logger = SystemLogger(cfg, session_id=session_id)
    fail_detect = FailureDetector(cfg)
    metrics = MetricsCollector(session_dir=sys_logger.get_session_dir()) if evaluate else None
    fps_stats = _FPSStats(window=30)

    entity_states: dict[int, MultiCameraEntityState] = {}
    next_global_id = 1
    cam0_track_map: dict[int, int] = {}  # track_id -> global_id
    cam1_track_map: dict[int, int] = {}  # track_id -> global_id

    frame_id = 0
    depth_skip = int(cfg.get("depth", {}).get("skip_frames", 4))
    _last_depth0 = None
    _last_depth1 = None
    _depth_frame_ctr = 0

    composite = None
    last_frame0 = None
    last_frame1 = None
    cam0_ended = False
    cam1_ended = False

    logger.info("Dual-Primary Pipeline running. Press 'q' to quit.")

    try:
        while True:
            t0_frame = time.perf_counter()
            ret0, frame0, _ = cam0.read()
            ret1, frame1, _ = cam1.read()

            if ret0 and frame0 is not None:
                last_frame0 = frame0
            else:
                cam0_ended = True
                frame0 = last_frame0

            if ret1 and frame1 is not None:
                last_frame1 = frame1
            else:
                cam1_ended = True
                frame1 = last_frame1

            # Play till both videos complete their full length
            if (cam0_ended and cam1_ended) or frame0 is None or frame1 is None:
                logger.info("Dual video stream reached end of playback for both cameras.")
                break

            frame_id += 1
            if max_frames and frame_id > max_frames:
                logger.info("Reached max frames (%d). Exiting dual loop.", max_frames)
                break

            now_ts = time.time()

            # YOLO inference on both cameras
            dr0 = detector.detect(frame0)
            dr1 = detector.detect(frame1)

            # Tracking on both cameras
            tr0 = tracker0.update(dr0.objects, frame_id=frame_id)
            tr1 = tracker1.update(dr1.objects, frame_id=frame_id)

            for obj in dr0.objects:
                for t in tr0:
                    if abs(t.center[0] - obj.center[0]) < 15 and abs(t.center[1] - obj.center[1]) < 15:
                        obj.track_id = t.track_id
                        break

            for obj in dr1.objects:
                for t in tr1:
                    if abs(t.center[0] - obj.center[0]) < 15 and abs(t.center[1] - obj.center[1]) < 15:
                        obj.track_id = t.track_id
                        break

            # Build CameraObservation lists
            obs0_list: list[CameraObservation] = []
            obs1_list: list[CameraObservation] = []

            hfov_tan = math.tan(cam_model0.half_hfov_rad)
            f_px0 = (W0 / 2.0) / hfov_tan

            for t in tr0:
                ux = (t.center[0] - W0 / 2.0) / f_px0
                b_loc = math.atan(ux)
                w_b = cam_model0.local_to_world_bearing(b_loc)
                tb = getattr(t, "bbox", getattr(t, "box", None))
                box_h = max(30, tb[3] - tb[1]) if (tb and len(tb) >= 4) else 60
                dist_m = max(0.8, min(6.0, 2.8 * (H0 / float(box_h))))
                rate_rad_s = (t.velocity[0] / f_px0) * 30.0  # approximate px/frame to rad/s

                pred0 = rho_predictor.predict_survival(
                    entity_id=t.track_id,
                    bearing_rad=b_loc,
                    bearing_rate=rate_rad_s,
                    camera_id="CAM0",
                    current_time=now_ts,
                )
                obs0 = CameraObservation(
                    camera_id="CAM0",
                    track_id=t.track_id,
                    timestamp=now_ts,
                    local_bearing_rad=b_loc,
                    local_bearing_deg=math.degrees(b_loc),
                    local_bearing_rate_rad_s=rate_rad_s,
                    world_bearing_rad=w_b,
                    world_bearing_deg=math.degrees(w_b),
                    distance_m=dist_m,
                    world_pos=(dist_m * math.sin(w_b), dist_m * math.cos(w_b)),
                    world_vel=(dist_m * rate_rad_s * math.cos(w_b), -dist_m * rate_rad_s * math.sin(w_b)),
                    rho_fov=pred0.rho_fov,
                    detection_confidence=0.90,
                    class_name=getattr(t, "class_name", "obstacle"),
                    in_fov=True,
                )
                obs0_list.append(obs0)

            W1 = frame1.shape[1]
            H1 = frame1.shape[0]
            f_px1 = (W1 / 2.0) / hfov_tan

            for t in tr1:
                ux = (t.center[0] - W1 / 2.0) / f_px1
                b_loc = math.atan(ux)
                w_b = cam_model1.local_to_world_bearing(b_loc)
                tb = getattr(t, "bbox", getattr(t, "box", None))
                box_h = max(30, tb[3] - tb[1]) if (tb and len(tb) >= 4) else 60
                dist_m = max(0.8, min(6.0, 2.8 * (H1 / float(box_h))))
                rate_rad_s = (t.velocity[0] / f_px1) * 30.0

                pred1 = rho_predictor.predict_survival(
                    entity_id=t.track_id + 500,
                    bearing_rad=b_loc,
                    bearing_rate=rate_rad_s,
                    camera_id="CAM1",
                    current_time=now_ts,
                )
                obs1 = CameraObservation(
                    camera_id="CAM1",
                    track_id=t.track_id,
                    timestamp=now_ts,
                    local_bearing_rad=b_loc,
                    local_bearing_deg=math.degrees(b_loc),
                    local_bearing_rate_rad_s=rate_rad_s,
                    world_bearing_rad=w_b,
                    world_bearing_deg=math.degrees(w_b),
                    distance_m=dist_m,
                    world_pos=(dist_m * math.sin(w_b), dist_m * math.cos(w_b)),
                    world_vel=(dist_m * rate_rad_s * math.cos(w_b), -dist_m * rate_rad_s * math.sin(w_b)),
                    rho_fov=pred1.rho_fov,
                    detection_confidence=0.88,
                    class_name=getattr(t, "class_name", "obstacle"),
                    in_fov=True,
                )
                obs1_list.append(obs1)

            # Cross-camera association in overlap zone
            matched = handover_mgr.associator.associate(obs0_list, obs1_list, now=now_ts)
            matched_pairs = {m[0].track_id: (m[1].track_id, m[2]) for m in matched}

            # Update global multi-camera entity mapping
            for ent in entity_states.values():
                ent.observations.clear()

            for o0 in obs0_list:
                t0_id = o0.track_id
                if t0_id not in cam0_track_map:
                    gid = next_global_id
                    next_global_id += 1
                    cam0_track_map[t0_id] = gid
                    entity_states[gid] = MultiCameraEntityState(global_entity_id=gid, class_name=o0.class_name)
                gid = cam0_track_map[t0_id]
                entity_states[gid].observations["CAM0"] = o0

                if t0_id in matched_pairs:
                    t1_id, conf = matched_pairs[t0_id]
                    cam1_track_map[t1_id] = gid

            for o1 in obs1_list:
                t1_id = o1.track_id
                if t1_id in cam1_track_map:
                    gid = cam1_track_map[t1_id]
                else:
                    gid = next_global_id
                    next_global_id += 1
                    cam1_track_map[t1_id] = gid
                    entity_states[gid] = MultiCameraEntityState(global_entity_id=gid, class_name=o1.class_name)
                entity_states[gid].observations["CAM1"] = o1

            # Update Handover state machine for each entity
            for gid, ent in entity_states.items():
                c_conf = 0.0
                obs0 = ent.observations.get("CAM0")
                if obs0 and obs0.track_id in matched_pairs:
                    c_conf = matched_pairs[obs0.track_id][1]
                handover_mgr.update_entity_handover(ent, association_confidence=c_conf, current_time=now_ts)

            # Depth & Freespace on both Primary Cameras (CAM0 Left, CAM1 Right)
            _depth_frame_ctr += 1
            if _depth_frame_ctr % depth_skip == 0 or _last_depth0 is None:
                dpr0 = depth_estimator.estimate(frame0)
                _last_depth0 = dpr0
            else:
                dpr0 = _last_depth0

            if _depth_frame_ctr % depth_skip == 0 or _last_depth1 is None:
                dpr1 = depth_estimator.estimate(frame1)
                _last_depth1 = dpr1
            else:
                dpr1 = _last_depth1

            fs0 = fs_estimator.estimate(frame0, dpr0, dr0.objects)
            fs1 = fs_estimator.estimate(frame1, dpr1, dr1.objects)

            # Unified spatial map updating from both primary cameras
            spatial_map.update_dual(
                freespace_result0=fs0,
                depth_result0=dpr0,
                tracks0=tr0,
                freespace_result1=fs1,
                depth_result1=dpr1,
                tracks1=tr1,
            )
            geom_3d = geometry_3d_engine.analyze(dpr0.depth_map, dr0.objects) if (dpr0 and dpr0.is_valid and dpr0.depth_map is not None) else None
            wall_proximity = spatial_map.evaluate_wall_proximity(geometry_3d_result=geom_3d)
            candidates = path_generator.generate(spatial_map, geometry_3d_result=geom_3d)
            candidates = scorer.score(candidates)

            # Unified Spatial Entities & Critical Regions across both Primary cameras
            se0 = create_spatial_entities(
                tracks=tr0,
                depth_map=dpr0.depth_normalized if dpr0 else None,
                depth_estimator=depth_estimator,
                frame_w=W0,
                frame_h=H0,
            )
            for e in se0:
                e.relative_bearing -= 17.5  # shift from CAM0 optical axis to wearer center

            se1 = create_spatial_entities(
                tracks=tr1,
                depth_map=dpr1.depth_normalized if dpr1 else None,
                depth_estimator=depth_estimator,
                frame_w=W1,
                frame_h=H1,
            )
            for e in se1:
                e.relative_bearing += 17.5  # shift from CAM1 optical axis to wearer center

            spatial_entities = se0 + se1
            critical_regions = critical_extractor.extract_critical_regions(
                candidates=candidates,
                spatial_entities=spatial_entities,
            )

            # Stage B Multi-Camera Corridor Support Gate
            corridor_support = multi_support_gate.evaluate_corridors(
                candidates=candidates,
                critical_regions=critical_regions,
                entities=entity_states,
                use_multi_cam=True,
            )
            legacy_support = {k: v.to_legacy_corridor_support_record() for k, v in corridor_support.items()}

            decision = decision_maker.decide(
                candidates=candidates,
                frame_id=frame_id,
                spatial_entities=spatial_entities,
                wall_proximity=wall_proximity,
                corridor_support=legacy_support,
                primary_camera_id="CAM0",
            )

            fps = fps_stats.tick()
            total_ms = (time.perf_counter() - t0_frame) * 1000.0

            # Render Merged Dual Dashboard
            composite = renderer.render_dual(
                frame0=frame0,
                frame1=frame1,
                detection_result0=dr0,
                tracks0=tr0,
                detection_result1=dr1,
                tracks1=tr1,
                entity_states=entity_states,
                associations=matched,
                candidates=candidates,
                decision=decision,
                corridor_support_records=corridor_support,
                fps=fps,
                frame_id=frame_id,
                latency_ms=total_ms,
                freespace_result0=fs0,
                freespace_result1=fs1,
                spatial_map=spatial_map,
                geometry_3d_result=geom_3d,
                wall_proximity=wall_proximity,
            )

            if save_snapshot and frame_id == 25:
                cv2.imwrite(save_snapshot, composite)
                logger.info("Saved snapshot to %s", save_snapshot)

            key = renderer.show(composite)
            if key == ord("q"):
                logger.info("User quit.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in dual pipeline.")
    finally:
        cam0.release()
        cam1.release()
        renderer.destroy()
        if save_snapshot and not Path(save_snapshot).exists() and composite is not None:
            cv2.imwrite(save_snapshot, composite)
            logger.info("Saved exit snapshot to %s", save_snapshot)
        logger.info("Dual pipeline complete. Processed %d frames.", frame_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ORCA EYE — Baseline Assistive Vision Navigation System\n"
                    "RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", default=None,
                   help="int webcam index or URL/path to single-camera video.")
    p.add_argument("--dual-sources", nargs=2, default=None, metavar=("CAM0", "CAM1"),
                   help="Paths to CAM0 and CAM1 video files for Stage B dual-camera pipeline.")
    p.add_argument("--dual", action="store_true",
                   help="Run Stage B dual-camera pipeline using config.yaml sources.")
    p.add_argument("--save-snapshot", default=None,
                   help="Save diagnostic frame snapshot to this path.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--evaluate", action="store_true",
                   help="Enable full metrics report.")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure per-stage GPU latency and exit.")
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximum number of frames to run before exiting.")
    p.add_argument("--loop", "-l", action="store_true",
                   help="Continuously loop video playback when playing video files.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_scenarios:
        from evaluation.scenarios import list_scenarios
        list_scenarios()
        return

    cfg = load_config(args.config)
    setup_gpu(cfg)

    # 1. Dual-camera mode
    if args.dual_sources or args.dual:
        if args.dual_sources:
            src0, src1 = args.dual_sources
        else:
            dual_cfg = cfg.get("dual_camera", {})
            src0 = dual_cfg.get("cam0", {}).get("source", r"D:\orca\videos\Dual\LEFT CAMERA.mp4")
            src1 = dual_cfg.get("cam1", {}).get("source", r"D:\orca\videos\Dual\RIGHT CAMERA.mp4")

        run_dual_pipeline(
            source0=src0,
            source1=src1,
            cfg=cfg,
            evaluate=args.evaluate,
            max_frames=args.max_frames,
            save_snapshot=args.save_snapshot,
            loop=args.loop,
        )
        return

    # 2. Single-camera mode
    default_test_video = Path(r"D:\orca\videos\WhatsApp Video 2026-09-16 at 7.37.26 PM.mp4")
    if args.source is not None:
        source = args.source
    elif default_test_video.exists():
        source = str(default_test_video)
        logger.info("Using default local test video: %s", source)
    else:
        source = cfg.get("camera", {}).get("source", 0)

    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    logger.info("Source: %s", source)

    if args.benchmark:
        run_benchmark(source, cfg)
    else:
        run_pipeline(source, cfg, evaluate=args.evaluate, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

