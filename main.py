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
    independent CUDA streams.
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

        fut_det = executor.submit(_run_det)
        fut_depth = executor.submit(_run_depth)

        det_result = fut_det.result()
        depth_result = fut_depth.result()

        # Wait for both streams to complete execution
        torch.cuda.synchronize()
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
    from navigation.spatial_map    import SpatialMap
    from navigation.path_generator import PathGenerator
    from navigation.path_scorer    import PathScorer
    from navigation.decision       import DecisionMaker
    from navigation.spatial_entity import create_spatial_entities
    from navigation.dynamic_conflict import DynamicConflictEngine
    from navigation.audio_guidance import InstructionGenerator
    from visualization.renderer    import Renderer
    from system_logging.logger     import SystemLogger
    from failure_analysis.detector import FailureDetector
    from evaluation.metrics        import MetricsCollector

    logger.info("=" * 60)
    logger.info("ORCA EYE — Baseline Assistive Vision Navigation System")
    logger.info("RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE")
    logger.info("=" * 60)

    # ── Init all modules ─────────────────────────────────────────────
    camera  = CameraSource(source, cfg.get("camera", {}))
    W, H    = camera.get_resolution()
    logger.info("Camera: %dx%d", W, H)

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

    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys_logger  = SystemLogger(cfg, session_id=session_id)
    fail_detect = FailureDetector(cfg)

    metrics = MetricsCollector(session_dir=sys_logger.get_session_dir()) \
              if evaluate else None

    fps_stats  = _FPSStats(window=30)
    frame_id   = 0

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

            # ── Stages 2+4a: YOLO + MiDaS (parallel CUDA streams) ───
            t_gpu = time.perf_counter()
            detection_result, depth_result = _run_parallel_gpu_inference(
                detector, depth_estimator, frame,
            )
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

            # ── Stage 4b: Free-space (CUDA tensors) ──────────────────
            freespace_result = fs_estimator.estimate(
                frame, depth_result, detection_result.objects
            )

            # ── Stage 5: Spatial map ──────────────────────────────────
            spatial_map.update(freespace_result, depth_result, tracks)
            wall_proximity = spatial_map.evaluate_wall_proximity()

            # ── Stage 6: Path generation ──────────────────────────────
            candidates = path_generator.generate(spatial_map)

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

            # ── Stage 8: Decision (NavigationState) ───────────────────
            decision   = decision_maker.decide(
                candidates,
                frame_id=frame_id,
                spatial_entities=spatial_entities,
                dynamic_conflicts=active_conflicts,
                wall_proximity=wall_proximity,
            )
            stability  = decision_maker.get_stability_metrics()

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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ORCA EYE — Baseline Assistive Vision Navigation System\n"
                    "RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source",     default=None,
                   help="int webcam index or URL/path to video.")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--evaluate",   action="store_true",
                   help="Enable full metrics report.")
    p.add_argument("--benchmark",  action="store_true",
                   help="Measure per-stage GPU latency and exit.")
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximum number of frames to run before exiting.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_scenarios:
        from evaluation.scenarios import list_scenarios
        list_scenarios()
        return

    cfg = load_config(args.config)
    setup_gpu(cfg)

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
