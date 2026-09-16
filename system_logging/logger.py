"""
ORCA EYE — Stage 10: System Logger
=====================================
Writes per-frame JSONL logs and time-series CSV metrics.

Per-frame JSONL record:
  timestamp, frame_id, fps, detections, tracks, depth_stats,
  freespace_stats, candidate_paths, path_scores, selected_path,
  min_clearance, uncertainty, navigation_command, stability_metrics

The log must allow later analysis of:
  "Why did ORCA EYE choose this path at this frame?"

This module is the research audit trail. Do NOT suppress entries.

Inputs  : all pipeline state dicts (already serialized by each module)
Outputs : JSONL file + CSV file in logs/
"""

import csv
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemLogger:
    """
    Thread-safe structured logger for ORCA EYE pipeline state.

    Configuration keys (from cfg['logging']):
      log_dir           : directory for output files
      per_frame_jsonl   : bool — write JSONL log
      metrics_csv       : bool — write CSV metrics
      save_frames       : bool — save JPEG frame per log entry (disk-heavy)
      flush_interval    : flush every N frames
      max_log_size_mb   : rotate log above this size
    """

    CSV_FIELDS = [
        "frame_id", "timestamp", "fps", "latency_ms",
        "num_detections", "num_tracks",
        "depth_mean", "depth_uncertainty",
        "freespace_free_frac", "freespace_obstacle_frac", "freespace_unknown_frac",
        "command", "score", "clearance", "uncertainty",
        "n_switch_total", "n_switch_window", "t_stable_mean_s",
        "is_oscillating",
    ]

    def __init__(self, cfg: dict, session_id: Optional[str] = None) -> None:
        log_cfg = cfg.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "logs"))
        self.per_frame_jsonl: bool = log_cfg.get("per_frame_jsonl", True)
        self.metrics_csv: bool = log_cfg.get("metrics_csv", True)
        self.flush_interval: int = log_cfg.get("flush_interval", 50)
        self.max_log_size_mb: float = log_cfg.get("max_log_size_mb", 500)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.session_dir / "frames.jsonl"
        self._csv_path = self.session_dir / "metrics.csv"
        self._frame_count: int = 0

        # Thread-safe write queue
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="orca_logger"
        )
        self._writer_thread.start()

        # CSV writer setup
        self._csv_file = None
        self._csv_writer = None
        if self.metrics_csv:
            self._open_csv()

        logger.info(
            "SystemLogger initialized. Session: %s, dir: %s",
            self.session_id, self.session_dir
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_frame(
        self,
        frame_id: int,
        fps: float,
        latency_ms: float,
        detection_result,
        tracks: List,
        depth_result,
        freespace_result,
        candidates: List,
        decision,
        stability_metrics: Dict[str, Any],
        detector_serializer,
        tracker_serializer,
        scorer_serializer,
        decision_serializer,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Enqueue a complete per-frame log record.

        All module serializers are passed in to keep this module
        decoupled from each module's internal representation.
        """
        record = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 2),
            "detections": detector_serializer(detection_result),
            "tracks": tracker_serializer(tracks),
            "depth_stats": depth_result.statistics if depth_result else {},
            "freespace_stats": freespace_result.statistics if freespace_result else {},
            "candidate_paths": scorer_serializer(candidates),
            "selected_path": {
                "direction": decision.selected_path_direction,
                "score": round(decision.score, 4),
            },
            "min_clearance": round(decision.clearance, 4),
            "uncertainty": round(decision.uncertainty, 4),
            "navigation_command": decision.command,
            "reason": decision.reason,
            "stability": stability_metrics,
        }
        if extra:
            record["extra"] = extra

        self._queue.put(("frame", record))
        self._frame_count += 1

        # CSV row (lightweight, synchronous — small enough)
        if self.metrics_csv and self._csv_writer is not None:
            self._write_csv_row(
                frame_id=frame_id,
                fps=fps,
                latency_ms=latency_ms,
                detection_result=detection_result,
                tracks=tracks,
                depth_result=depth_result,
                freespace_result=freespace_result,
                decision=decision,
                stability=stability_metrics,
            )

    def log_event(self, event_type: str, data: Dict) -> None:
        """Log a named event (non-frame, e.g. startup, shutdown, failure)."""
        record = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        self._queue.put(("event", record))

    def close(self) -> None:
        """Flush remaining records and close files."""
        self._stop_event.set()
        self._writer_thread.join(timeout=5.0)
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        logger.info(
            "SystemLogger closed. %d frames logged to %s",
            self._frame_count, self.session_dir
        )

    def get_session_dir(self) -> Path:
        return self.session_dir

    # ------------------------------------------------------------------
    # Internal — background writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Background thread that drains the queue to disk."""
        buffer: List[str] = []
        flush_count = 0

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                _, record = item
                buffer.append(json.dumps(record, default=str))
                flush_count += 1

                if flush_count >= self.flush_interval:
                    self._flush_jsonl(buffer)
                    buffer.clear()
                    flush_count = 0

            except queue.Empty:
                if buffer:
                    self._flush_jsonl(buffer)
                    buffer.clear()
                    flush_count = 0

        # Final flush
        if buffer:
            self._flush_jsonl(buffer)

    def _flush_jsonl(self, lines: List[str]) -> None:
        if not self.per_frame_jsonl or not lines:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as exc:
            logger.error("JSONL write error: %s", exc)

    # ------------------------------------------------------------------
    # Internal — CSV
    # ------------------------------------------------------------------

    def _open_csv(self) -> None:
        try:
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self.CSV_FIELDS, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()
        except OSError as exc:
            logger.error("CSV open error: %s", exc)
            self._csv_writer = None

    def _write_csv_row(
        self,
        frame_id, fps, latency_ms,
        detection_result, tracks, depth_result,
        freespace_result, decision, stability,
    ) -> None:
        depth_stats = depth_result.statistics if depth_result else {}
        fs_stats = freespace_result.statistics if freespace_result else {}

        row = {
            "frame_id": frame_id,
            "timestamp": round(time.time(), 4),
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 2),
            "num_detections": detection_result.num_detections if detection_result else 0,
            "num_tracks": len(tracks),
            "depth_mean": round(depth_stats.get("mean", 0.0), 4),
            "depth_uncertainty": round(depth_stats.get("uncertainty", 1.0), 4),
            "freespace_free_frac": round(fs_stats.get("free_fraction", 0.0), 4),
            "freespace_obstacle_frac": round(fs_stats.get("obstacle_fraction", 0.0), 4),
            "freespace_unknown_frac": round(fs_stats.get("unknown_fraction", 1.0), 4),
            "command": decision.command,
            "score": round(decision.score, 4),
            "clearance": round(decision.clearance, 4),
            "uncertainty": round(decision.uncertainty, 4),
            "n_switch_total": stability.get("n_switch_total", 0),
            "n_switch_window": stability.get("n_switch_window", 0),
            "t_stable_mean_s": round(stability.get("t_stable_mean_s", 0.0), 3),
            "is_oscillating": int(stability.get("is_oscillating", False)),
        }
        try:
            self._csv_writer.writerow(row)
            if frame_id % self.flush_interval == 0:
                self._csv_file.flush()
        except Exception as exc:
            logger.warning("CSV write error: %s", exc)
