"""
ORCA EYE — Stage 12a: Metrics Collector
=========================================
Collects, aggregates, and reports perception/navigation/stability metrics
across a session. Outputs a Markdown summary report + CSV.

Metrics tracked:
  Perception    : FPS, latency, detection count stats
  Free-space    : uncertainty ratio over time
  Navigation    : command distribution, N_switch, T_stable, clearance
  Stability     : oscillation events, mean/min clearance
"""

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics across frames and produces session reports.

    Call update() each frame. Call report() at session end.
    """

    def __init__(self, session_dir: Optional[Path] = None) -> None:
        self.session_dir = session_dir or Path("logs/default")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Perception
        self.fps_history: List[float] = []
        self.latency_history: List[float] = []
        self.detection_counts: List[int] = []
        self.confidence_values: List[float] = []

        # Free-space
        self.freespace_uncertainty: List[float] = []
        self.free_fractions: List[float] = []
        self.obstacle_fractions: List[float] = []

        # Navigation
        self.commands: List[str] = []
        self.scores: List[float] = []
        self.clearances: List[float] = []

        # Stability
        self.n_switch_events: List[int] = []  # n_switch_window per frame
        self.t_stable_values: List[float] = []
        self.oscillation_events: int = 0

        self._frame_count: int = 0
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        fps: float,
        latency_ms: float,
        detection_result,
        freespace_result,
        decision,
        stability_metrics: Dict[str, Any],
    ) -> None:
        """Update metrics from current frame outputs."""
        self._frame_count += 1

        # Perception
        self.fps_history.append(fps)
        self.latency_history.append(latency_ms)
        self.detection_counts.append(detection_result.num_detections if detection_result else 0)
        if detection_result and detection_result.num_detections > 0:
            self.confidence_values.append(detection_result.mean_confidence)

        # Free-space
        if freespace_result:
            stats = freespace_result.statistics
            self.freespace_uncertainty.append(stats.get("uncertainty", 1.0))
            self.free_fractions.append(stats.get("free_fraction", 0.0))
            self.obstacle_fractions.append(stats.get("obstacle_fraction", 0.0))

        # Navigation
        self.commands.append(decision.command)
        self.scores.append(decision.score)
        self.clearances.append(decision.clearance)

        # Stability
        self.n_switch_events.append(stability_metrics.get("n_switch_window", 0))
        if stability_metrics.get("is_oscillating", False):
            self.oscillation_events += 1

    def report(self) -> Dict[str, Any]:
        """
        Generate and return a summary metrics dict.
        Also writes a Markdown report file.
        """
        session_duration = time.time() - self._start_time
        n = self._frame_count

        # --- Perception
        fps_arr = np.array(self.fps_history) if self.fps_history else np.array([0.0])
        lat_arr = np.array(self.latency_history) if self.latency_history else np.array([0.0])
        det_arr = np.array(self.detection_counts) if self.detection_counts else np.array([0])
        conf_arr = np.array(self.confidence_values) if self.confidence_values else np.array([0.0])

        # --- Free-space
        unc_arr = np.array(self.freespace_uncertainty) if self.freespace_uncertainty else np.array([1.0])
        free_arr = np.array(self.free_fractions) if self.free_fractions else np.array([0.0])

        # --- Navigation
        cmd_counter = Counter(self.commands)
        score_arr = np.array(self.scores) if self.scores else np.array([0.0])
        clear_arr = np.array(self.clearances) if self.clearances else np.array([0.0])

        stop_rate = cmd_counter.get("STOP", 0) / max(n, 1)
        caution_rate = cmd_counter.get("CAUTION", 0) / max(n, 1)

        metrics = {
            "session": {
                "frame_count": n,
                "duration_s": round(session_duration, 2),
            },
            "perception": {
                "fps_mean": round(float(fps_arr.mean()), 2),
                "fps_min": round(float(fps_arr.min()), 2),
                "fps_max": round(float(fps_arr.max()), 2),
                "latency_mean_ms": round(float(lat_arr.mean()), 2),
                "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
                "detection_mean": round(float(det_arr.mean()), 2),
                "detection_max": int(det_arr.max()),
                "mean_confidence": round(float(conf_arr.mean()), 4),
            },
            "freespace": {
                "uncertainty_mean": round(float(unc_arr.mean()), 4),
                "uncertainty_p95": round(float(np.percentile(unc_arr, 95)), 4),
                "free_fraction_mean": round(float(free_arr.mean()), 4),
            },
            "navigation": {
                "command_distribution": dict(cmd_counter),
                "stop_rate": round(stop_rate, 4),
                "caution_rate": round(caution_rate, 4),
                "score_mean": round(float(score_arr.mean()), 4),
                "score_min": round(float(score_arr.min()), 4),
                "clearance_mean": round(float(clear_arr.mean()), 4),
                "clearance_min": round(float(clear_arr.min()), 4),
            },
            "stability": {
                "total_oscillation_events": self.oscillation_events,
                "oscillation_rate": round(self.oscillation_events / max(n, 1), 4),
                "n_switch_window_mean": round(float(np.mean(self.n_switch_events)), 2)
                if self.n_switch_events else 0.0,
            },
        }

        self._write_markdown_report(metrics)
        self._write_json_report(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_markdown_report(self, metrics: Dict) -> None:
        report_path = self.session_dir / "performance_report.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# ORCA EYE — Performance Report\n\n")
                f.write("> RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE\n\n")

                f.write("## Session\n")
                f.write(f"- Frames: {metrics['session']['frame_count']}\n")
                f.write(f"- Duration: {metrics['session']['duration_s']:.1f}s\n\n")

                f.write("## Perception\n")
                p = metrics["perception"]
                f.write(f"| Metric | Value |\n|---|---|\n")
                f.write(f"| Mean FPS | {p['fps_mean']} |\n")
                f.write(f"| Min FPS | {p['fps_min']} |\n")
                f.write(f"| Mean Latency | {p['latency_mean_ms']} ms |\n")
                f.write(f"| P95 Latency | {p['latency_p95_ms']} ms |\n")
                f.write(f"| Mean Detections | {p['detection_mean']} |\n")
                f.write(f"| Mean Confidence | {p['mean_confidence']} |\n\n")

                f.write("## Free-Space\n")
                fs = metrics["freespace"]
                f.write(f"| Metric | Value |\n|---|---|\n")
                f.write(f"| Uncertainty Mean | {fs['uncertainty_mean']} |\n")
                f.write(f"| Uncertainty P95 | {fs['uncertainty_p95']} |\n")
                f.write(f"| Free Fraction Mean | {fs['free_fraction_mean']} |\n\n")

                f.write("## Navigation\n")
                nav = metrics["navigation"]
                f.write(f"| Metric | Value |\n|---|---|\n")
                f.write(f"| STOP Rate | {nav['stop_rate']:.1%} |\n")
                f.write(f"| CAUTION Rate | {nav['caution_rate']:.1%} |\n")
                f.write(f"| Mean Score | {nav['score_mean']} |\n")
                f.write(f"| Min Clearance | {nav['clearance_min']} |\n\n")

                f.write("### Command Distribution\n")
                for cmd, count in sorted(nav['command_distribution'].items()):
                    f.write(f"- {cmd}: {count} ({count/max(metrics['session']['frame_count'],1):.1%})\n")
                f.write("\n")

                f.write("## Stability\n")
                s = metrics["stability"]
                f.write(f"| Metric | Value |\n|---|---|\n")
                f.write(f"| Oscillation Events | {s['total_oscillation_events']} |\n")
                f.write(f"| Oscillation Rate | {s['oscillation_rate']:.1%} |\n")
                f.write(f"| Mean N_switch/window | {s['n_switch_window_mean']} |\n\n")

        except OSError as exc:
            logger.error("Failed to write performance report: %s", exc)

    def _write_json_report(self, metrics: Dict) -> None:
        json_path = self.session_dir / "performance_report.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except OSError as exc:
            logger.error("Failed to write JSON report: %s", exc)
