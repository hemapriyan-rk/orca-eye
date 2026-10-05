"""
ORCA EYE — Stage A: Computational Overhead & Runtime Profiler
============================================================
Task 11: Measures runtime overhead of Stage-A Navigation-Coupled Perceptual Support:
  1. End-to-end DecisionMaker latency: Baseline (B0) vs Stage-A (B1) (Mean, P95, FPS).
  2. Stage-A component breakdown:
     - CriticalRegionExtractor latency
     - RhoFOVPredictor latency across N in [50, 100, 200, 500]
     - PerceptualSupportGate latency
     - Total overhead Delta t (ms)
  3. Peak memory usage (CPU RSS and GPU VRAM if CUDA available).

Outputs:
  evaluation/results/runtime_results.csv
"""

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Set project root
sys.path.insert(0, os.path.abspath("."))

from navigation.critical_region import CriticalRegionExtractor
from navigation.camera_support import PerceptualSupportGate
from navigation.decision import DecisionMaker
from navigation.geometry import EntityKinematics
from evaluation.reproducibility import seed_everything
from evaluation.scenarios import ControlledScenarioType, create_synthetic_spatial_state
from navigation.path_generator import PathGenerator
from navigation.path_scorer import PathScorer
from navigation.rho_fov import RhoFOVPredictor


def generate_benchmark_data():
    """Creates realistic candidate corridors and spatial entities using existing pipeline classes."""
    smap, entities = create_synthetic_spatial_state(ControlledScenarioType.CROSSING_PEDESTRIAN)
    pgen = PathGenerator({"use_astar": False}, smap.rows, smap.cols)
    scorer = PathScorer({})
    candidates = scorer.score(pgen.generate(smap))
    return candidates, entities


def profile_runtime(
    n_iterations: int = 500,
    output_csv: str = "evaluation/results/runtime_results.csv",
) -> Dict:
    print("\n" + "=" * 60)
    print("Running Task 11: Stage-A Computational Overhead Profiler")
    print("=" * 60)
    seed_everything(42)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    candidates, entities = generate_benchmark_data()

    # Profiling modules
    crit_ext = CriticalRegionExtractor()
    gate = PerceptualSupportGate(tau_safe=0.35, tau_cam=0.40)
    dec_b0 = DecisionMaker({"enable_support_gate": False})
    dec_b1 = DecisionMaker({"enable_support_gate": True})

    # 1. Profile RhoFOVPredictor across N in [50, 100, 200, 500]
    n_values = [50, 100, 200, 500]
    rho_times_by_n = {}

    for n in n_values:
        pred_n = RhoFOVPredictor(num_samples=n, seed=42)
        # Warm-up
        for _ in range(50):
            pred_n.predict_survival(1, bearing_rad=0.2, bearing_rate=0.1)

        t0 = time.perf_counter()
        for _ in range(n_iterations):
            for e in entities:
                b_val = getattr(e, 'relative_bearing', getattr(e, 'bearing_deg', 0.0))
                b_rad = math.radians(b_val)
                pred_n.predict_survival(e.track_id, bearing_rad=b_rad, bearing_rate=0.15)
        total_time = (time.perf_counter() - t0) * 1000.0  # ms
        avg_time_per_entity = total_time / (n_iterations * len(entities))
        avg_time_per_call = total_time / n_iterations  # for all 3 entities
        rho_times_by_n[n] = (avg_time_per_entity, avg_time_per_call)
        print(f"  RhoFOV (N={n:3d}): {avg_time_per_entity:.4f} ms/entity | {avg_time_per_call:.4f} ms total ({len(entities)} entities)")

    # 2. Profile CriticalRegionExtractor
    for _ in range(50):
        crit_ext.extract_critical_regions(candidates, entities)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        crit_regions = crit_ext.extract_critical_regions(candidates, entities)
    crit_ext_time_ms = ((time.perf_counter() - t0) * 1000.0) / n_iterations
    print(f"  CriticalRegionExtractor: {crit_ext_time_ms:.4f} ms/frame")

    # 3. Profile PerceptualSupportGate
    rho_pred_200 = RhoFOVPredictor(num_samples=200, seed=42)
    rho_preds = {
        e.track_id: rho_pred_200.predict_survival(
            e.track_id,
            math.radians(getattr(e, 'relative_bearing', getattr(e, 'bearing_deg', 0.0))),
            0.15,
        )
        for e in entities
    }
    crit_regions = crit_ext.extract_critical_regions(candidates, entities)

    for _ in range(50):
        gate.evaluate_corridors(candidates, crit_regions, rho_preds)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        corridor_support = gate.evaluate_corridors(candidates, crit_regions, rho_preds)
    gate_time_ms = ((time.perf_counter() - t0) * 1000.0) / n_iterations
    print(f"  PerceptualSupportGate  : {gate_time_ms:.4f} ms/frame")

    # 4. Total Stage-A Overhead Delta t (for N=200)
    delta_t_ms = crit_ext_time_ms + rho_times_by_n[200][1] + gate_time_ms
    print(f"  Total Stage-A Overhead (N=200): {delta_t_ms:.4f} ms/frame")

    # 5. Full DecisionMaker End-to-End Latency: B0 vs B1
    b0_latencies = []
    b1_latencies = []

    # Warm-up
    for _ in range(50):
        dec_b0.decide(candidates, frame_id=1, spatial_entities=entities)
        dec_b1.decide(candidates, frame_id=1, spatial_entities=entities, corridor_support=corridor_support)

    for i in range(n_iterations):
        # B0
        t0 = time.perf_counter()
        dec_b0.decide(candidates, frame_id=i, spatial_entities=entities)
        b0_latencies.append((time.perf_counter() - t0) * 1000.0)

        # B1 (including Stage-A support evaluation)
        t0 = time.perf_counter()
        cr = crit_ext.extract_critical_regions(candidates, entities)
        rp = {
            e.track_id: rho_pred_200.predict_survival(
                e.track_id,
                math.radians(getattr(e, 'relative_bearing', getattr(e, 'bearing_deg', 0.0))),
                0.15,
            )
            for e in entities
        }
        cs = gate.evaluate_corridors(candidates, cr, rp)
        dec_b1.decide(candidates, frame_id=i, spatial_entities=entities, corridor_support=cs)
        b1_latencies.append((time.perf_counter() - t0) * 1000.0)

    b0_mean = float(np.mean(b0_latencies))
    b0_p95 = float(np.percentile(b0_latencies, 95))
    b0_fps = 1000.0 / b0_mean if b0_mean > 0 else 9999.0

    b1_mean = float(np.mean(b1_latencies))
    b1_p95 = float(np.percentile(b1_latencies, 95))
    b1_fps = 1000.0 / b1_mean if b1_mean > 0 else 9999.0

    print(f"\n  Decision + Planning Latency:")
    print(f"    Baseline B0 : Mean = {b0_mean:.4f} ms | P95 = {b0_p95:.4f} ms | Nav Module FPS = {b0_fps:.1f}")
    print(f"    Stage-A  B1 : Mean = {b1_mean:.4f} ms | P95 = {b1_p95:.4f} ms | Nav Module FPS = {b1_fps:.1f}")
    print(f"    Delta Latency: {b1_mean - b0_mean:.4f} ms")

    # 6. Check Memory
    cpu_rss_mb = 0.0
    gpu_vram_mb = 0.0
    try:
        import psutil
        process = psutil.Process()
        cpu_rss_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            gpu_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    # Save to CSV
    rows = [
        {
            "metric": "b0_latency_mean_ms",
            "value": round(b0_mean, 4),
            "description": "Baseline B0 decision latency mean (ms)",
        },
        {
            "metric": "b0_latency_p95_ms",
            "value": round(b0_p95, 4),
            "description": "Baseline B0 decision latency 95th percentile (ms)",
        },
        {
            "metric": "b0_fps",
            "value": round(b0_fps, 2),
            "description": "Baseline B0 navigation loop max throughput (FPS)",
        },
        {
            "metric": "b1_latency_mean_ms",
            "value": round(b1_mean, 4),
            "description": "Stage-A B1 decision latency mean (ms)",
        },
        {
            "metric": "b1_latency_p95_ms",
            "value": round(b1_p95, 4),
            "description": "Stage-A B1 decision latency 95th percentile (ms)",
        },
        {
            "metric": "b1_fps",
            "value": round(b1_fps, 2),
            "description": "Stage-A B1 navigation loop max throughput (FPS)",
        },
        {
            "metric": "delta_latency_ms",
            "value": round(b1_mean - b0_mean, 4),
            "description": "Incremental overhead of Stage-A (ms)",
        },
        {
            "metric": "crit_ext_latency_ms",
            "value": round(crit_ext_time_ms, 4),
            "description": "CriticalRegionExtractor latency (ms)",
        },
        {
            "metric": "gate_latency_ms",
            "value": round(gate_time_ms, 4),
            "description": "PerceptualSupportGate latency (ms)",
        },
        {
            "metric": "rho_fov_n50_ms",
            "value": round(rho_times_by_n[50][0], 4),
            "description": "RhoFOV latency per entity at N=50 (ms)",
        },
        {
            "metric": "rho_fov_n100_ms",
            "value": round(rho_times_by_n[100][0], 4),
            "description": "RhoFOV latency per entity at N=100 (ms)",
        },
        {
            "metric": "rho_fov_n200_ms",
            "value": round(rho_times_by_n[200][0], 4),
            "description": "RhoFOV latency per entity at N=200 (ms)",
        },
        {
            "metric": "rho_fov_n500_ms",
            "value": round(rho_times_by_n[500][0], 4),
            "description": "RhoFOV latency per entity at N=500 (ms)",
        },
        {
            "metric": "cpu_rss_mb",
            "value": round(cpu_rss_mb, 2),
            "description": "Current process CPU RSS memory (MB)",
        },
        {
            "metric": "gpu_vram_mb",
            "value": round(gpu_vram_mb, 2),
            "description": "Peak GPU VRAM allocated (MB)",
        },
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "description"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Runtime profiling complete. Results saved to: {output_csv}")
    return {r["metric"]: r["value"] for r in rows}


if __name__ == "__main__":
    profile_runtime()
