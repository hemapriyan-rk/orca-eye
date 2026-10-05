"""
ORCA EYE — Stage A: Sensitivity Analysis (N & tau_cam)
======================================================
Tasks 6 & 7:
  - Task 6: N Sensitivity (N in {50, 100, 200, 500, 1000}) vs N=1000 reference.
  - Task 7: tau_cam Sensitivity (tau_cam in {0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80}).

Generates:
  evaluation/results/n_sensitivity.csv
  evaluation/results/n_sensitivity.png
  evaluation/results/threshold_sensitivity.csv
  evaluation/results/threshold_sensitivity.png
"""

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from evaluation.ground_truth import evaluate_ground_truth_survival
from evaluation.reproducibility import seed_everything
from evaluation.scenarios import ControlledScenarioType, create_synthetic_spatial_state
from navigation.camera_support import PerceptualSupportGate
from navigation.critical_region import CriticalRegionExtractor
from navigation.decision import DecisionMaker
from navigation.path_generator import PathGenerator
from navigation.path_scorer import PathScorer
from navigation.rho_fov import RhoFOVPredictor
from navigation.spatial_entity import SpatialEntity
from navigation.spatial_map import SpatialMap


# ---------------------------------------------------------------------------
# Task 6: N Sensitivity
# ---------------------------------------------------------------------------

def run_n_sensitivity_analysis(
    n_values: List[int] = [50, 100, 200, 500, 1000],
    num_test_states: int = 100,
    base_seed: int = 42,
    tau_cam: float = 0.40,
    output_csv: str = "evaluation/results/n_sensitivity.csv",
    output_png: str = "evaluation/results/n_sensitivity.png",
) -> List[Dict]:
    print("\n" + "=" * 60)
    print("Running Task 6: N Sensitivity Analysis (N in [50, 100, 200, 500, 1000])")
    print("=" * 60)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(base_seed)

    # Generate diverse test states across bearing [-30 deg, +30 deg] and bearing rate [0, 30 deg/s]
    test_states: List[Tuple[float, float]] = []
    for _ in range(num_test_states):
        b_deg = float(rng.uniform(-30.0, 30.0))
        rate_deg_s = float(rng.uniform(0.0, 30.0))
        test_states.append((math.radians(b_deg), math.radians(rate_deg_s)))

    # Compute reference values at N=1000
    ref_n = 1000
    ref_preds: List[float] = []
    ref_admiss: List[bool] = []
    ref_predictor = RhoFOVPredictor(num_samples=ref_n, seed=base_seed)

    for b_rad, rate_rad in test_states:
        pred = ref_predictor.predict_survival(1, b_rad, rate_rad)
        ref_preds.append(pred.rho_fov)
        ref_admiss.append(pred.rho_fov >= tau_cam)

    records: List[Dict] = []

    for N in n_values:
        predictor = RhoFOVPredictor(num_samples=N, seed=base_seed)
        t0 = time.perf_counter()

        n_preds: List[float] = []
        n_admiss: List[bool] = []
        for b_rad, rate_rad in test_states:
            pred = predictor.predict_survival(1, b_rad, rate_rad)
            n_preds.append(pred.rho_fov)
            n_admiss.append(pred.rho_fov >= tau_cam)

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        avg_runtime_ms = total_time_ms / float(num_test_states)

        abs_diffs = np.abs(np.array(n_preds) - np.array(ref_preds))
        mae = float(np.mean(abs_diffs))
        max_err = float(np.max(abs_diffs))
        flips = sum(1 for a, b in zip(n_admiss, ref_admiss) if a != b)
        flip_rate_pct = (flips / float(num_test_states)) * 100.0

        rec = {
            "N": N,
            "mean_mae_vs_1000": round(mae, 4),
            "max_error_vs_1000": round(max_err, 4),
            "decision_flips": flips,
            "flip_rate_pct": round(flip_rate_pct, 2),
            "runtime_ms_per_prediction": round(avg_runtime_ms, 4),
            "mean_rho": round(float(np.mean(n_preds)), 4),
        }
        records.append(rec)
        print(f"  N={N:4d} | MAE={mae:.4f} | MaxErr={max_err:.4f} | Flips={flips:2d} ({flip_rate_pct:4.1f}%) | Runtime={avg_runtime_ms:.4f} ms")

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ns = [r["N"] for r in records]
    maes = [r["mean_mae_vs_1000"] for r in records]
    flips = [r["flip_rate_pct"] for r in records]
    times = [r["runtime_ms_per_prediction"] for r in records]

    # Plot 1: Error and decision flips vs N
    color = "tab:blue"
    ax1.set_xlabel("Monte Carlo Sample Size (N)")
    ax1.set_ylabel("Mean Absolute Error vs N=1000", color=color)
    line1 = ax1.plot(ns, maes, marker="o", color=color, linewidth=2, label="MAE vs N=1000")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color = "tab:red"
    ax1_twin.set_ylabel("Decision Flip Rate (%)", color=color)
    line2 = ax1_twin.plot(ns, flips, marker="s", linestyle="--", color=color, linewidth=2, label="Flip Rate (%)")
    ax1_twin.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Prediction Error & Gate Decision Stability vs N")

    # Plot 2: Runtime scaling vs N
    ax2.plot(ns, times, marker="^", color="tab:green", linewidth=2)
    ax2.set_xlabel("Monte Carlo Sample Size (N)")
    ax2.set_ylabel("Computation Time (ms per entity)")
    ax2.set_title("Runtime Scaling vs Sample Size N")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Task 6 Output: CSV -> {output_csv}, Plot -> {output_png}")
    return records


# ---------------------------------------------------------------------------
# Task 7: tau_cam Sensitivity
# ---------------------------------------------------------------------------

def run_tau_cam_sensitivity_analysis(
    tau_cam_values: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    trials: int = 100,
    base_seed: int = 42,
    tau_safe: float = 0.35,
    output_csv: str = "evaluation/results/threshold_sensitivity.csv",
    output_png: str = "evaluation/results/threshold_sensitivity.png",
) -> List[Dict]:
    print("\n" + "=" * 60)
    print("Running Task 7: tau_cam Threshold Sensitivity (0.20 to 0.80)")
    print("=" * 60)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    for tau_cam in tau_cam_values:
        false_safes = 0
        false_rejections = 0
        stops = 0
        rejected_corridors_count = 0
        total_directional_corridors = 0
        hazard_selections = 0

        for trial_id in range(trials):
            trial_seed = base_seed + trial_id * 13
            rng = seed_everything(trial_seed)

            # Construct boundary scenario: right flank open, approaching entity near FOV edge
            smap = SpatialMap({"grid_rows": 12, "grid_cols": 20}, 640, 480)
            for r in range(8, 12):
                for c in range(9, 13):
                    smap.grid[r][c].occupancy = 0.01
                    smap.grid[r][c].free_prob = 0.99
            for r in range(8):
                for c in range(12):
                    smap.grid[r][c].occupancy = 0.95
                    smap.grid[r][c].free_prob = 0.05
            for r in range(12):
                for c in range(12, 20):
                    smap.grid[r][c].occupancy = 0.02
                    smap.grid[r][c].free_prob = 0.98

            # Stochastically vary bearing and rate around FOV boundary
            b_deg = float(rng.uniform(25.0, 32.0))
            rate_deg_s = float(rng.uniform(10.0, 20.0))
            b_rad = math.radians(b_deg)
            rate_rad = math.radians(rate_deg_s)

            entity = SpatialEntity(
                track_id=1, class_id=0, class_name="person", confidence=0.95,
                bbox=[150, 480, 250, 600], image_position=[540.0, 200.0], ground_position=[540.0, 400.0],
                relative_bearing=b_deg, relative_distance=0.35, velocity=[15.0, 0.0],
                tracking_confidence=0.95, depth_confidence=0.90, depth_valid=True, navigation_relevance=0.95,
            )
            entities = [entity]

            # Ground-truth survival
            gt = evaluate_ground_truth_survival(b_rad, rate_rad, horizon_s=2.0)
            actual_survived = gt.stayed_continuous_valid

            # Pipeline execution
            pgen = PathGenerator({"use_astar": False}, 12, 20)
            scorer = PathScorer({})
            candidates = scorer.score(pgen.generate(smap))

            crit_ext = CriticalRegionExtractor()
            crit_regs = crit_ext.extract_critical_regions(candidates=candidates, spatial_entities=entities)

            rho_pred = RhoFOVPredictor(num_samples=200, seed=trial_seed)
            rho_dict = {1: rho_pred.predict_survival(1, b_rad, rate_rad, rng=rng)}

            gate = PerceptualSupportGate(tau_safe=tau_safe, tau_cam=tau_cam)
            sup = gate.evaluate_corridors(candidates=candidates, critical_regions=crit_regs, rho_predictions=rho_dict)

            dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": True})
            dec = dm.decide(candidates=candidates, frame_id=trial_id, spatial_entities=entities, corridor_support=sup)

            # Measure corridor rejection rate
            for c in candidates:
                if not c.is_stop:
                    total_directional_corridors += 1
                    s_rec = sup.get(c.direction)
                    if s_rec and not s_rec.is_admissible and s_rec.status_label == "INADMISSIBLE_LOW_SUPPORT":
                        rejected_corridors_count += 1

            if dec.command == "STOP":
                stops += 1

            # False-Safe: system commanded turn into unsupported path when true survival failed
            if ("RIGHT" in dec.command) and not actual_survived:
                false_safes += 1
                hazard_selections += 1

            # False-Rejection: corridor support rejected a path when true survival actually succeeded
            if (dec.command == "STOP") and actual_survived:
                false_rejections += 1

        fs_rate = (false_safes / float(trials)) * 100.0
        fr_rate = (false_rejections / float(trials)) * 100.0
        stop_rate = (stops / float(trials)) * 100.0
        corridor_rej_rate = (rejected_corridors_count / float(max(total_directional_corridors, 1))) * 100.0

        rec = {
            "tau_cam": tau_cam,
            "false_safe_rate_pct": round(fs_rate, 2),
            "false_rejection_rate_pct": round(fr_rate, 2),
            "hazardous_selection_rate_pct": round((hazard_selections / float(trials)) * 100.0, 2),
            "stop_rate_pct": round(stop_rate, 2),
            "corridor_rejection_rate_pct": round(corridor_rej_rate, 2),
        }
        records.append(rec)
        print(f"  tau_cam={tau_cam:.2f} | FalseSafe={fs_rate:4.1f}% | FalseRej={fr_rate:4.1f}% | StopRate={stop_rate:4.1f}% | RejCorridors={corridor_rej_rate:4.1f}%")

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Plot
    taus = [r["tau_cam"] for r in records]
    fs_rates = [r["false_safe_rate_pct"] for r in records]
    fr_rates = [r["false_rejection_rate_pct"] for r in records]
    stop_rates = [r["stop_rate_pct"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, fs_rates, marker="o", color="tab:red", linewidth=2, label="False-Safe Rate (%) [Hazard Allowed]")
    ax.plot(taus, fr_rates, marker="s", color="tab:orange", linewidth=2, label="False-Rejection Rate (%) [Unnecessary Stop]")
    ax.plot(taus, stop_rates, marker="^", color="tab:blue", linestyle="--", linewidth=1.5, label="Overall STOP Rate (%)")

    # Highlight current operating point tau_cam = 0.40
    ax.axvline(0.40, color="gray", linestyle=":", label="Current Operating Threshold (0.40)")

    ax.set_xlabel("Perceptual Support Admissibility Threshold (tau_cam)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Operating Trade-off: False-Safe vs False-Rejection vs tau_cam")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Task 7 Output: CSV -> {output_csv}, Plot -> {output_png}")
    return records


if __name__ == "__main__":
    run_n_sensitivity_analysis()
    run_tau_cam_sensitivity_analysis()
