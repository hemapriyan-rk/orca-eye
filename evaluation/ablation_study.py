"""
ORCA EYE — Stage A: Controlled Ablation Study
==============================================
Task 8: 5-Level Controlled Ablation:
  Level A: Safety-only baseline (no FOV checks).
  Level B: Safety + Instantaneous hard FOV boundary (|phi| <= theta_c).
  Level C: Safety + Instantaneous FOV + bearing rate (|phi| <= theta_c and q_BR >= q_min).
  Level D: Safety + Continuous q_BR + Monte Carlo future propagation (soft score penalty, un-gated).
  Level E: Full Stage-A rho_FOV + Hard Perceptual Support Admissibility Gate.

Outputs:
  evaluation/results/ablation_results.csv
  evaluation/results/ablation.png
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.ground_truth import evaluate_ground_truth_survival
from evaluation.reproducibility import seed_everything
from navigation.camera_support import PerceptualSupportGate
from navigation.critical_region import CriticalRegionExtractor
from navigation.decision import DecisionMaker
from navigation.geometry import bearing_rate_quality, is_in_fov
from navigation.path_generator import PathGenerator
from navigation.path_scorer import PathScorer
from navigation.rho_fov import RhoFOVPredictor
from navigation.spatial_entity import SpatialEntity
from navigation.spatial_map import SpatialMap


def run_ablation_experiment(
    trials: int = 100,
    base_seed: int = 42,
    tau_safe: float = 0.35,
    tau_cam: float = 0.40,
    half_hfov_rad: float = math.radians(32.5),
    output_csv: str = "evaluation/results/ablation_results.csv",
    output_png: str = "evaluation/results/ablation.png",
) -> List[Dict]:
    print("\n" + "=" * 60)
    print("Running Task 8: Controlled 5-Level Ablation Study (100 trials)")
    print("=" * 60)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    ablation_levels = [
        ("A_SAFETY_ONLY", "Safety-Only Baseline"),
        ("B_INSTANT_FOV", "Safety + Instant FOV"),
        ("C_INSTANT_FOV_BR", "Safety + Instant FOV & Rate"),
        ("D_SOFT_MC_PROP", "Safety + Soft MC Prop (Un-gated)"),
        ("E_FULL_STAGE_A", "Full Stage-A Admissibility Gate"),
    ]

    records: List[Dict] = []

    for level_code, level_name in ablation_levels:
        hazardous_selections = 0
        false_safes = 0
        false_rejections = 0
        observation_losses = 0
        stops = 0
        latencies = []

        for trial_id in range(trials):
            trial_seed = base_seed + trial_id * 19
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

            b_deg = float(rng.uniform(26.0, 31.0))
            rate_deg_s = float(rng.uniform(11.0, 18.0))
            b_rad = math.radians(b_deg)
            rate_rad = math.radians(rate_deg_s)

            entity = SpatialEntity(
                track_id=1, class_id=0, class_name="person", confidence=0.95,
                bbox=[150, 480, 250, 600], image_position=[540.0, 200.0], ground_position=[540.0, 400.0],
                relative_bearing=b_deg, relative_distance=0.35, velocity=[15.0, 0.0],
                tracking_confidence=0.95, depth_confidence=0.90, depth_valid=True, navigation_relevance=0.95,
            )
            entities = [entity]

            # Ground truth reference
            gt = evaluate_ground_truth_survival(b_rad, rate_rad, horizon_s=2.0)
            actual_survived = gt.stayed_continuous_valid

            pgen = PathGenerator({"use_astar": False}, 12, 20)
            scorer = PathScorer({})
            candidates = scorer.score(pgen.generate(smap))

            t0 = time.perf_counter()

            # Execute ablation level logic
            if level_code == "A_SAFETY_ONLY":
                # Mode B0: no support gate
                dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": False})
                dec = dm.decide(candidates=candidates, frame_id=trial_id, spatial_entities=entities)

            elif level_code == "B_INSTANT_FOV":
                # Check instantaneous FOV at t=0 only
                instant_in_fov = is_in_fov(b_rad, half_hfov_rad)
                # If in FOV at t=0, allow directional; if not, reject
                corridor_sup = {}
                for c in candidates:
                    from navigation.camera_support import CorridorSupportRecord
                    status = "ADMISSIBLE" if (c.is_stop or instant_in_fov) else "INADMISSIBLE_LOW_SUPPORT"
                    corridor_sup[c.direction] = CorridorSupportRecord(
                        corridor_direction=c.direction, safety_score=c.score,
                        perceptual_support=1.0 if instant_in_fov else 0.0,
                        is_admissible=(status == "ADMISSIBLE"), status_label=status,
                    )
                dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": True})
                dec = dm.decide(candidates=candidates, frame_id=trial_id, spatial_entities=entities, corridor_support=corridor_sup)

            elif level_code == "C_INSTANT_FOV_BR":
                # Check instantaneous FOV and q_BR at t=0
                instant_in_fov = is_in_fov(b_rad, half_hfov_rad)
                q_0 = bearing_rate_quality(rate_rad)
                instant_valid = instant_in_fov and (q_0 >= 0.20)
                corridor_sup = {}
                for c in candidates:
                    from navigation.camera_support import CorridorSupportRecord
                    status = "ADMISSIBLE" if (c.is_stop or instant_valid) else "INADMISSIBLE_LOW_SUPPORT"
                    corridor_sup[c.direction] = CorridorSupportRecord(
                        corridor_direction=c.direction, safety_score=c.score,
                        perceptual_support=q_0 if instant_in_fov else 0.0,
                        is_admissible=(status == "ADMISSIBLE"), status_label=status,
                    )
                dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": True})
                dec = dm.decide(candidates=candidates, frame_id=trial_id, spatial_entities=entities, corridor_support=corridor_sup)

            elif level_code == "D_SOFT_MC_PROP":
                # Soft penalty in score without hard gating
                crit_ext = CriticalRegionExtractor()
                crit_regs = crit_ext.extract_critical_regions(candidates=candidates, spatial_entities=entities)
                rho_pred = RhoFOVPredictor(num_samples=200, seed=trial_seed)
                rho_val = rho_pred.predict_survival(1, b_rad, rate_rad, rng=rng).rho_fov
                # Soft penalize candidates with critical entities: score = score * (0.5 + 0.5 * rho_val)
                penalized_candidates = []
                import copy
                for c in candidates:
                    c_copy = copy.copy(c)
                    if "RIGHT" in c.direction:
                        c_copy.score = c.score * (0.4 + 0.6 * rho_val)
                    penalized_candidates.append(c_copy)
                penalized_candidates.sort(key=lambda x: x.score, reverse=True)
                dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": False})
                dec = dm.decide(candidates=penalized_candidates, frame_id=trial_id, spatial_entities=entities)

            elif level_code == "E_FULL_STAGE_A":
                # Full Stage-A hard admissibility gate
                crit_ext = CriticalRegionExtractor()
                crit_regs = crit_ext.extract_critical_regions(candidates=candidates, spatial_entities=entities)
                rho_pred = RhoFOVPredictor(num_samples=200, seed=trial_seed)
                rho_dict = {1: rho_pred.predict_survival(1, b_rad, rate_rad, rng=rng)}
                gate = PerceptualSupportGate(tau_safe=tau_safe, tau_cam=tau_cam)
                sup = gate.evaluate_corridors(candidates=candidates, critical_regions=crit_regs, rho_predictions=rho_dict)
                dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": True})
                dec = dm.decide(candidates=candidates, frame_id=trial_id, spatial_entities=entities, corridor_support=sup)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            if dec.command == "STOP":
                stops += 1

            # Hazard selection: commanded turn into the path when true survival failed
            if ("RIGHT" in dec.command) and not actual_survived:
                hazardous_selections += 1
                false_safes += 1
                observation_losses += 1

            # False rejection: stopped even though target observation was actually safe
            if (dec.command == "STOP") and actual_survived:
                false_rejections += 1

        rec = {
            "level_code": level_code,
            "level_name": level_name,
            "hazardous_selections_pct": round((hazardous_selections / float(trials)) * 100.0, 2),
            "false_safe_rate_pct": round((false_safes / float(trials)) * 100.0, 2),
            "false_rejection_rate_pct": round((false_rejections / float(trials)) * 100.0, 2),
            "observation_loss_pct": round((observation_losses / float(trials)) * 100.0, 2),
            "stop_rate_pct": round((stops / float(trials)) * 100.0, 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 4),
        }
        records.append(rec)
        print(f"  {level_code:18s} | Hazard={rec['hazardous_selections_pct']:5.1f}% | FalseSafe={rec['false_safe_rate_pct']:5.1f}% | FalseRej={rec['false_rejection_rate_pct']:5.1f}% | STOP={rec['stop_rate_pct']:5.1f}% | Latency={rec['mean_latency_ms']:.3f} ms")

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Plot
    labels = [r["level_name"] for r in records]
    hazards = [r["hazardous_selections_pct"] for r in records]
    stops_val = [r["stop_rate_pct"] for r in records]
    latencies_val = [r["mean_latency_ms"] for r in records]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    rects1 = ax1.bar(x - width/2, hazards, width, label="Hazardous Selection Rate (%) [Blind Turn]", color="tab:red")
    rects2 = ax1.bar(x + width/2, stops_val, width, label="Protective STOP Rate (%)", color="tab:blue")

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Ablation Study: Progression of Navigation Safety Across 5 Formulation Levels")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Task 8 Output: CSV -> {output_csv}, Plot -> {output_png}")
    return records


if __name__ == "__main__":
    from navigation.path_generator import PathCandidate
    run_ablation_experiment()
