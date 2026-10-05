"""
ORCA EYE — Stage A: 7-Scenario Controlled Scientific Benchmark
==============================================================
Runs 100 stochastic trials across all 7 controlled navigation scenarios
under both Mode B0 (Baseline) and Mode B1 (Stage-A Support Gated):
  Total = 7 scenarios x 100 trials x 2 modes = 1,400 runs.

Outputs:
  evaluation/results/stage_a_ab_results.csv
"""

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))

import numpy as np

from evaluation.ground_truth import evaluate_ground_truth_survival
from evaluation.reproducibility import TrialRecord, seed_everything
from evaluation.scenarios import (
    CONTROLLED_SUITE,
    ControlledScenarioType,
    create_synthetic_spatial_state,
)
from navigation.camera_support import PerceptualSupportGate
from navigation.critical_region import CriticalRegionExtractor
from navigation.decision import DecisionMaker
from navigation.path_generator import PathGenerator
from navigation.path_scorer import PathScorer
from navigation.rho_fov import RhoFOVPredictor
from navigation.spatial_entity import SpatialEntity


def run_benchmark(
    trials_per_scenario: int = 100,
    base_seed: int = 1000,
    output_csv_path: str = "evaluation/results/stage_a_ab_results.csv",
    tau_safe: float = 0.35,
    tau_cam: float = 0.40,
    n_samples: int = 200,
    dt_s: float = 0.20,
    horizon_s: float = 2.0,
) -> List[TrialRecord]:
    """
    Execute 7 scenarios x 100 trials x 2 modes.
    """
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    all_records: List[TrialRecord] = []

    scenario_types = [
        ControlledScenarioType.CLEAR_PATH,
        ControlledScenarioType.CENTRAL_STATIC_OBSTACLE,
        ControlledScenarioType.LEFT_BLOCKED_RIGHT_OPEN,
        ControlledScenarioType.RIGHT_BLOCKED_LEFT_OPEN,
        ControlledScenarioType.BOTH_SIDES_BLOCKED,
        ControlledScenarioType.CROSSING_PEDESTRIAN,
        ControlledScenarioType.BLIND_UNCERTAINTY,
    ]

    print(f"Starting 7-Scenario Benchmark: {len(scenario_types)} scenarios x {trials_per_scenario} trials x 2 modes = {len(scenario_types) * trials_per_scenario * 2} runs")

    for scen_type in scenario_types:
        scen_name = scen_type.value

        for trial_id in range(trials_per_scenario):
            trial_seed = base_seed + trial_id * 17
            rng = seed_everything(trial_seed)

            # 1. Base spatial state
            smap, entities = create_synthetic_spatial_state(scen_type)

            # Apply realistic stochastic variations per trial
            entity_kinematics: Dict[int, Tuple[float, float]] = {}

            if scen_type == ControlledScenarioType.CROSSING_PEDESTRIAN:
                # Fast crossing pedestrian with bearing rate noise
                b_noise = float(rng.normal(0.0, 1.5))
                rate_noise = float(rng.normal(0.0, 2.0))
                b_deg = -20.0 + b_noise
                rate_deg_s = 18.0 + rate_noise
                for e in entities:
                    e.relative_bearing = b_deg
                    entity_kinematics[e.track_id] = (math.radians(b_deg), math.radians(rate_deg_s))

            elif scen_type == ControlledScenarioType.BLIND_UNCERTAINTY:
                # Add boundary-exiting entity at right FOV flank (+27 deg, +14 deg/s outward)
                b_noise = float(rng.normal(0.0, 1.2))
                rate_noise = float(rng.normal(0.0, 2.0))
                b_deg = 27.5 + b_noise
                rate_deg_s = 14.0 + rate_noise
                e = SpatialEntity(
                    track_id=1,
                    class_id=0,
                    class_name="person",
                    confidence=0.95,
                    bbox=[150, 480, 250, 600],
                    image_position=[540.0, 200.0],
                    ground_position=[540.0, 400.0],
                    relative_bearing=b_deg,
                    relative_distance=0.35,
                    velocity=[15.0, 0.0],
                    tracking_confidence=0.95,
                    depth_confidence=0.90,
                    depth_valid=True,
                    navigation_relevance=0.95,
                )
                entities = [e]
                entity_kinematics[1] = (math.radians(b_deg), math.radians(rate_deg_s))

            elif entities:
                for e in entities:
                    b_noise = float(rng.normal(0.0, 0.5))
                    b_deg = e.relative_bearing + b_noise
                    e.relative_bearing = b_deg
                    entity_kinematics[e.track_id] = (math.radians(b_deg), 0.0)

            # Evaluate independent ground truth survival for entities
            gt_survivals = {}
            for eid, (b_rad, rate_rad) in entity_kinematics.items():
                gt_res = evaluate_ground_truth_survival(
                    initial_bearing_rad=b_rad,
                    initial_bearing_rate_rad_s=rate_rad,
                    horizon_s=horizon_s,
                    q_min=0.20,
                )
                gt_survivals[eid] = gt_res

            # Generate and score candidate corridors
            pgen = PathGenerator({"use_astar": False}, smap.rows, smap.cols)
            scorer = PathScorer({})
            candidates = scorer.score(pgen.generate(smap))

            # Critical regions & rho_FOV
            crit_ext = CriticalRegionExtractor(horizon_s=horizon_s)
            crit_regs = crit_ext.extract_critical_regions(candidates=candidates, spatial_entities=entities)

            rho_pred = RhoFOVPredictor(num_samples=n_samples, horizon_s=horizon_s, dt_sample_s=dt_s, seed=trial_seed)
            rho_dict = {}
            for eid, (b_rad, rate_rad) in entity_kinematics.items():
                rho_dict[eid] = rho_pred.predict_survival(eid, b_rad, rate_rad, rng=rng)

            gate = PerceptualSupportGate(tau_safe=tau_safe, tau_cam=tau_cam)
            corridor_support = gate.evaluate_corridors(
                candidates=candidates, critical_regions=crit_regs, rho_predictions=rho_dict
            )

            # Evaluate under both modes
            for mode_name, enable_gate in [("B0_BASELINE", False), ("B1_RHO_FOV", True)]:
                t0 = time.perf_counter()
                dm = DecisionMaker({"min_go_score": tau_safe, "enable_support_gate": enable_gate})
                dec = dm.decide(
                    candidates=candidates,
                    frame_id=trial_id,
                    spatial_entities=entities,
                    corridor_support=corridor_support,
                    enable_support_gate=enable_gate,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                sup_rec = corridor_support.get(dec.command)
                chosen_support = sup_rec.perceptual_support if sup_rec else 1.0
                is_admissible = sup_rec.is_admissible if sup_rec else True

                # Representative entity stats
                if entities and 1 in entity_kinematics:
                    rep_eid = 1
                elif entities:
                    rep_eid = entities[0].track_id
                else:
                    rep_eid = None

                if rep_eid and rep_eid in gt_survivals:
                    gt_item = gt_survivals[rep_eid]
                    actual_fov = gt_item.stayed_in_fov
                    actual_cont = gt_item.stayed_continuous_valid
                    time_to_loss = gt_item.time_to_fov_loss_s
                    init_b = entity_kinematics[rep_eid][0]
                    init_br = entity_kinematics[rep_eid][1]
                    rep_rho = rho_dict[rep_eid].rho_fov if rep_eid in rho_dict else 1.0
                else:
                    actual_fov = True
                    actual_cont = True
                    time_to_loss = None
                    init_b = 0.0
                    init_br = 0.0
                    rep_rho = 1.0

                # False-safe: chose a moving command toward/involving an entity whose observation actually failed
                is_false_safe = (dec.command != "STOP") and (not actual_cont) and ("RIGHT" in dec.command if scen_type == ControlledScenarioType.BLIND_UNCERTAINTY else False)
                # False-rejection: rejected candidate when actual observation was completely safe
                is_false_rejection = (dec.command == "STOP") and actual_cont and (scen_type in (ControlledScenarioType.CLEAR_PATH, ControlledScenarioType.LEFT_BLOCKED_RIGHT_OPEN))

                rec = TrialRecord(
                    scenario=scen_name,
                    trial_id=trial_id,
                    mode=mode_name,
                    seed=trial_seed,
                    n_samples=n_samples,
                    dt_s=dt_s,
                    horizon_s=horizon_s,
                    tau_safe=tau_safe,
                    tau_cam=tau_cam,
                    initial_bearing_rad=round(init_b, 4),
                    initial_bearing_rate_rad_s=round(init_br, 4),
                    state_uncertainty=0.05,
                    rho_fov=round(rep_rho, 4),
                    support=round(chosen_support, 4),
                    is_admissible=is_admissible,
                    actual_fov_survival=actual_fov,
                    actual_continuous_survival=actual_cont,
                    time_to_actual_fov_loss_s=time_to_loss,
                    selected_corridor=dec.command,
                    command=dec.command,
                    score=round(dec.score, 4),
                    clearance=round(dec.clearance, 4),
                    is_false_safe=is_false_safe,
                    is_false_rejection=is_false_rejection,
                    latency_ms=round(latency_ms, 3),
                )
                all_records.append(rec)

    # Write CSV
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].to_dict().keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(r.to_dict())

    print(f"Benchmark completed successfully! Total records: {len(all_records)} written to {output_csv_path}")
    return all_records


if __name__ == "__main__":
    run_benchmark()
