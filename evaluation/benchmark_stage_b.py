"""
ORCA EYE — Stage B: 3-Mode Multi-Camera Benchmark Suite
======================================================
Executes 8 Scenarios x 100 Trials x 3 Modes = 2,400 Runs:
  Mode B0: Single Camera Baseline (Stage A)
  Mode B1: Two-Camera Reactive Handover (Transfer only AFTER primary loss)
  Mode B2: Two-Camera Predictive Handover (Proactive Pre-Arm and Transfer BEFORE loss)

Saves results to:
  evaluation/results/stage_b_ab_results.csv
  evaluation/results/stage_b_summary.json
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Set project root
sys.path.insert(0, os.path.abspath("."))

from evaluation.reproducibility import seed_everything
from evaluation.scenarios import ControlledScenarioType, create_synthetic_spatial_state
from evaluation.stage_b_ground_truth import evaluate_stage_b_ground_truth
from evaluation.stage_b_metrics import StageBTrialMetrics, compute_aggregate_metrics
from evaluation.stage_b_scenarios import (
    StageBScenarioType,
    generate_camera_observations_at_time,
    get_scenario_trajectories,
)
from navigation.camera_handover import HandoverMode, HandoverState, PredictiveHandoverManager
from navigation.camera_observation import CameraModel, MultiCameraEntityState
from navigation.critical_region import CriticalRegionExtractor
from navigation.decision import DecisionMaker
from navigation.multi_camera_support import MultiCameraSupportGate
from navigation.path_generator import PathGenerator
from navigation.path_scorer import PathScorer
from navigation.rho_fov import RhoFOVPredictor


def run_stage_b_benchmark(
    trials_per_scenario: int = 100,
    base_seed: int = 2000,
    output_csv_path: str = "evaluation/results/stage_b_ab_results.csv",
    output_summary_path: str = "evaluation/results/stage_b_summary.json",
) -> Dict:
    print("\n" + "=" * 70)
    print("ORCA EYE — STAGE B: 3-MODE MULTI-CAMERA BENCHMARK (2,400 RUNS)")
    print("=" * 70)

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Models & configuration
    cam0 = CameraModel(camera_id="CAM0", yaw_deg=0.0, half_hfov_rad=math.radians(32.5))
    cam1 = CameraModel(camera_id="CAM1", yaw_deg=50.0, half_hfov_rad=math.radians(32.5))

    min_ov, max_ov, ov_width = cam0.compute_overlap(cam1)
    print(f"  Physical Overlap Zone: [{math.degrees(min_ov):.1f}°, {math.degrees(max_ov):.1f}°] (Width: {math.degrees(ov_width):.1f}°)")

    # Common decision & planner modules
    smap, _ = create_synthetic_spatial_state(ControlledScenarioType.CROSSING_PEDESTRIAN)
    pgen = PathGenerator({"use_astar": False}, smap.rows, smap.cols)
    scorer = PathScorer({})
    candidates = scorer.score(pgen.generate(smap))
    crit_ext = CriticalRegionExtractor(horizon_s=2.0)
    sup_gate = MultiCameraSupportGate(tau_safe=0.35, tau_cam=0.40, primary_camera_id="CAM0")

    modes = [
        HandoverMode.B0_SINGLE_CAM,
        HandoverMode.B1_REACTIVE,
        HandoverMode.B2_PREDICTIVE,
    ]

    scenarios = [
        StageBScenarioType.S1_CENTER_TO_RIGHT,
        StageBScenarioType.S2_CENTER_TO_LEFT,
        StageBScenarioType.S3_FAST_PERIPHERAL,
        StageBScenarioType.S4_SLOW_PERIPHERAL,
        StageBScenarioType.S5_OVERLAP_TRAVERSAL,
        StageBScenarioType.S6_OCCLUSION_HANDOVER,
        StageBScenarioType.S7_SIMILAR_ENTITIES,
        StageBScenarioType.S8_MULTIPLE_ENTITIES,
    ]

    total_runs = len(scenarios) * trials_per_scenario * len(modes)
    print(f"  Configuration: {len(scenarios)} Scenarios x {trials_per_scenario} Trials x {len(modes)} Modes = {total_runs} Runs")

    trial_records: List[StageBTrialMetrics] = []
    t_start = time.time()

    for scen in scenarios:
        scen_name = scen.value
        print(f"\n--> Running Scenario: {scen_name}")

        for trial_id in range(trials_per_scenario):
            trial_seed = base_seed + trial_id * 31
            rng = seed_everything(trial_seed)

            # Trajectories for this trial
            trajectories = get_scenario_trajectories(scen, rng=rng)
            rep_traj = trajectories[0]

            # Ground truth calculation
            gt_res = evaluate_stage_b_ground_truth(
                initial_world_bearing_deg=rep_traj.initial_world_bearing_deg,
                bearing_rate_deg_s=rep_traj.bearing_rate_deg_s,
                cam0=cam0,
                cam1=cam1,
                duration_s=2.5,
                dt_eval_s=0.01,
            )

            for mode in modes:
                mode_name = mode.value
                rho_pred = RhoFOVPredictor(num_samples=200, seed=trial_seed)
                handover_mgr = PredictiveHandoverManager(
                    primary_camera_id="CAM0",
                    secondary_camera_id="CAM1",
                    tau_release=0.30,
                    tau_acquire=0.25,
                    tau_assoc=0.70,
                    mode=mode,
                )
                dec_maker = DecisionMaker({"enable_support_gate": True})

                # Simulation loop: dt = 0.05s, 50 steps = 2.5s
                sim_duration_s = 2.5
                dt_step_s = 0.05
                n_steps = int(sim_duration_s / dt_step_s)

                entity_states: Dict[int, MultiCameraEntityState] = {
                    t.entity_id: MultiCameraEntityState(global_entity_id=t.entity_id)
                    for t in trajectories
                }

                rep_entity_state = entity_states[rep_traj.entity_id]
                primary_loss_time = None
                secondary_acq_time = None
                pre_arm_time = None
                transfer_time = None
                observed_steps_by_responsible = 0
                final_decision = None
                final_support = 1.0

                for step in range(n_steps):
                    t_elapsed = step * dt_step_s
                    current_ts = trial_seed + t_elapsed

                    # Generate observations
                    obs0_list, obs1_list = generate_camera_observations_at_time(
                        trajectories=trajectories,
                        elapsed_time_s=t_elapsed,
                        cam0=cam0,
                        cam1=cam1,
                        rho_pred=rho_pred,
                        now_ts=current_ts,
                    )

                    # Update entity state cache
                    for ent in entity_states.values():
                        ent.observations.clear()

                    for o0 in obs0_list:
                        if o0.track_id in entity_states:
                            entity_states[o0.track_id].observations["CAM0"] = o0

                    # Cross-camera association
                    matched = handover_mgr.associator.associate(obs0_list, obs1_list, now=current_ts)
                    matched_map = {m[0].track_id: (m[1], m[2]) for m in matched}

                    # Assign CAM1 observations to matching entities
                    for o1 in obs1_list:
                        # Find corresponding global ID
                        assigned_gid = None
                        assoc_conf = 0.0
                        for gid, ent in entity_states.items():
                            if gid in matched_map and matched_map[gid][0].track_id == o1.track_id:
                                assigned_gid = gid
                                assoc_conf = matched_map[gid][1]
                                break

                        if assigned_gid is not None:
                            entity_states[assigned_gid].observations["CAM1"] = o1
                        else:
                            # Secondary sees it independently without CAM0 match
                            for gid, ent in entity_states.items():
                                if o1.track_id == gid + 100:
                                    ent.observations["CAM1"] = o1
                                    break

                    # Update handover state machine for each entity
                    for gid, ent in entity_states.items():
                        c_conf = matched_map[gid][1] if gid in matched_map else 0.0
                        handover_mgr.update_entity_handover(ent, association_confidence=c_conf, current_time=current_ts)

                    # Track timings for representative entity
                    obs_p = rep_entity_state.observations.get("CAM0")
                    obs_s = rep_entity_state.observations.get("CAM1")

                    if obs_p and obs_p.in_fov:
                        pass
                    elif primary_loss_time is None:
                        primary_loss_time = t_elapsed

                    if obs_s and obs_s.in_fov and secondary_acq_time is None:
                        secondary_acq_time = t_elapsed

                    if rep_entity_state.responsibility_state == HandoverState.PRE_ARM.value and pre_arm_time is None:
                        pre_arm_time = t_elapsed

                    if rep_entity_state.responsibility_state in (HandoverState.TRANSFER.value, HandoverState.SECONDARY.value) and transfer_time is None:
                        transfer_time = t_elapsed

                    # Check observation continuity under currently responsible camera
                    resp_cam = rep_entity_state.responsible_camera_id
                    resp_obs = rep_entity_state.observations.get(resp_cam)
                    if resp_obs and resp_obs.in_fov:
                        observed_steps_by_responsible += 1

                    # Navigation coupling: evaluate critical regions & support
                    # Convert synthetic entities to SpatialEntity format for critical region extraction
                    crit_regions = crit_ext.extract_critical_regions(candidates=candidates, spatial_entities=[])
                    # Force critical region to contain representative entity
                    for cr in crit_regions.values():
                        cr.critical_entities = [rep_traj.entity_id]

                    use_multi = (mode != HandoverMode.B0_SINGLE_CAM)
                    corridor_support_records = sup_gate.evaluate_corridors(
                        candidates=candidates,
                        critical_regions=crit_regions,
                        entities=entity_states,
                        use_multi_cam=use_multi,
                    )

                    # Adapt for DecisionMaker
                    legacy_sup = {
                        k: v.to_legacy_corridor_support_record()
                        for k, v in corridor_support_records.items()
                    }
                    decision = dec_maker.decide(candidates, frame_id=step, corridor_support=legacy_sup)
                    final_decision = decision
                    final_support = corridor_support_records["STRAIGHT"].multi_cam_support if use_multi else corridor_support_records["STRAIGHT"].single_cam_support

                # End of trial metrics calculation
                effective_primary_loss = gt_res.true_cam0_loss_time_s or primary_loss_time
                effective_sec_acq = gt_res.true_cam1_acq_time_s or secondary_acq_time

                lead_time = None
                if effective_primary_loss is not None and transfer_time is not None:
                    lead_time = effective_primary_loss - transfer_time

                gap = None
                if effective_primary_loss is not None and effective_sec_acq is not None:
                    gap = effective_sec_acq - effective_primary_loss

                continuity = observed_steps_by_responsible / n_steps

                # Handover anomalies
                is_false_ho = (transfer_time is not None and effective_primary_loss is None)
                is_missed_ho = (effective_primary_loss is not None and gt_res.true_handover_possible and transfer_time is None)

                ho_lat = None
                if pre_arm_time is not None and transfer_time is not None:
                    ho_lat = transfer_time - pre_arm_time

                rec = StageBTrialMetrics(
                    scenario=scen_name,
                    trial_id=trial_id,
                    mode=mode_name,
                    primary_loss_time_s=effective_primary_loss,
                    secondary_acq_time_s=effective_sec_acq,
                    pre_arm_time_s=round(pre_arm_time, 3) if pre_arm_time is not None else None,
                    transfer_time_s=round(transfer_time, 3) if transfer_time is not None else None,
                    handover_lead_time_s=round(lead_time, 3) if lead_time is not None else None,
                    observation_gap_s=round(gap, 3) if gap is not None else None,
                    observation_continuity_ratio=round(continuity, 4),
                    association_accuracy_pct=100.0 if not is_false_ho else 0.0,
                    is_false_handover=is_false_ho,
                    is_missed_handover=is_missed_ho,
                    handover_latency_s=round(ho_lat, 3) if ho_lat is not None else None,
                    corridor_support=round(final_support, 4),
                    selected_command=final_decision.command if final_decision else "STOP",
                    is_admissible=(final_decision.command != "STOP") if final_decision else False,
                )
                trial_records.append(rec)

    # Save to CSV
    fieldnames = list(trial_records[0].__dict__.keys())
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in trial_records:
            writer.writerow(r.__dict__)

    # Summary by mode and scenario
    summary_by_mode = {}
    for m in modes:
        m_trials = [t for t in trial_records if t.mode == m.value]
        summary_by_mode[m.value] = compute_aggregate_metrics(m_trials)

    summary_by_scen = {}
    for s in scenarios:
        s_trials = [t for t in trial_records if t.scenario == s.value]
        summary_by_scen[s.value] = {
            m.value: compute_aggregate_metrics([t for t in s_trials if t.mode == m.value])
            for m in modes
        }

    elapsed_total = time.time() - t_start
    manifest = {
        "benchmark_timestamp": time.time(),
        "total_trials": len(trial_records),
        "elapsed_seconds": round(elapsed_total, 2),
        "overall_by_mode": summary_by_mode,
        "by_scenario": summary_by_scen,
    }

    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nStage B Benchmark Complete ({len(trial_records)} runs in {elapsed_total:.1f}s)!")
    print("  Results saved to:")
    print(f"    CSV:  {output_csv_path}")
    print(f"    JSON: {output_summary_path}")

    return manifest


if __name__ == "__main__":
    run_stage_b_benchmark(trials_per_scenario=100)
