"""
ORCA EYE — Stage B: Quantitative Handover & Perceptual Metrics
=============================================================
Calculates core scientific metrics:
  1. Handover Lead Time: T_lead = t_actual_loss - t_transfer
  2. Observation Gap: G = t_sec_acq - t_prim_loss
  3. Observation Continuity: C_e
  4. Association Accuracy
  5. False & Missed Handover Rates
  6. Handover State Transition Latencies
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StageBTrialMetrics:
    """Quantitative performance record for a single evaluation trial."""
    scenario: str
    trial_id: int
    mode: str                                  # "B0_SINGLE_CAM" | "B1_REACTIVE" | "B2_PREDICTIVE"
    primary_loss_time_s: Optional[float]
    secondary_acq_time_s: Optional[float]
    pre_arm_time_s: Optional[float]
    transfer_time_s: Optional[float]
    handover_lead_time_s: Optional[float]      # T_lead (positive = proactive before loss)
    observation_gap_s: Optional[float]         # G (positive = blackout period, <= 0 = seamless)
    observation_continuity_ratio: float        # Fraction of duration entity was tracked by responsible camera
    association_accuracy_pct: float            # 100% if correct global ID maintained
    is_false_handover: bool                    # Handover occurred but primary never lost target
    is_missed_handover: bool                   # Primary lost target but secondary never assumed responsibility
    handover_latency_s: Optional[float]        # Elapsed time from PRE_ARM to TRANSFER
    corridor_support: float                    # Support_B value
    selected_command: str                      # Decision commanded by planner
    is_admissible: bool


def compute_aggregate_metrics(trials: List[StageBTrialMetrics]) -> Dict[str, float]:
    """
    Computes summary distribution statistics across an evaluation suite.
    """
    if not trials:
        return {}

    lead_times = [t.handover_lead_time_s for t in trials if t.handover_lead_time_s is not None]
    gaps = [t.observation_gap_s for t in trials if t.observation_gap_s is not None]
    continuities = [t.observation_continuity_ratio for t in trials]
    latencies = [t.handover_latency_s for t in trials if t.handover_latency_s is not None]

    false_handovers = sum(1 for t in trials if t.is_false_handover)
    missed_handovers = sum(1 for t in trials if t.is_missed_handover)

    return {
        "trial_count": len(trials),
        "mean_lead_time_s": float(np.mean(lead_times)) if lead_times else 0.0,
        "median_lead_time_s": float(np.median(lead_times)) if lead_times else 0.0,
        "p95_lead_time_s": float(np.percentile(lead_times, 95)) if lead_times else 0.0,
        "mean_observation_gap_s": float(np.mean(gaps)) if gaps else 0.0,
        "median_observation_gap_s": float(np.median(gaps)) if gaps else 0.0,
        "mean_continuity_ratio": float(np.mean(continuities)),
        "p5_continuity_ratio": float(np.percentile(continuities, 5)),
        "mean_handover_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "false_handover_rate_pct": round((false_handovers / len(trials)) * 100.0, 2),
        "missed_handover_rate_pct": round((missed_handovers / len(trials)) * 100.0, 2),
    }
