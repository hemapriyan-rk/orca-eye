"""
ORCA EYE — Stage A: Calibration & Brier Score Analysis
======================================================
Task 9: Evaluates predicted rho_FOV against empirical ground-truth survival outcomes.
Computes:
  - Binned empirical survival vs predicted probability (5 bins in [0, 1]).
  - Brier Score: (1/M) * sum((rho_m - y_m)^2)
  - Expected Calibration Error (ECE)
  - Reliability Diagram Plot

Outputs:
  evaluation/results/rho_calibration.csv
  evaluation/results/rho_calibration.png
"""

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.ground_truth import evaluate_ground_truth_survival
from evaluation.reproducibility import seed_everything
from navigation.rho_fov import RhoFOVPredictor


def run_calibration_evaluation(
    num_samples: int = 500,
    base_seed: int = 42,
    output_csv: str = "evaluation/results/rho_calibration.csv",
    output_png: str = "evaluation/results/rho_calibration.png",
) -> Dict:
    print("\n" + "=" * 60)
    print("Running Task 9: rho_FOV Calibration & Brier Score Evaluation")
    print("=" * 60)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(base_seed)

    predicted_rhos: List[float] = []
    actual_outcomes: List[int] = []

    # Predictor under test
    predictor = RhoFOVPredictor(num_samples=200, seed=base_seed)

    # Generate diverse states across the full continuum:
    # 1. Safe inside FOV (low rate) -> rho ~ 1.0
    # 2. Definite exit (high rate near boundary) -> rho ~ 0.0
    # 3. Intermediate boundary / stochastic uncertainty -> rho in (0.1, 0.9)
    for _ in range(num_samples):
        # Sample across full bearing range [-35 deg, +35 deg]
        b_deg = float(rng.uniform(-34.0, 34.0))
        # Sample rates [0, 30 deg/s] with directional bias toward or away from edge
        rate_deg_s = float(rng.uniform(0.0, 25.0))
        if b_deg > 0:
            # Positive rate moves outward; negative moves inward
            if rng.random() > 0.4:
                rate_deg_s = rate_deg_s  # outward
            else:
                rate_deg_s = -rate_deg_s  # inward
        else:
            if rng.random() > 0.4:
                rate_deg_s = -rate_deg_s # outward (left)
            else:
                rate_deg_s = rate_deg_s  # inward

        b_rad = math.radians(b_deg)
        rate_rad = math.radians(rate_deg_s)

        # Ground truth outcome: true physical reference trajectory over horizon H=2.0s
        gt = evaluate_ground_truth_survival(
            initial_bearing_rad=b_rad,
            initial_bearing_rate_rad_s=rate_rad,
            horizon_s=2.0,
        )
        y_true = 1 if gt.stayed_continuous_valid else 0

        # Model prediction
        pred = predictor.predict_survival(1, b_rad, rate_rad, rng=rng)
        predicted_rhos.append(pred.rho_fov)
        actual_outcomes.append(y_true)

    preds = np.array(predicted_rhos)
    y = np.array(actual_outcomes)

    # Calculate overall Brier Score: MSE between probability and binary outcome
    brier_score = float(np.mean((preds - y) ** 2))

    # Binning: 5 bins in [0, 1]
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_records = []
    ece = 0.0

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            in_bin = (preds >= low) & (preds <= high)
        else:
            in_bin = (preds >= low) & (preds < high)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_pred = float(np.mean(preds[in_bin]))
            empirical_prob = float(np.mean(y[in_bin]))
            calib_err = abs(mean_pred - empirical_prob)
            ece += (count / float(num_samples)) * calib_err
        else:
            mean_pred = (low + high) / 2.0
            empirical_prob = 0.0
            calib_err = 0.0

        bin_records.append({
            "bin_range": f"[{low:.1f}, {high:.1f}]",
            "sample_count": count,
            "mean_predicted_rho": round(mean_pred, 4),
            "empirical_survival_prob": round(empirical_prob, 4),
            "absolute_calibration_error": round(calib_err, 4),
        })

    print(f"Total Samples Evaluated       : {num_samples}")
    print(f"Overall Brier Score           : {brier_score:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    for b in bin_records:
        print(f"  Bin {b['bin_range']:12s} | N={b['sample_count']:4d} | Pred={b['mean_predicted_rho']:.4f} | True={b['empirical_survival_prob']:.4f} | Err={b['absolute_calibration_error']:.4f}")

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_records[0].keys()))
        writer.writeheader()
        for r in bin_records:
            writer.writerow(r)

    # Plot Reliability Diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Reliability diagram
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    emp_probs = [r["empirical_survival_prob"] for r in bin_records]
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax1.plot(bin_centers, emp_probs, marker="o", color="tab:blue", linewidth=2, label=f"rho_FOV (ECE={ece:.3f})")
    ax1.set_xlabel("Mean Predicted rho_FOV")
    ax1.set_ylabel("Empirical Survival Probability")
    ax1.set_title(f"Reliability Diagram (Brier Score = {brier_score:.4f})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    # Sample distribution histogram
    counts = [r["sample_count"] for r in bin_records]
    ax2.bar(bin_centers, counts, width=0.15, color="tab:gray", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted rho_FOV Bins")
    ax2.set_ylabel("Sample Count")
    ax2.set_title(f"Sample Count Distribution (Total N={num_samples})")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Task 9 Output: CSV -> {output_csv}, Plot -> {output_png}")

    return {
        "brier_score": round(brier_score, 4),
        "ece": round(ece, 4),
        "bins": bin_records,
    }


if __name__ == "__main__":
    run_calibration_evaluation()
