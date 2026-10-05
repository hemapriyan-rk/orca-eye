"""
ORCA EYE — Stage A: Reproducibility & Seeding Framework
========================================================
Provides deterministic random seed management and structured experiment logging
for scientific verification of Stage-A Navigation-Coupled Perceptual Support.
"""

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


def seed_everything(seed: int = 42) -> np.random.Generator:
    """
    Set deterministic seeds across Python's random and NumPy.
    Returns a newly seeded NumPy Generator for isolated stochastic sampling.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


@dataclass
class TrialRecord:
    """Structured record for a single stochastic evaluation trial."""
    scenario: str
    trial_id: int
    mode: str                            # "B0_BASELINE" or "B1_RHO_FOV"
    seed: int
    n_samples: int
    dt_s: float
    horizon_s: float
    tau_safe: float
    tau_cam: float
    initial_bearing_rad: float
    initial_bearing_rate_rad_s: float
    state_uncertainty: float
    rho_fov: float
    support: float
    is_admissible: bool
    actual_fov_survival: bool            # Ground-truth: stayed inside FOV over horizon H
    actual_continuous_survival: bool     # Ground-truth: stayed in FOV and q_BR >= q_min over H
    time_to_actual_fov_loss_s: Optional[float]
    selected_corridor: str
    command: str
    score: float
    clearance: float
    is_false_safe: bool                  # Selected maneuver when actual survival failed
    is_false_rejection: bool             # Rejected maneuver when actual survival passed
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
