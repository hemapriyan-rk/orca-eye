"""
ORCA EYE — Unit Tests: FailureDetector
Verifies that failure conditions are correctly detected.
"""

import time
import numpy as np
import pytest
from failure_analysis.detector import FailureDetector, FailureCode
from navigation.decision import NavigationDecision
from perception.detector import DetectionResult, ObjectState
from perception.freespace import FreeSpaceResult, LABEL_FREE, LABEL_OBSTACLE, LABEL_UNKNOWN
from perception.depth import DepthResult
from navigation.path_generator import PathCandidate


def make_cfg(tmp_path=None):
    return {
        "failure_detection": {
            "low_clearance_threshold": 0.15,
            "low_detection_confidence": 0.30,
            "high_uncertainty_threshold": 0.60,
            "depth_inconsistency_threshold": 0.25,
            "rapid_change_window": 5,
            "unknown_dominance_threshold": 0.55,
            "save_snapshots": False,  # disable disk writes in tests
        },
        "logging": {
            "failure_dir": str(tmp_path) if tmp_path else "failure_cases_test",
        },
    }


def make_decision(command="STRAIGHT", score=0.7, clearance=0.6, uncertainty=0.1,
                  is_stop=False, is_caution=False, reason="Best scored path"):
    return NavigationDecision(
        command=command,
        selected_path_direction=command,
        score=score,
        clearance=clearance,
        uncertainty=uncertainty,
        reason=reason,
        is_stop=is_stop,
        is_caution=is_caution,
    )


def make_detection_result(n=2, conf=0.8, is_low_conf=False):
    objs = [
        ObjectState(0, "person", conf, [0, 0, 100, 200], [50, 100], 20000.0)
        for _ in range(n)
    ]
    return DetectionResult(
        frame_id=1, timestamp=0.0, inference_ms=15.0,
        num_detections=n, objects=objs,
        mean_confidence=conf, is_low_confidence=is_low_conf,
    )


def make_freespace(uncertainty=0.1, free_frac=0.7, unk_frac=0.1):
    H, W = 480, 640
    label_map = np.zeros((H, W), dtype=np.uint8)
    return FreeSpaceResult(
        label_map=label_map,
        free_prob_map=np.ones((H, W), dtype=np.float32) * 0.7,
        uncertainty=uncertainty,
        statistics={
            "free_fraction": free_frac,
            "obstacle_fraction": 1 - free_frac - unk_frac,
            "unknown_fraction": unk_frac,
            "uncertainty": uncertainty,
        },
    )


def make_depth(mean=0.5, valid=True):
    dm = np.full((480, 640), mean, dtype=np.float32)
    return DepthResult(
        depth_map=dm, raw_depth=dm, inference_ms=0.0,
        statistics={"mean": mean, "std": 0.0, "min": mean, "max": mean, "uncertainty": 0.0},
        is_valid=valid,
    )


def make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_no_failure_clean_scene(tmp_path):
    fd = FailureDetector(make_cfg(tmp_path))
    decision = make_decision()
    det = make_detection_result()
    fs = make_freespace()
    depth = make_depth()
    result = fd.check(1, make_frame(), decision, det, [], fs, depth, {}, [])
    # Clean scene should produce no failure (or only minor ones)
    if result is not None:
        # Accept only LOW severity for clean scene edge cases
        assert result.severity in ("LOW",)


def test_freespace_unstable_flagged(tmp_path):
    fd = FailureDetector(make_cfg(tmp_path))
    decision = make_decision()
    det = make_detection_result()
    fs = make_freespace(uncertainty=0.75)  # above 0.60 threshold
    depth = make_depth()
    result = fd.check(1, make_frame(), decision, det, [], fs, depth, {}, [])
    assert result is not None
    assert FailureCode.FREESPACE_UNSTABLE in result.failure_codes


def test_low_clearance_flagged(tmp_path):
    fd = FailureDetector(make_cfg(tmp_path))
    decision = make_decision(command="STRAIGHT", clearance=0.05)  # below 0.15
    det = make_detection_result()
    fs = make_freespace()
    depth = make_depth()
    result = fd.check(1, make_frame(), decision, det, [], fs, depth, {}, [])
    assert result is not None
    assert FailureCode.LOW_CLEARANCE in result.failure_codes


def test_unknown_dominance_flagged(tmp_path):
    fd = FailureDetector(make_cfg(tmp_path))
    decision = make_decision()
    det = make_detection_result()
    fs = make_freespace(unk_frac=0.65, uncertainty=0.3)  # above 0.55
    depth = make_depth()
    result = fd.check(1, make_frame(), decision, det, [], fs, depth, {}, [])
    assert result is not None
    assert FailureCode.UNKNOWN_DOMINANCE in result.failure_codes


def test_depth_inconsistency_flagged(tmp_path):
    fd = FailureDetector(make_cfg(tmp_path))
    decision = make_decision()
    det = make_detection_result()
    fs = make_freespace()
    depth1 = make_depth(mean=0.5)
    depth2 = make_depth(mean=0.9)  # jump of 0.4 > threshold 0.25
    # First frame establishes baseline
    fd.check(1, make_frame(), decision, det, [], fs, depth1, {}, [])
    # Second frame should flag inconsistency
    result = fd.check(2, make_frame(), decision, det, [], fs, depth2, {}, [])
    assert result is not None
    assert FailureCode.DEPTH_INCONSISTENCY in result.failure_codes


def test_summary_counts(tmp_path):
    fd = FailureDetector(make_cfg(tmp_path))
    decision = make_decision(clearance=0.05)
    det = make_detection_result()
    fs = make_freespace()
    depth = make_depth()
    fd.check(1, make_frame(), decision, det, [], fs, depth, {}, [])
    summary = fd.get_summary()
    assert "total_failures" in summary
    assert summary["total_failures"] >= 1
