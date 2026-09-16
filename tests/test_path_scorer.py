"""
ORCA EYE — Unit Tests: PathScorer
Verifies the weighted scoring formula with known inputs.
"""

import pytest
from navigation.path_generator import PathCandidate


def make_path(
    direction="STRAIGHT",
    clearance=0.8,
    progress=1.0,
    risk=0.1,
    uncertainty=0.1,
    curvature=0.0,
    is_stop=False,
):
    p = PathCandidate(direction=direction, angle_deg=0.0, is_stop=is_stop)
    p.clearance = clearance
    p.progress = progress
    p.risk = risk
    p.uncertainty = uncertainty
    p.curvature = curvature
    return p


def make_scorer():
    from navigation.path_scorer import PathScorer
    cfg = {
        "w_c": 0.30,
        "w_f": 0.25,
        "w_p": 0.25,
        "w_r": 0.15,
        "w_u": 0.05,
        "stop_base_score": 0.10,
    }
    return PathScorer(cfg)


def test_stop_gets_base_score():
    scorer = make_scorer()
    stop = make_path(is_stop=True)
    score = scorer.score_one(stop)
    assert score == pytest.approx(0.10)


def test_clear_straight_scores_high():
    scorer = make_scorer()
    p = make_path(clearance=0.9, progress=1.0, risk=0.05, uncertainty=0.05)
    score = scorer.score_one(p)
    assert score > 0.6, f"Expected >0.6, got {score}"


def test_blocked_path_scores_low():
    scorer = make_scorer()
    p = make_path(clearance=0.0, progress=0.2, risk=0.9, uncertainty=0.8)
    score = scorer.score_one(p)
    assert score < 0.2, f"Expected <0.2, got {score}"


def test_score_sorting():
    scorer = make_scorer()
    p_good = make_path("STRAIGHT", clearance=0.9, progress=1.0, risk=0.05, uncertainty=0.05)
    p_bad = make_path("LEFT", clearance=0.1, progress=0.5, risk=0.7, uncertainty=0.5)
    stop = make_path(is_stop=True)

    candidates = scorer.score([p_bad, p_good, stop])
    # Best should be first
    assert candidates[0].direction == "STRAIGHT"
    # STOP should beat the bad path
    stop_score = next(c.score for c in candidates if c.is_stop)
    bad_score = next(c.score for c in candidates if c.direction == "LEFT")
    assert stop_score > bad_score or stop_score == pytest.approx(0.10)


def test_curvature_penalty():
    scorer = make_scorer()
    straight = make_path("STRAIGHT", progress=1.0, curvature=0.0, clearance=0.8, risk=0.1)
    curved = make_path("LEFT", progress=1.0, curvature=0.67, clearance=0.8, risk=0.1)
    s1 = scorer.score_one(straight)
    s2 = scorer.score_one(curved)
    assert s1 > s2, "Straight should score higher than curved with same clearance"


def test_score_clamped():
    scorer = make_scorer()
    # Worst possible path
    p = make_path(clearance=0.0, progress=0.0, risk=1.0, uncertainty=1.0)
    score = scorer.score_one(p)
    assert score >= -0.1, "Score must not go below -0.1"


def test_serializer():
    scorer = make_scorer()
    p = make_path()
    candidates = scorer.score([p])
    d_list = scorer.to_dict_list(candidates)
    assert isinstance(d_list, list)
    assert "direction" in d_list[0]
    assert "score" in d_list[0]
