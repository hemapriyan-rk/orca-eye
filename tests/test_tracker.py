"""
ORCA EYE — Unit Tests: CentroidTracker
Verifies track creation, update, velocity, and expiry.
"""

import pytest
from perception.tracker import CentroidTracker
from perception.detector import ObjectState


def make_cfg():
    return {
        "max_disappeared": 3,
        "max_distance": 200,
        "iou_weight": 0.6,
        "centroid_weight": 0.4,
        "velocity_alpha": 0.3,
        "min_track_age": 2,
    }


def make_det(class_id=0, class_name="person", conf=0.9, x1=100, y1=100, x2=200, y2=300):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return ObjectState(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bbox=[x1, y1, x2, y2],
        center=[cx, cy],
        area=float((x2 - x1) * (y2 - y1)),
    )


def test_single_track_created():
    tracker = CentroidTracker(make_cfg())
    det = make_det()
    tracks = tracker.update([det], frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert tracks[0].age == 1
    assert tracks[0].class_name == "person"


def test_track_continuity():
    tracker = CentroidTracker(make_cfg())
    d1 = make_det(x1=100, y1=100, x2=200, y2=300)  # center ~ (150, 200)
    d2 = make_det(x1=105, y1=105, x2=205, y2=305)  # center ~ (155, 205) — moved slightly
    tracks1 = tracker.update([d1], frame_id=1)
    tracks2 = tracker.update([d2], frame_id=2)
    # Should be the same track ID (continuous)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == tracks1[0].track_id
    assert tracks2[0].age == 2


def test_track_stability_flag():
    tracker = CentroidTracker(make_cfg())  # min_track_age = 2
    det = make_det()
    tracker.update([det], frame_id=1)
    tracks = tracker.update([det], frame_id=2)
    assert tracks[0].is_stable is True


def test_velocity_computed():
    tracker = CentroidTracker(make_cfg())
    d1 = make_det(x1=100, y1=100, x2=200, y2=300)
    d2 = make_det(x1=120, y1=100, x2=220, y2=300)  # moved 20px right
    tracker.update([d1], frame_id=1)
    tracks = tracker.update([d2], frame_id=2)
    vx = tracks[0].velocity[0]
    assert vx > 0, f"Expected positive vx for rightward motion, got {vx}"


def test_track_expiry():
    tracker = CentroidTracker(make_cfg())  # max_disappeared = 3
    det = make_det()
    tracker.update([det], frame_id=1)
    # Now send 4 empty frames
    for i in range(2, 6):
        tracks = tracker.update([], frame_id=i)
    # Track should be expired after 3 disappeared frames
    assert len(tracks) == 0


def test_two_distinct_tracks():
    tracker = CentroidTracker(make_cfg())
    d1 = make_det(x1=0, y1=0, x2=50, y2=100)     # left side
    d2 = make_det(x1=400, y1=0, x2=450, y2=100)   # far right
    tracks = tracker.update([d1, d2], frame_id=1)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2


def test_reset():
    tracker = CentroidTracker(make_cfg())
    det = make_det()
    tracker.update([det], frame_id=1)
    tracker.reset()
    assert tracker.active_track_count() == 0
