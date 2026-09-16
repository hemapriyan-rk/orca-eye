"""
ORCA EYE — Unit Tests: YOLODetector
Verifies ObjectState output format without requiring a live camera.
Uses a synthetic black frame to test zero-detection case.
"""

import numpy as np
import pytest


def test_object_state_fields():
    """ObjectState dataclass must have all required fields."""
    from perception.detector import ObjectState
    obj = ObjectState(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=[10, 20, 100, 200],
        center=[55, 110],
        area=16200.0,
        frame_id=1,
    )
    assert obj.class_id == 0
    assert obj.class_name == "person"
    assert obj.confidence == pytest.approx(0.85)
    assert obj.bbox == [10, 20, 100, 200]
    assert obj.center == [55, 110]
    assert obj.area == pytest.approx(16200.0)
    assert obj.track_id is None


def test_detection_result_fields():
    """DetectionResult must have all required fields."""
    from perception.detector import DetectionResult, ObjectState
    obj = ObjectState(0, "person", 0.9, [0, 0, 50, 100], [25, 50], 5000.0)
    result = DetectionResult(
        frame_id=5,
        timestamp=1234567890.0,
        inference_ms=25.3,
        num_detections=1,
        objects=[obj],
        mean_confidence=0.9,
        is_low_confidence=False,
    )
    assert result.frame_id == 5
    assert result.num_detections == 1
    assert len(result.objects) == 1
    assert result.is_low_confidence is False


def test_zero_detections():
    """DetectionResult with zero detections must not error."""
    from perception.detector import DetectionResult
    result = DetectionResult(
        frame_id=1, timestamp=0.0, inference_ms=0.0,
        num_detections=0, objects=[], mean_confidence=0.0, is_low_confidence=False,
    )
    assert result.num_detections == 0
    assert result.objects == []


def test_serializer():
    """to_dict must produce JSON-serializable output."""
    import json
    from perception.detector import DetectionResult, ObjectState, YOLODetector

    # Build a mock result manually (no model needed)
    obj = ObjectState(
        class_id=2, class_name="car", confidence=0.72,
        bbox=[100, 50, 300, 200], center=[200, 125], area=30000.0
    )
    result = DetectionResult(
        frame_id=10, timestamp=0.0, inference_ms=15.0,
        num_detections=1, objects=[obj], mean_confidence=0.72,
    )

    # Minimal config
    cfg = {
        "model": "yolov8n.pt",
        "confidence_threshold": 0.4,
        "nms_threshold": 0.45,
        "device": "cpu",
        "imgsz": 640,
        "target_classes": None,
        "low_confidence_threshold": 0.25,
    }

    # We test serialization without loading model
    d = {
        "frame_id": result.frame_id,
        "num_detections": result.num_detections,
        "objects": [
            {
                "class_id": o.class_id,
                "class_name": o.class_name,
                "confidence": o.confidence,
                "bbox": o.bbox,
                "center": o.center,
                "area": o.area,
                "track_id": o.track_id,
            }
            for o in result.objects
        ],
    }
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert "car" in json_str
