"""Unit tests for vision_bridge.owlvit_postprocess.

No rclpy/torch/transformers needed -- pure Python. Run directly with:
    cd ros2_ws/src/vision_bridge && PYTHONPATH=. pytest test/test_owlvit_postprocess.py
"""

from vision_bridge.owlvit_postprocess import build_detections


def test_build_detections_keeps_scores_at_or_above_threshold() -> None:
    scores = [0.05, 0.3, 0.5]
    boxes = [(0, 0, 1, 1), (10, 10, 20, 20), (30, 30, 40, 40)]

    detections = build_detections(scores, boxes, label="widget", score_threshold=0.3)

    assert [d.score for d in detections] == [0.3, 0.5]
    assert all(d.label == "widget" for d in detections)


def test_build_detections_maps_box_fields_correctly() -> None:
    detections = build_detections([0.9], [(1.0, 2.0, 3.0, 4.0)], label="x", score_threshold=0.1)

    bbox = detections[0].bbox
    assert (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) == (1.0, 2.0, 3.0, 4.0)


def test_build_detections_empty_input() -> None:
    assert build_detections([], [], label="x", score_threshold=0.1) == []


def test_build_detections_all_below_threshold() -> None:
    detections = build_detections([0.01, 0.02], [(0, 0, 1, 1), (0, 0, 1, 1)], "x", 0.5)
    assert detections == []


def test_build_detections_preserves_order() -> None:
    scores = [0.9, 0.4, 0.8]
    boxes = [(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)]

    detections = build_detections(scores, boxes, label="x", score_threshold=0.3)

    assert [d.score for d in detections] == [0.9, 0.4, 0.8]
