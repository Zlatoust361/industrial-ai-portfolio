"""Unit tests for vision_bridge.detection.

No rclpy/colcon build needed -- pure Python + numpy. Run directly with:
    cd ros2_ws/src/vision_bridge && PYTHONPATH=. pytest test/test_detection.py
"""

import numpy as np
from vision_bridge.detection import BoundingBox, Detection, StubDetector


def test_stub_detector_returns_no_detections() -> None:
    detector = StubDetector()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert detector.detect(image, "any query") == []


def test_stub_detector_ignores_query_content() -> None:
    detector = StubDetector()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert detector.detect(image, "") == []
    assert detector.detect(image, "a photo of a defect") == []


def test_bounding_box_and_detection_are_plain_dataclasses() -> None:
    bbox = BoundingBox(x_min=1.0, y_min=2.0, x_max=3.0, y_max=4.0)
    detection = Detection(label="widget", score=0.9, bbox=bbox)

    assert detection.label == "widget"
    assert detection.score == 0.9
    assert detection.bbox.x_min == 1.0
    assert detection.bbox.x_max == 3.0
