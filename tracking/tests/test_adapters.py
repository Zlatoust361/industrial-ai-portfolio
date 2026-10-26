"""Tests for `adapters.py`.

The second test below imports the REAL
`ros2_ws/src/vision_bridge/vision_bridge/detection.py` module (same lazy
`sys.path` addition `assistant/vision_bridge.py` already established,
confirmed importable standalone since that module has zero rclpy
dependency -- only `numpy`) to prove `from_vision_bridge_detection`
against the actual class, not just an object that happens to look like it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from adapters import from_vision_bridge_detection
from sort import BBox, Detection

_ROS2_VISION_BRIDGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "ros2_ws" / "src" / "vision_bridge"
)


class _FakeBBox:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


class _FakeDetection:
    def __init__(self, label: str, score: float, bbox: _FakeBBox) -> None:
        self.label = label
        self.score = score
        self.bbox = bbox


def test_converts_duck_typed_detection() -> None:
    fake = _FakeDetection("defect", 0.87, _FakeBBox(1.0, 2.0, 3.0, 4.0))
    converted = from_vision_bridge_detection(fake)
    assert converted == Detection(label="defect", score=0.87, bbox=BBox(1.0, 2.0, 3.0, 4.0))


def test_converts_real_vision_bridge_detection() -> None:
    path_str = str(_ROS2_VISION_BRIDGE_ROOT)
    assert _ROS2_VISION_BRIDGE_ROOT.is_dir(), (
        f"ros2_ws/src/vision_bridge not found at {_ROS2_VISION_BRIDGE_ROOT} -- "
        "can't prove this adapter against the real week-03 Detection class."
    )
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    from vision_bridge.detection import BoundingBox
    from vision_bridge.detection import Detection as RealDetection

    real = RealDetection(label="ok", score=0.42, bbox=BoundingBox(10.0, 20.0, 30.0, 40.0))
    converted = from_vision_bridge_detection(real)

    assert converted == Detection(label="ok", score=0.42, bbox=BBox(10.0, 20.0, 30.0, 40.0))
