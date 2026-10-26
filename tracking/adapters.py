"""Convert a real week-03 `Detection` into `tracking.sort`'s own `Detection` -- one line.

Pure attribute copy, duck-typed -- works on
`ros2_ws/src/vision_bridge/vision_bridge/detection.py`'s real `Detection`/
`BoundingBox` classes without importing them (no `sys.path` tricks needed
here, unlike `assistant/vision_bridge.py`'s `detect_objects`, which needs
the real `detector_factory.build_detector` function, not just the data
shape). `tests/test_adapters.py` still imports the REAL
`vision_bridge.detection` module (via the same lazy `sys.path` addition
`assistant/vision_bridge.py` already established) to prove this against
the actual class, not just an object that happens to look like it.
"""

from __future__ import annotations

from sort import BBox, Detection


def from_vision_bridge_detection(detection: object) -> Detection:
    """Convert a `vision_bridge.detection.Detection`-shaped object into `sort.Detection`.

    Args:
        detection: Any object with `.label` (str), `.score` (float), and
            `.bbox` (an object with `.x_min`/`.y_min`/`.x_max`/`.y_max`)
            -- week 03's real `Detection`/`BoundingBox` satisfy this
            exactly, as does any other detector's output shaped the same way.

    Returns:
        The equivalent `tracking.sort.Detection`.
    """
    bbox = detection.bbox
    return Detection(
        label=detection.label,
        score=detection.score,
        bbox=BBox(x_min=bbox.x_min, y_min=bbox.y_min, x_max=bbox.x_max, y_max=bbox.y_max),
    )
