"""Backend-agnostic detection result representation for vision_bridge.

Kept free of rclpy and rosidl-generated message types so it's unit-testable
without a colcon-built ROS2 workspace: `vision_bridge_node.py` is the only
place that translates a `Detection` into the real `vision_bridge_msgs`
message types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box, in image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class Detection:
    """One open-vocabulary detection result."""

    label: str
    score: float
    bbox: BoundingBox


class Detector(Protocol):
    """Interface a detection backend must implement to plug into the node.

    `VisionBridgeNode` calls `detect()` once per received image frame; the
    concrete backend is injected rather than hardcoded, so swapping it
    (e.g. for the OWL-ViT/TensorRT backend planned for week 03 day 3) never
    touches the ROS2 plumbing in `vision_bridge_node.py`.
    """

    def detect(self, image_rgb: np.ndarray, text_query: str) -> list[Detection]:
        """Return every detection matching `text_query` in `image_rgb`.

        Args:
            image_rgb: An `(height, width, 3)` uint8 RGB image array.
            text_query: The open-vocabulary text query to detect against.

        Returns:
            Zero or more `Detection`s found in the image.
        """
        ...


class StubDetector:
    """Placeholder `Detector` that always reports zero detections.

    Useful for exercising the ROS2 node/message wiring/launch file without
    any model, or for dev-machine testing (`detector_backend:=stub`) where
    neither a TensorRT engine nor `transformers`/`torch` may be available.
    The real backends are `owlvit_pytorch.PyTorchOwlViTDetector` and
    `owlvit_tensorrt.TensorRTOwlViTDetector`, selected automatically by
    `detector_factory.build_detector`.
    """

    def detect(self, image_rgb: np.ndarray, text_query: str) -> list[Detection]:
        """Always returns an empty list, regardless of input."""
        return []
