"""Decode a ROS `sensor_msgs/Image`-shaped object into a numpy array.

Duck-typed on purpose: only reads `.encoding`, `.data`, `.height`, `.width`,
not the real `sensor_msgs.msg.Image` class -- so this is unit-testable
without rclpy/a colcon-built workspace, using any object that has those
four attributes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SUPPORTED_IMAGE_ENCODINGS = ("rgb8", "bgr8")


class ImageMessageLike(Protocol):
    """The subset of `sensor_msgs/Image` this module actually reads."""

    encoding: str
    data: bytes
    height: int
    width: int


def image_message_to_array(msg: ImageMessageLike) -> np.ndarray:
    """Decode an image-message-like object into an `(height, width, 3)` uint8 array.

    No `cv_bridge` dependency: only handles the raw `rgb8`/`bgr8` layouts a
    camera driver or `image_transport` republish node typically produces.

    Args:
        msg: An object with `.encoding`, `.data`, `.height`, `.width`
            (e.g. a real `sensor_msgs.msg.Image`).

    Returns:
        An `(msg.height, msg.width, 3)` uint8 array.

    Raises:
        ValueError: If `msg.encoding` isn't `rgb8`/`bgr8` -- fails loudly
            instead of silently misinterpreting pixel data with the wrong
            channel layout.
    """
    if msg.encoding not in SUPPORTED_IMAGE_ENCODINGS:
        raise ValueError(
            f"Unsupported image encoding {msg.encoding!r}; expected one of "
            f"{SUPPORTED_IMAGE_ENCODINGS}."
        )
    array = np.frombuffer(msg.data, dtype=np.uint8)
    return array.reshape(msg.height, msg.width, 3)
