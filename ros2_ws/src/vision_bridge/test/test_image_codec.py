"""Unit tests for vision_bridge.image_codec.

No rclpy/colcon build needed -- pure Python + numpy. `_FakeImageMsg` stands
in for `sensor_msgs.msg.Image` since only `.encoding`/`.data`/`.height`/
`.width` are ever read (see `ImageMessageLike`). Run directly with:
    cd ros2_ws/src/vision_bridge && PYTHONPATH=. pytest test/test_image_codec.py
"""

from dataclasses import dataclass

import numpy as np
import pytest
from vision_bridge.image_codec import image_message_to_array


@dataclass
class _FakeImageMsg:
    encoding: str
    data: bytes
    height: int
    width: int


def test_image_message_to_array_decodes_rgb8() -> None:
    height, width = 2, 3
    pixels = np.arange(height * width * 3, dtype=np.uint8)
    msg = _FakeImageMsg(encoding="rgb8", data=pixels.tobytes(), height=height, width=width)

    array = image_message_to_array(msg)

    assert array.shape == (height, width, 3)
    assert array.dtype == np.uint8
    np.testing.assert_array_equal(array, pixels.reshape(height, width, 3))


def test_image_message_to_array_accepts_bgr8() -> None:
    msg = _FakeImageMsg(encoding="bgr8", data=bytes(2 * 2 * 3), height=2, width=2)
    array = image_message_to_array(msg)
    assert array.shape == (2, 2, 3)


def test_image_message_to_array_rejects_unsupported_encoding() -> None:
    msg = _FakeImageMsg(encoding="mono8", data=bytes(4), height=2, width=2)
    with pytest.raises(ValueError, match="mono8"):
        image_message_to_array(msg)
