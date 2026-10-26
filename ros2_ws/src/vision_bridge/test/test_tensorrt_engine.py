"""Unit tests for vision_bridge.tensorrt_engine.

No rclpy needed. This sandbox has no tensorrt/pycuda installed, which the
second test below leans on deliberately: it proves the file-existence
check runs *before* tensorrt/pycuda are imported (TensorRTEngineError, not
a raw ImportError leaking out) -- see the class docstring. Run directly with:
    cd ros2_ws/src/vision_bridge && PYTHONPATH=. pytest test/test_tensorrt_engine.py
"""

from pathlib import Path

import pytest
from vision_bridge.tensorrt_engine import TensorRTEngine, TensorRTEngineError


def test_missing_engine_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TensorRTEngine(tmp_path / "does_not_exist.engine")


def test_existing_file_without_tensorrt_installed_raises_engine_error(tmp_path: Path) -> None:
    engine_path = tmp_path / "garbage.engine"
    engine_path.write_bytes(b"not a real engine")

    with pytest.raises(TensorRTEngineError, match="tensorrt/pycuda"):
        TensorRTEngine(engine_path)
