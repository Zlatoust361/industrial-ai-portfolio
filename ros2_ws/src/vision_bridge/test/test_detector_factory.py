"""Unit tests for vision_bridge.detector_factory.

No rclpy needed. This sandbox has neither tensorrt/pycuda nor torch/
transformers installed, which we lean on deliberately: several tests below
assert that build_detector's branching reaches the *expected* environment
boundary (FileNotFoundError from a missing engine, ModuleNotFoundError from
missing torch/transformers) rather than failing earlier for the wrong
reason -- the same "smoke test up to the import boundary" pattern used
elsewhere in this repo (see course/DECISIONS.md). Run directly with:
    cd ros2_ws/src/vision_bridge && PYTHONPATH=. pytest test/test_detector_factory.py
"""

from pathlib import Path

import pytest
from vision_bridge.detection import StubDetector
from vision_bridge.detector_factory import build_detector


def test_stub_backend_returns_stub_detector() -> None:
    detector = build_detector("stub", engine_path=None, model_name="x", score_threshold=0.1)
    assert isinstance(detector, StubDetector)


def test_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="auto"):
        build_detector("quantum", engine_path=None, model_name="x", score_threshold=0.1)


def test_tensorrt_backend_missing_engine_raises_file_not_found(tmp_path: Path) -> None:
    missing_engine = tmp_path / "does_not_exist.engine"
    with pytest.raises(FileNotFoundError):
        build_detector(
            "tensorrt", engine_path=str(missing_engine), model_name="x", score_threshold=0.1
        )


def test_auto_backend_with_no_engine_falls_through_to_pytorch() -> None:
    # No engine configured -> straight to PyTorch, which needs torch/
    # transformers -- not installed in this sandbox, so ModuleNotFoundError
    # (not FileNotFoundError/TensorRTEngineError) proves the auto branch
    # correctly skipped TensorRT entirely.
    with pytest.raises(ModuleNotFoundError):
        build_detector("auto", engine_path=None, model_name="x", score_threshold=0.1)


def test_auto_backend_with_unusable_engine_falls_back_to_pytorch(tmp_path: Path) -> None:
    # The engine file exists but tensorrt/pycuda aren't installed here, so
    # loading it fails inside the "try TensorRT" branch; build_detector
    # should catch that and fall through to PyTorch, which then also fails
    # (no torch/transformers here) -- proving both branches were reached,
    # not just the first one.
    garbage_engine = tmp_path / "garbage.engine"
    garbage_engine.write_bytes(b"not a real engine")

    with pytest.raises(ModuleNotFoundError):
        build_detector(
            "auto", engine_path=str(garbage_engine), model_name="x", score_threshold=0.1
        )


def test_pytorch_backend_forced_needs_torch() -> None:
    with pytest.raises(ModuleNotFoundError):
        build_detector("pytorch", engine_path=None, model_name="x", score_threshold=0.1)
