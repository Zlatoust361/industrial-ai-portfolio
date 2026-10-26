"""Unit tests for `src.models.factory.build_inference`'s backend dispatch.

Mirrors `ros2_ws/src/vision_bridge/vision_bridge/tests` (conceptually) for
`detector_factory.build_detector`: `"pytorch"` always succeeds without
touching TensorRT at all, `"tensorrt"` requires a working engine and raises
otherwise, and `"auto"` falls back to PyTorch whenever no engine is
configured/present -- `tensorrt`/`pycuda` aren't installed in this sandbox
(see `tests/test_clip_tensorrt.py`), so `"auto"` with an engine actually
present exercises the real fallback-on-`ModuleNotFoundError` path, not a
mocked one.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import load_config
from src.models.clip import CLIPInference
from src.models.clip_tensorrt import CLIPTensorRTInference
from src.models.factory import BACKENDS, build_inference


def _config(backend: str, engine_path: Path | str | None = None):
    config = load_config("config/model_config.yaml")
    return replace(config, model=replace(config.model, backend=backend, engine_path=engine_path))


def _silent_log(_message: str) -> None:
    pass


def test_backends_tuple_matches_documented_values() -> None:
    assert BACKENDS == ("auto", "tensorrt", "pytorch")


def test_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown model.backend"):
        build_inference(_config("not-a-real-backend"), log=_silent_log)


def test_pytorch_backend_returns_clip_inference_unloaded() -> None:
    inference = build_inference(_config("pytorch"), log=_silent_log)

    assert isinstance(inference, CLIPInference)
    assert not inference.loader.is_loaded  # lazy: unchanged from the pre-factory behavior


def test_pytorch_backend_ignores_engine_path(tmp_path: Path) -> None:
    """`backend="pytorch"` must ignore `engine_path` entirely, even if set."""
    bogus_engine = tmp_path / "some.engine"
    inference = build_inference(_config("pytorch", engine_path=bogus_engine), log=_silent_log)

    assert isinstance(inference, CLIPInference)


def test_tensorrt_backend_without_engine_path_raises_value_error() -> None:
    with pytest.raises(ValueError, match="engine_path"):
        build_inference(_config("tensorrt", engine_path=None), log=_silent_log)


def test_tensorrt_backend_with_missing_engine_file_raises_file_not_found(tmp_path: Path) -> None:
    missing_engine = tmp_path / "does_not_exist.engine"

    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        build_inference(_config("tensorrt", engine_path=missing_engine), log=_silent_log)


def test_tensorrt_backend_with_present_but_unloadable_engine_raises_module_not_found_error(
    tmp_path: Path,
) -> None:
    """`backend="tensorrt"` is a hard requirement: no fallback, unlike `"auto"`."""
    engine_path = tmp_path / "image_encoder.engine"
    engine_path.write_bytes(b"not a real engine")

    with pytest.raises(ModuleNotFoundError, match="pycuda"):
        build_inference(_config("tensorrt", engine_path=engine_path), log=_silent_log)


def test_auto_backend_falls_back_to_pytorch_when_engine_path_unset() -> None:
    messages = []
    inference = build_inference(_config("auto", engine_path=None), log=messages.append)

    assert isinstance(inference, CLIPInference)
    assert any("no engine" in message for message in messages)


def test_auto_backend_falls_back_to_pytorch_when_engine_file_missing(tmp_path: Path) -> None:
    missing_engine = tmp_path / "does_not_exist.engine"
    messages = []

    inference = build_inference(_config("auto", engine_path=missing_engine), log=messages.append)

    assert isinstance(inference, CLIPInference)
    assert any("no engine" in message for message in messages)


def test_auto_backend_falls_back_to_pytorch_when_engine_present_but_unloadable(
    tmp_path: Path,
) -> None:
    """`backend="auto"` with a present-but-broken engine logs why, then falls back.

    `tensorrt`/`pycuda` aren't installed in this sandbox (see module
    docstring), so this exercises the real `ModuleNotFoundError` ->
    `ImportError` fallback path, not a mocked failure.
    """
    engine_path = tmp_path / "image_encoder.engine"
    engine_path.write_bytes(b"not a real engine")
    messages = []

    inference = build_inference(_config("auto", engine_path=engine_path), log=messages.append)

    assert isinstance(inference, CLIPInference)
    assert any("TensorRT backend unavailable" in message for message in messages)


def test_auto_backend_does_not_fall_back_when_no_engine_would_be_needed_to_try() -> None:
    """Sanity check: `"auto"` never even attempts TensorRT without a configured path."""
    inference = build_inference(_config("auto", engine_path=""), log=_silent_log)

    assert isinstance(inference, CLIPInference)


def test_clip_tensorrt_inference_is_exported_for_the_tensorrt_path() -> None:
    # Structural check that the factory module actually wires up
    # CLIPTensorRTInference for a *working* engine load, even though no test
    # in this sandbox can construct one (tensorrt/pycuda aren't installed).
    import src.models.factory as factory_module

    assert factory_module.CLIPTensorRTInference is CLIPTensorRTInference
