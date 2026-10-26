"""Unit tests for `src.models.clip_tensorrt`.

`tensorrt`/`pycuda` aren't installed in this sandbox (same as
`quantization/tests/`'s own environment -- see that package's tests for the
established pattern this file follows): every test here either stays above
the `tensorrt`/`pycuda` import boundary entirely, or asserts the exact
`ModuleNotFoundError` those tests raise at that boundary, rather than
skipping. This mirrors `quantization/tests/test_script_06_trt_inference.py`'s
`test_main_force_reaches_pycuda_boundary`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.config import ModelSection
from src.models.clip_tensorrt import (
    CLIPTensorRTInference,
    CLIPTensorRTLoader,
    TensorRTEngineError,
    _quantization_root,
)
from tests.fakes import sample_memory_section, sample_model_section


def _model_config(engine_path: Path | str | None = None) -> ModelSection:
    return replace(sample_model_section(), backend="tensorrt", engine_path=engine_path)


def test_importing_module_never_requires_tensorrt_or_pycuda() -> None:
    """Importing `clip_tensorrt` must never require `tensorrt`/`pycuda` installed.

    Both are absent from this sandbox (see module docstring); the import at
    the top of this test file having already succeeded is itself most of
    this test, but assert it explicitly too so a future regression (e.g. a
    module-level `import tensorrt`) fails loudly here instead of only
    on-device.
    """
    assert "tensorrt" not in sys.modules
    assert "pycuda" not in sys.modules


def test_quantization_root_points_at_the_real_quantization_directory() -> None:
    quantization_root = _quantization_root()
    assert quantization_root.name == "quantization"
    assert (quantization_root / "utils" / "trt_inference.py").is_file()


def test_load_without_engine_path_raises_value_error() -> None:
    loader = CLIPTensorRTLoader(_model_config(engine_path=None), sample_memory_section(), 224)

    with pytest.raises(ValueError, match="engine_path"):
        loader.load()


def test_load_with_missing_engine_file_raises_file_not_found(tmp_path: Path) -> None:
    missing_engine = tmp_path / "does_not_exist.engine"
    loader = CLIPTensorRTLoader(
        _model_config(engine_path=missing_engine), sample_memory_section(), 224
    )

    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        loader.load()


def test_load_with_existing_engine_but_no_tensorrt_raises_module_not_found_error(
    tmp_path: Path,
) -> None:
    """The exact-import-boundary pattern this repo's own TensorRT tests use.

    Once `engine_path` exists, `CLIPTensorRTLoader.load()` reaches
    `quantization/utils/trt_inference.py::TrtImageEncoder.__init__`, which
    imports `pycuda`/`tensorrt` unguarded (see that module's docstring) --
    neither is installed here, so this must propagate as a plain
    `ModuleNotFoundError`, not get silently swallowed or turned into
    something else.
    """
    engine_path = tmp_path / "image_encoder.engine"
    engine_path.write_bytes(b"not a real engine")
    loader = CLIPTensorRTLoader(
        _model_config(engine_path=engine_path), sample_memory_section(), 224
    )

    with pytest.raises(ModuleNotFoundError, match="pycuda"):
        loader.load()


def test_image_encoder_property_raises_before_load() -> None:
    loader = CLIPTensorRTLoader(_model_config(engine_path=None), sample_memory_section(), 224)

    with pytest.raises(RuntimeError, match="not loaded"):
        _ = loader.image_encoder


def test_text_tower_is_a_clip_loader_for_the_same_config() -> None:
    config = _model_config(engine_path=None)
    loader = CLIPTensorRTLoader(config, sample_memory_section(), 224)

    assert loader.text_tower.config is config


def test_tensorrt_engine_error_is_a_runtime_error() -> None:
    assert issubclass(TensorRTEngineError, RuntimeError)


def test_inference_encode_image_lazily_loads_and_reraises_module_not_found_error(
    tmp_path: Path,
) -> None:
    """`CLIPTensorRTInference.encode_image` triggers `_ensure_loaded()` -> `load()`.

    Exercised through the public `VLMInference` surface (not just the loader
    directly), so a future change to `_ensure_loaded()`'s wiring is covered too.
    """
    from PIL import Image

    engine_path = tmp_path / "image_encoder.engine"
    engine_path.write_bytes(b"not a real engine")
    loader = CLIPTensorRTLoader(
        _model_config(engine_path=engine_path), sample_memory_section(), 224
    )
    inference = CLIPTensorRTInference(loader)

    with pytest.raises(ModuleNotFoundError, match="pycuda"):
        inference.encode_image([Image.new("RGB", (8, 8))])

    assert not loader.is_loaded
