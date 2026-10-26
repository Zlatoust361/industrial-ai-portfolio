"""Picks and builds the `VLMInference` backend `src/api/dependencies.py` runs with.

Mirrors `ros2_ws/src/vision_bridge/vision_bridge/detector_factory.py::build_detector`'s
shape and semantics, adapted from vision_bridge's `Detector` interface to this
repo's `VLMLoader`/`VLMInference` pair (`src/models/base.py`): try TensorRT
(week 9's INT4/INT8-quantized image-encoder engine, see
`quantization/scripts/05_build_engine.py` -- proven ~2.3x speedup and a
passing quality gate, `course/TECH-DEBT.md` item C.2) when
`backend="auto"` and an engine is configured and present, fall back to
PyTorch on any load failure. A TensorRT load failure is *expected*, not a
bug -- see `src.models.clip_tensorrt.TensorRTEngineError`'s docstring:
`.engine` files don't transfer between GPU architectures/TensorRT versions,
the most common reason a configured engine won't load on a given machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import AppConfig
from src.models.base import VLMInference
from src.models.clip import CLIPInference, CLIPLoader
from src.models.clip_tensorrt import (
    CLIPTensorRTInference,
    CLIPTensorRTLoader,
    TensorRTEngineError,
)

BACKENDS = ("auto", "tensorrt", "pytorch")
"""Valid `model.backend` config values (see `src.config.ModelSection`)."""


class _LogFn(Protocol):
    def __call__(self, message: str) -> None: ...


def build_inference(config: AppConfig, log: _LogFn = print) -> VLMInference:
    """Build the `VLMInference` backend requested by `config.model.backend`.

    Args:
        config: The application config. `config.model.backend`/
            `config.model.engine_path` select the backend; `config.model`/
            `config.memory`/`config.inference.image_size` configure it.
        log: Called with human-readable backend-selection messages (e.g.
            a FastAPI startup logger); defaults to `print`.

    Returns:
        A ready-to-use `VLMInference`. For `backend="tensorrt"` and a
        successful `backend="auto"` attempt, the model is already loaded --
        backend selection needs to actually try loading the engine to know
        whether it works, mirroring `detector_factory.build_detector`'s
        eager `_build_tensorrt`. For `backend="pytorch"` and a
        fallen-back `backend="auto"`, loading stays lazy on first
        `encode_image`/`encode_text` call -- `CLIPInference`'s normal,
        unchanged contract.

    Raises:
        ValueError: If `config.model.backend` isn't one of `BACKENDS`, or if
            `backend="tensorrt"` but `config.model.engine_path` isn't set.
        FileNotFoundError: If `backend="tensorrt"` and `engine_path` is set
            but doesn't exist.
        src.models.clip_tensorrt.TensorRTEngineError: If `backend="tensorrt"`
            and the engine exists but fails to load.
    """
    backend = config.model.backend
    if backend not in BACKENDS:
        raise ValueError(f"Unknown model.backend {backend!r}; expected one of {BACKENDS}.")

    if backend == "pytorch":
        return _build_pytorch(config, log)

    if backend == "tensorrt":
        return _build_tensorrt(config, log)

    # backend == "auto"
    engine_path = config.model.engine_path
    if engine_path and Path(engine_path).exists():
        try:
            return _build_tensorrt(config, log)
        except (TensorRTEngineError, FileNotFoundError, ImportError) as exc:
            log(
                f"src.models.factory: TensorRT backend unavailable ({exc}); "
                "falling back to PyTorch."
            )
    else:
        log(
            f"src.models.factory: no engine at {engine_path!r}; using PyTorch backend. "
            "Build one on-device with quantization/scripts/05_build_engine.py."
        )
    return _build_pytorch(config, log)


def _build_pytorch(config: AppConfig, log: _LogFn) -> VLMInference:
    log(f"src.models.factory: using PyTorch CLIP backend ({config.model.name}).")
    return CLIPInference(CLIPLoader(config.model, config.memory))


def _build_tensorrt(config: AppConfig, log: _LogFn) -> VLMInference:
    if not config.model.engine_path:
        raise ValueError(
            "model.backend='tensorrt' requires model.engine_path to be set in the config "
            "(see quantization/scripts/05_build_engine.py). Use model.backend='auto' to "
            "fall back to PyTorch automatically instead of hard-requiring an engine."
        )
    log(f"src.models.factory: using TensorRT CLIP backend (engine={config.model.engine_path!r}).")
    loader = CLIPTensorRTLoader(
        config.model, config.memory, image_size=config.inference.image_size
    )
    loader.load()  # eager, so "auto" can catch a load failure and fall back.
    return CLIPTensorRTInference(loader)
