"""Picks and builds the detection backend `vision_bridge_node.py` runs with.

`build_detector(backend="auto", ...)` is the interesting case: try
TensorRT if an engine is configured and present, fall back to PyTorch on
any load failure. A TensorRT load failure here is *expected*, not a bug --
the most common cause is a `.engine` built for a different GPU/TensorRT
version (see `tensorrt_engine.TensorRTEngineError`), since engines don't
transfer between devices/TensorRT builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from vision_bridge.detection import Detector, StubDetector
from vision_bridge.tensorrt_engine import TensorRTEngineError

DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"
DEFAULT_SCORE_THRESHOLD = 0.1
BACKENDS = ("auto", "tensorrt", "pytorch", "stub")


class _LogFn(Protocol):
    def __call__(self, message: str) -> None: ...


def build_detector(
    backend: str,
    engine_path: str | None,
    model_name: str,
    score_threshold: float,
    log: _LogFn = print,
) -> Detector:
    """Build the detection backend requested by `backend`.

    Args:
        backend: One of `BACKENDS`:
            - `"auto"`: try TensorRT (if `engine_path` is set and exists),
              fall back to PyTorch on any load failure or if no engine is configured.
            - `"tensorrt"`: require a working engine; raise if unavailable.
            - `"pytorch"`: force the PyTorch backend, ignoring `engine_path`.
            - `"stub"`: always `StubDetector` (zero detections) -- dev/testing
              without any model.
        engine_path: Path to a `.engine` file. Ignored for `"pytorch"`/`"stub"`.
        model_name: HuggingFace Hub id of the OWL-ViT checkpoint.
        score_threshold: Minimum detection confidence to report.
        log: Called with human-readable backend-selection messages (e.g.
            `node.get_logger().info`); defaults to `print`.

    Returns:
        A ready-to-use `Detector`.

    Raises:
        ValueError: If `backend` isn't one of `BACKENDS`.
        FileNotFoundError: If `backend="tensorrt"` and `engine_path` doesn't exist.
        vision_bridge.tensorrt_engine.TensorRTEngineError: If
            `backend="tensorrt"` and the engine exists but fails to load.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown detector backend {backend!r}; expected one of {BACKENDS}.")

    if backend == "stub":
        log("vision_bridge: using StubDetector (no model, zero detections).")
        return StubDetector()

    if backend == "pytorch":
        return _build_pytorch(model_name, score_threshold, log)

    if backend == "tensorrt":
        return _build_tensorrt(engine_path, model_name, score_threshold, log)

    # backend == "auto"
    if engine_path and Path(engine_path).exists():
        try:
            return _build_tensorrt(engine_path, model_name, score_threshold, log)
        except (TensorRTEngineError, FileNotFoundError, ImportError) as exc:
            log(f"vision_bridge: TensorRT backend unavailable ({exc}); falling back to PyTorch.")
    else:
        log(
            f"vision_bridge: no engine at {engine_path!r}; using PyTorch backend. "
            "Build one on-device with ros2_ws/scripts/convert_owlvit_tensorrt.py."
        )
    return _build_pytorch(model_name, score_threshold, log)


def _build_pytorch(model_name: str, score_threshold: float, log: _LogFn) -> Detector:
    from vision_bridge.owlvit_pytorch import PyTorchOwlViTDetector

    log(f"vision_bridge: using PyTorch OWL-ViT backend ({model_name}).")
    return PyTorchOwlViTDetector(model_name=model_name, score_threshold=score_threshold)


def _build_tensorrt(
    engine_path: str, model_name: str, score_threshold: float, log: _LogFn
) -> Detector:
    from vision_bridge.owlvit_tensorrt import TensorRTOwlViTDetector

    log(f"vision_bridge: using TensorRT OWL-ViT backend (engine={engine_path!r}).")
    return TensorRTOwlViTDetector(
        engine_path=engine_path, model_name=model_name, score_threshold=score_threshold
    )
