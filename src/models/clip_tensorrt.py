"""CLIP `VLMLoader`/`VLMInference` backed by week 9's TensorRT image-encoder engine.

Only the vision tower runs through TensorRT. Week 9's own ONNX export
(`quantization/scripts/02_export_onnx.py`) only ever traces
`CLIPModel.get_image_features` -- there is no text-tower ONNX/TensorRT
artifact anywhere in this repo to load -- so `encode_text` here delegates to
a plain PyTorch/HuggingFace `src.models.clip.CLIPInference` underneath. This
mirrors `ros2_ws/src/vision_bridge/vision_bridge/owlvit_tensorrt.py`'s split
for OWL-ViT exactly: TensorRT vision tower, PyTorch text tower (and, for
OWL-ViT, detection heads too; CLIP has no heads to speak of).

**Reusing week 9's code across the `src/`<->`quantization/` boundary.**
`quantization/utils/trt_inference.py`'s `TrtImageEncoder` is the tested,
already-working engine-loading/inference code (buffer allocation,
`execute_async_v3`, preprocessing) that `06_trt_inference.py`/
`07_evaluate_quality.py` already exercise -- reused here unmodified rather
than reimplemented. Importing it needs `quantization/` itself on
`sys.path` first: `trt_inference.py` does `from utils.tensorrt_builder
import preprocess_calibration_batch` as a bare, unqualified import (it is
only ever run as `cd quantization && PYTHONPATH=. pytest tests/`, per
`quantization/README.md`'s own documented invocation -- never as
`quantization.utils.trt_inference` with just the repo root on `sys.path`,
which is how this repo's root `pytest -q` runs `tests/`). So `from
quantization.utils.trt_inference import TrtImageEncoder` would fail at
that inner `from utils.tensorrt_builder import ...` line even with the repo
root importable as a namespace package.
`quantization/scripts/01_baseline_inference.py` already does the
mirror-image hack in the other direction (`sys.path.insert(0,
str(_QUANTIZATION_ROOT.parent))` to reach `src/` from `quantization/`);
`_import_trt_image_encoder` below does the reverse -- appends
`quantization/` to `sys.path`, lazily, only when the TensorRT backend is
actually loaded -- rather than extracting a third shared module, since
`trt_inference.py`'s own unqualified `utils.*` imports would need rewriting
either way and this repo's scripts already treat this sys.path shape as
normal, not something to route around. The one fragility worth flagging:
this would shadow a *real* top-level `utils` package if the repo ever grew
one at the root -- none exists today (only `quantization/utils/`), and using
`sys.path.append` (not `insert(0, ...)`) means any such future package
would still resolve first.

`TrtImageEncoder.encode()` only accepts image file *paths* -- it feeds them
through `preprocess_calibration_batch`, written for on-disk
calibration/test-set images -- but `VLMInference.encode_image` receives
in-memory `PIL.Image`s from the API layer. Each image is therefore written
to a temporary file before being handed to the (otherwise untouched)
`TrtImageEncoder.encode()` call, rather than duplicating
`preprocess_calibration_batch`'s normalization logic for an in-memory
input path. The engine's optimization profile also pins a single, fixed
batch size (see `quantization/utils/tensorrt_builder.py::
configure_builder_config`); both builder profiles shipped in
`quantization/config/config.yaml` (`dev`, `orin_nano_8gb`) build with
`batch_size: 1`, so `_ENGINE_BATCH_SIZE` below is fixed at `1` and this
module processes one image per engine call, stacking the results, rather
than guessing a batch size that might not match how the loaded `.engine`
was actually built.

Lazy-import discipline mirrors `src.models.clip` and the `ros2_ws`
TensorRT modules: importing this module never requires `tensorrt`/
`pycuda`, and never touches `sys.path` -- only `CLIPTensorRTLoader.load()`
does, exactly like `ros2_ws/.../owlvit_tensorrt.py`'s `TensorRTEngine`
construction happens "before the imports below" in its own docstring.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import MemorySection, ModelSection
from src.models.base import EmbeddingResult, VLMInference, VLMLoader
from src.models.clip import CLIPInference, CLIPLoader

_ENGINE_BATCH_SIZE = 1
"""Fixed engine input batch size -- matches every builder profile this repo ships
(`quantization/config/config.yaml`'s `dev`/`orin_nano_8gb`, both `batch_size: 1`).
If a future profile ever builds with a different batch size, this constant --
and the per-image loop in `CLIPTensorRTInference.encode_image` -- needs revisiting."""


class TensorRTEngineError(RuntimeError):
    """Raised when the configured `.engine` file can't be loaded or run.

    Mirrors `src.models.memory.MemoryBudgetExceededError`'s "clear, actionable
    message" style, and `ros2_ws/src/vision_bridge/vision_bridge/tensorrt_engine.py`'s
    `TensorRTEngineError` semantics: the most common cause is a `.engine` built
    for a different GPU architecture or TensorRT version than the one currently
    installed -- engines are not portable across builds (see
    `quantization/scripts/05_build_engine.py`'s module docstring). Callers (see
    `src.models.factory.build_inference`'s `backend="auto"` path) should treat
    this as an expected fallback trigger, not a hard failure.
    """


def _quantization_root() -> Path:
    """Absolute path to this repo's `quantization/` directory."""
    return Path(__file__).resolve().parent.parent.parent / "quantization"


def _import_trt_image_encoder() -> type:
    """Import `quantization/utils/trt_inference.py`'s `TrtImageEncoder`, lazily.

    Appends `quantization/` to `sys.path` first -- see this module's docstring
    for why that's needed and how it's scoped to avoid shadowing a real
    top-level `utils` package, should one ever exist.

    Returns:
        The `TrtImageEncoder` class (not instantiated -- constructing it is
        what actually needs `tensorrt`/`pycuda` installed).
    """
    quantization_root = str(_quantization_root())
    if quantization_root not in sys.path:
        sys.path.append(quantization_root)

    from utils.trt_inference import TrtImageEncoder  # type: ignore[import-not-found]

    return TrtImageEncoder


class CLIPTensorRTLoader(VLMLoader):
    """Loads week 9's TensorRT image-encoder engine, plus a plain PyTorch CLIP text tower.

    `model`/`processor` (the base `VLMLoader` contract) expose the TensorRT
    image encoder and the HuggingFace `CLIPProcessor` respectively, once
    loaded; the PyTorch text tower is reached through `text_tower` instead
    (see module docstring for why CLIP's text side stays PyTorch).
    """

    def __init__(self, config: ModelSection, memory: MemorySection, image_size: int) -> None:
        """Args:
            config: Model section of the application config. `config.engine_path`
                must be set (and exist) for `load()` to succeed; `config.name`
                selects the HuggingFace checkpoint the *text* tower and
                processor are loaded from (must match the checkpoint the
                engine was built from -- not verified here, same caveat
                `ros2_ws/.../owlvit_tensorrt.py`'s `model_name` parameter documents).
            memory: Memory-budget section of the application config, applied
                to the PyTorch text tower via the same `src.models.memory`
                guardrails `CLIPLoader` uses.
            image_size: Square edge (pixels) the engine's input expects (from
                `config.inference.image_size` -- `ModelSection` itself has no
                image-size field, see `src.config.InferenceSection`).
        """
        super().__init__(config, memory)
        self._image_size = image_size
        self._text_loader = CLIPLoader(config, memory)
        self._image_encoder: object | None = None

    @property
    def image_encoder(self) -> object:
        """The loaded `TrtImageEncoder`.

        Raises:
            RuntimeError: If `load()` has not been called yet.
        """
        if self._image_encoder is None:
            raise RuntimeError("TensorRT image encoder not loaded; call load() first.")
        return self._image_encoder

    @property
    def text_tower(self) -> CLIPLoader:
        """The `CLIPLoader` backing text encoding (see module docstring)."""
        return self._text_loader

    def load(self) -> None:
        """Load the TensorRT image encoder and the PyTorch CLIP text tower.

        Raises:
            ValueError: If `self.config.engine_path` is not set.
            FileNotFoundError: If `self.config.engine_path` is set but doesn't exist.
            ModuleNotFoundError: If `tensorrt`/`pycuda` aren't importable at all
                -- propagated as-is (not wrapped), matching
                `quantization/utils/trt_inference.py::TrtImageEncoder`'s own
                behavior and this repo's own tests for it (assert the exact
                `ModuleNotFoundError` rather than skip). `src.models.factory.
                build_inference`'s `backend="auto"` path catches this
                (`ModuleNotFoundError` is an `ImportError`) as an expected
                fallback trigger, same as `TensorRTEngineError`.
            TensorRTEngineError: If the engine file exists but fails to
                deserialize -- most commonly a GPU-architecture/TensorRT-
                version mismatch.
            src.models.memory.MemoryBudgetExceededError: If the PyTorch text
                tower doesn't fit the configured memory budget (see
                `CLIPLoader.load`).
        """
        if not self.config.engine_path:
            raise ValueError(
                "model.backend='tensorrt' requires model.engine_path to be set in the "
                "config (see quantization/scripts/05_build_engine.py, which writes the "
                "INT4/INT8-quantized image-encoder .engine on-device). Set model.engine_path "
                "in config/model_config.yaml (or the target's config file), or use "
                "model.backend='auto' to fall back to PyTorch automatically instead of "
                "hard-requiring an engine."
            )
        engine_path = Path(self.config.engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}. Build one with "
                "quantization/scripts/05_build_engine.py, run on the target device -- "
                "engines are not portable across GPU architectures/TensorRT versions, "
                "so this must be built on the exact device that will run it."
            )

        trt_image_encoder_cls = _import_trt_image_encoder()
        try:
            image_encoder = trt_image_encoder_cls(
                engine_path, batch_size=_ENGINE_BATCH_SIZE, image_size=self._image_size
            )
        except RuntimeError as exc:
            # ModuleNotFoundError (tensorrt/pycuda missing) is an ImportError, not a
            # RuntimeError, and intentionally propagates unwrapped -- see this
            # method's docstring.
            raise TensorRTEngineError(
                f"Failed to load TensorRT engine '{engine_path}': {exc}. Most likely cause: "
                "this .engine was built for a different GPU architecture or TensorRT "
                "version than the one installed here -- rebuild it on this exact device "
                "with quantization/scripts/05_build_engine.py."
            ) from exc

        # Text tower + processor stay PyTorch/HuggingFace (see module docstring).
        self._text_loader.load()

        self._image_encoder = image_encoder
        self._model = image_encoder
        self._processor = self._text_loader.processor

    def unload(self) -> None:
        """Release the TensorRT image encoder and the PyTorch text tower."""
        self._image_encoder = None
        self._model = None
        self._processor = None
        self._text_loader.unload()


class CLIPTensorRTInference(VLMInference):
    """Unified image/text embedding interface: TensorRT image tower, PyTorch text tower."""

    def __init__(self, loader: CLIPTensorRTLoader) -> None:
        """Args:
            loader: A `CLIPTensorRTLoader` instance; `load()` is called lazily on first use.
        """
        super().__init__(loader)
        # Reuses CLIPInference's own tested encode_text unmodified, bound to the
        # loader's PyTorch text tower -- see module docstring.
        self._text_inference = CLIPInference(loader.text_tower)

    def encode_image(self, images: list[Image.Image]) -> EmbeddingResult:
        """Encode a batch of PIL images via the TensorRT image encoder.

        Each image is written to a temporary file and run through the engine
        one at a time (`_ENGINE_BATCH_SIZE`), since `TrtImageEncoder.encode()`
        both requires on-disk paths and only accepts its fixed built batch
        size -- see module docstring.
        """
        self._ensure_loaded()
        loader: CLIPTensorRTLoader = self.loader  # type: ignore[assignment]
        image_encoder = loader.image_encoder

        if not images:
            return EmbeddingResult(
                vectors=np.empty((0, 0), dtype=np.float32), model_name=self.loader.config.name
            )

        embeddings = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for index, image in enumerate(images):
                image_path = Path(tmp_dir) / f"image_{index:04d}.png"
                image.convert("RGB").save(image_path)
                embeddings.append(image_encoder.encode([image_path]))

        vectors = np.concatenate(embeddings, axis=0).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return EmbeddingResult(vectors=vectors, model_name=self.loader.config.name)

    def encode_text(self, texts: list[str]) -> EmbeddingResult:
        """Encode a batch of text prompts via the PyTorch CLIP text tower.

        No text-tower TensorRT engine exists (see module docstring), so this
        delegates entirely to a `CLIPInference` bound to the loader's PyTorch
        text tower -- the same, already-tested code `CLIPInference.encode_text`
        runs for the plain `backend="pytorch"` path.
        """
        self._ensure_loaded()
        return self._text_inference.encode_text(texts)
