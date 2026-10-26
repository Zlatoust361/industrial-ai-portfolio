"""Backend-agnostic VLMLoader / VLMInference contracts.

Concrete subclasses (see `src.models.clip`) own the framework-specific
details -- HuggingFace transformers, ONNX Runtime, TensorRT, ... -- behind
this single pair of interfaces, so callers in `src/api`, `src/evaluation`
and `scripts/` never depend on a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.config import MemorySection, ModelSection


@dataclass(frozen=True)
class EmbeddingResult:
    """A batch of embeddings produced by a `VLMInference` call.

    Attributes:
        vectors: Array of shape `(batch, embedding_dim)`, L2-normalized.
        model_name: Identifier of the model that produced the embeddings.
    """

    vectors: np.ndarray
    model_name: str


class VLMLoader(ABC):
    """Loads a vision-language model and its preprocessing pipeline.

    Subclasses implement `load`/`unload` for one backend and expose the
    loaded objects through the `model`/`processor` properties.
    """

    def __init__(self, config: ModelSection, memory: MemorySection) -> None:
        """Store the model configuration; does not load weights yet.

        Args:
            config: Model section of the application config (name, precision,
                device, seed, cache_dir, ...).
            memory: Memory-budget section of the application config. Subclasses
                are expected to check it (see `src.models.memory`) before and
                after loading weights, so an oversized model fails with a clear
                error instead of an OOM crash.
        """
        self.config = config
        self.memory = memory
        self._model: object | None = None
        self._processor: object | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether `load()` has been called and succeeded."""
        return self._model is not None

    @property
    def model(self) -> object:
        """The loaded model object.

        Raises:
            RuntimeError: If `load()` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first.")
        return self._model

    @property
    def processor(self) -> object:
        """The loaded preprocessor/tokenizer object.

        Raises:
            RuntimeError: If `load()` has not been called yet.
        """
        if self._processor is None:
            raise RuntimeError("Processor not loaded; call load() first.")
        return self._processor

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory.

        Implementations must apply a deterministic seed (from
        `self.config.seed`) before any weight initialization/sampling, and
        honor `self.config.precision` (`"fp16"` or `"fp32"`) and
        `self.config.device`.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (GPU memory, file handles)."""


class VLMInference(ABC):
    """Unified image/text embedding interface over a `VLMLoader`."""

    def __init__(self, loader: VLMLoader) -> None:
        """Bind this inference wrapper to a (possibly not-yet-loaded) loader.

        Args:
            loader: A `VLMLoader` instance; `load()` is called lazily on first use.
        """
        self.loader = loader

    def _ensure_loaded(self) -> None:
        """Load the underlying model on first use."""
        if not self.loader.is_loaded:
            self.loader.load()

    @abstractmethod
    def encode_image(self, images: list[Image.Image]) -> EmbeddingResult:
        """Encode a batch of images into L2-normalized embedding vectors.

        Args:
            images: Batch of RGB PIL images.

        Returns:
            An `EmbeddingResult` with one vector per input image.
        """

    @abstractmethod
    def encode_text(self, texts: list[str]) -> EmbeddingResult:
        """Encode a batch of text prompts into L2-normalized embedding vectors.

        Args:
            texts: Batch of text prompts.

        Returns:
            An `EmbeddingResult` with one vector per input prompt.
        """
