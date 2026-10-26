"""Fake VLMLoader/VLMInference used by tests to avoid downloading real weights."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.config import MemorySection, ModelSection
from src.models.base import EmbeddingResult, VLMInference, VLMLoader


def sample_model_section() -> ModelSection:
    """A minimal `ModelSection` for tests that don't care about specific values."""
    return ModelSection(
        name="fake/model",
        type="clip",
        precision="fp32",
        device="cpu",
        seed=0,
        cache_dir=".cache/models",
    )


def sample_memory_section(
    memory_budget_mb: int = 6144, device_total_mb: int = 8192
) -> MemorySection:
    """A minimal `MemorySection` for tests, defaulting to the Jetson Orin Nano budget."""
    return MemorySection(memory_budget_mb=memory_budget_mb, device_total_mb=device_total_mb)


class FakeLoader(VLMLoader):
    """A `VLMLoader` that "loads" an in-memory sentinel instead of real weights."""

    def load(self) -> None:
        self._model = object()
        self._processor = object()

    def unload(self) -> None:
        self._model = None
        self._processor = None


class FakeInference(VLMInference):
    """A `VLMInference` that returns deterministic zero vectors, for structure tests."""

    embedding_dim: int = 8

    def encode_image(self, images: list[Image.Image]) -> EmbeddingResult:
        self._ensure_loaded()
        return EmbeddingResult(
            vectors=np.zeros((len(images), self.embedding_dim), dtype=np.float32),
            model_name="fake-model",
        )

    def encode_text(self, texts: list[str]) -> EmbeddingResult:
        self._ensure_loaded()
        return EmbeddingResult(
            vectors=np.zeros((len(texts), self.embedding_dim), dtype=np.float32),
            model_name="fake-model",
        )
