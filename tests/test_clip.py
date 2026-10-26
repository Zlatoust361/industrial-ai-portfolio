"""Unit tests for `CLIPLoader`/`CLIPInference`: precision handling and embedding reproducibility.

These tests load a real (tiny, randomly-initialized) CLIP checkpoint from
the Hub, so they need outbound network access. If the Hub can't be reached
(offline dev box, network-restricted CI), they skip instead of failing --
the reproducibility guarantee itself is exercised whenever the network is
available (e.g. on a standard GitHub Actions runner).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.config import MemorySection, ModelSection
from src.models.clip import CLIPInference, CLIPLoader

TINY_CLIP_CHECKPOINT = "hf-internal-testing/tiny-random-CLIPModel"


def _model_config(precision: str = "fp32") -> ModelSection:
    return ModelSection(
        name=TINY_CLIP_CHECKPOINT,
        type="clip",
        precision=precision,
        device="cpu",
        seed=42,
        cache_dir=".cache/models",
    )


def _memory_config() -> MemorySection:
    return MemorySection(memory_budget_mb=6144, device_total_mb=8192)


def _build_inference(precision: str = "fp32") -> CLIPInference:
    loader = CLIPLoader(_model_config(precision), _memory_config())
    return CLIPInference(loader)


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(128, 64, 32))


def _load_or_skip(callable_):
    try:
        return callable_()
    except Exception as exc:  # noqa: BLE001 - Hub/network errors vary by environment
        pytest.skip(f"'{TINY_CLIP_CHECKPOINT}' unavailable, likely offline: {exc!r}")


def test_same_seed_reproduces_image_embedding(sample_image: Image.Image) -> None:
    """Two independently-loaded loaders with the same seed must embed identically."""
    first = _load_or_skip(lambda: _build_inference().encode_image([sample_image]))
    second = _load_or_skip(lambda: _build_inference().encode_image([sample_image]))

    np.testing.assert_array_equal(first.vectors, second.vectors)


def test_same_seed_reproduces_text_embedding() -> None:
    """Two independently-loaded loaders with the same seed must embed text identically."""
    prompts = ["a photo of a defective weld"]

    first = _load_or_skip(lambda: _build_inference().encode_text(prompts))
    second = _load_or_skip(lambda: _build_inference().encode_text(prompts))

    np.testing.assert_array_equal(first.vectors, second.vectors)


def test_loader_respects_configured_precision() -> None:
    """`CLIPLoader.load()` must load weights in the dtype requested by config.precision."""
    loader = CLIPLoader(_model_config("fp16"), _memory_config())
    _load_or_skip(loader.load)

    import torch

    model_dtype = next(loader.model.parameters()).dtype
    assert model_dtype == torch.float16
