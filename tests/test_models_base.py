"""Contract tests for the VLMLoader / VLMInference abstract interfaces."""

from __future__ import annotations

import pytest

from src.models.base import VLMInference, VLMLoader
from tests.fakes import FakeInference, FakeLoader, sample_memory_section, sample_model_section


def test_loader_and_inference_are_abstract() -> None:
    with pytest.raises(TypeError):
        VLMLoader(sample_model_section(), sample_memory_section())  # type: ignore[abstract]
    with pytest.raises(TypeError):
        loader = FakeLoader(sample_model_section(), sample_memory_section())
        VLMInference(loader)  # type: ignore[abstract]


def test_loader_is_loaded_reflects_load_unload() -> None:
    loader = FakeLoader(sample_model_section(), sample_memory_section())
    assert not loader.is_loaded

    loader.load()
    assert loader.is_loaded

    loader.unload()
    assert not loader.is_loaded


def test_loader_model_property_raises_before_load() -> None:
    loader = FakeLoader(sample_model_section(), sample_memory_section())
    with pytest.raises(RuntimeError):
        _ = loader.model


def test_inference_lazily_loads_on_first_call() -> None:
    loader = FakeLoader(sample_model_section(), sample_memory_section())
    inference = FakeInference(loader)
    assert not loader.is_loaded

    result = inference.encode_image([])
    assert loader.is_loaded
    assert result.model_name == "fake-model"
