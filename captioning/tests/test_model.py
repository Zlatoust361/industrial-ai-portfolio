"""Tests for `captioning.model.CaptioningLoader`.

Mirrors `peft/tests/test_script_02_train_adapter.py`'s pattern for
`CLIPLoader`: the memory-budget check (already fully unit-tested in
`tests/test_memory.py`) is proven to be WIRED UP correctly here -- a too-
low budget must raise before any network call -- and the real load path
is proven to reach exactly the real network boundary, via
`assert_reaches_network_boundary`, not a random unrelated error.
"""

from __future__ import annotations

from pathlib import Path

from conftest import assert_reaches_network_boundary

from captioning.model import CaptioningLoader
from src.config import MemorySection, ModelSection
from src.models.memory import MemoryBudgetExceededError


def _model_section(**overrides: object) -> ModelSection:
    defaults: dict[str, object] = dict(
        name="HuggingFaceTB/SmolVLM-256M-Instruct",
        type="vlm-captioning",
        precision="fp32",
        device="cpu",
        seed=42,
        cache_dir=Path(".cache/models"),
    )
    defaults.update(overrides)
    return ModelSection(**defaults)  # type: ignore[arg-type]


def test_load_raises_before_network_call_when_budget_exceeds_available_ram(
    monkeypatch,
) -> None:
    import captioning.model as model_module

    monkeypatch.setattr(
        model_module, "available_system_memory_mb", lambda: 10.0  # far too little
    )
    loader = CaptioningLoader(
        _model_section(), MemorySection(memory_budget_mb=2048, device_total_mb=8192)
    )

    try:
        loader.load()
        raise AssertionError("Expected MemoryBudgetExceededError, load() did not raise.")
    except MemoryBudgetExceededError:
        pass


def test_load_reaches_real_network_boundary_when_budget_fits() -> None:
    loader = CaptioningLoader(
        _model_section(), MemorySection(memory_budget_mb=2048, device_total_mb=8192)
    )
    assert_reaches_network_boundary(loader.load)


def test_unload_clears_loaded_state() -> None:
    loader = CaptioningLoader(
        _model_section(), MemorySection(memory_budget_mb=2048, device_total_mb=8192)
    )
    loader._model = object()
    loader._processor = object()
    assert loader.is_loaded

    loader.unload()
    assert not loader.is_loaded
