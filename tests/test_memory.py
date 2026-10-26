"""Unit tests for `src.models.memory` budget guardrails."""

from __future__ import annotations

import pytest

from src.config import MemorySection
from src.models.memory import MemoryBudgetExceededError, check_memory_budget, check_process_memory


def _memory(memory_budget_mb: int = 6144, device_total_mb: int = 8192) -> MemorySection:
    return MemorySection(memory_budget_mb=memory_budget_mb, device_total_mb=device_total_mb)


def test_check_memory_budget_passes_when_enough_ram_available() -> None:
    check_memory_budget(_memory(memory_budget_mb=6144), available_mb=7000.0)


def test_check_memory_budget_raises_when_available_ram_below_budget() -> None:
    with pytest.raises(MemoryBudgetExceededError, match="memory_budget_mb=6144"):
        check_memory_budget(_memory(memory_budget_mb=6144), available_mb=4000.0)


def test_check_process_memory_passes_at_or_below_budget() -> None:
    memory = _memory(memory_budget_mb=6144)
    check_process_memory(memory, process_rss_mb=6144.0, stage="after model load")
    check_process_memory(memory, process_rss_mb=1000.0, stage="after model load")


def test_check_process_memory_raises_when_rss_exceeds_budget() -> None:
    with pytest.raises(MemoryBudgetExceededError, match="after model load"):
        check_process_memory(
            _memory(memory_budget_mb=6144), process_rss_mb=7000.0, stage="after model load"
        )
