"""Memory-budget guardrails for running VLMs on memory-constrained edge devices.

An 8 GB Jetson Orin Nano has no forgiveness for an oversized model: the
kernel OOM-killer terminates the process with no diagnostic beyond
`Killed`. `check_memory_budget` / `check_process_memory` turn that into a
clear, actionable `MemoryBudgetExceededError` raised before the OOM-killer
would otherwise step in.
"""

from __future__ import annotations

from src.config import MemorySection


class MemoryBudgetExceededError(RuntimeError):
    """Raised when the `memory_budget_mb` configured in `model_config.yaml` is exceeded."""


def available_system_memory_mb() -> float:
    """Currently available system memory, in MiB.

    Returns:
        `psutil.virtual_memory().available` converted to MiB.
    """
    import psutil

    return psutil.virtual_memory().available / (1024**2)


def process_resident_memory_mb() -> float:
    """This process's resident set size (RSS), in MiB.

    Returns:
        `psutil.Process().memory_info().rss` converted to MiB.
    """
    import psutil

    return psutil.Process().memory_info().rss / (1024**2)


def check_memory_budget(memory: MemorySection, available_mb: float) -> None:
    """Fail fast if the configured budget cannot fit in currently available RAM.

    Call this before loading any weights: if `available_mb` already can't
    cover `memory.memory_budget_mb`, proceeding would run the device out of
    memory and get the process OOM-killed instead of failing cleanly.

    Args:
        memory: `memory:` section of the config.
        available_mb: Currently available system memory, in MiB (see
            `available_system_memory_mb`).

    Raises:
        MemoryBudgetExceededError: If `available_mb` is below the configured budget.
    """
    if available_mb < memory.memory_budget_mb:
        raise MemoryBudgetExceededError(
            f"Configured memory_budget_mb={memory.memory_budget_mb} MB exceeds the "
            f"{available_mb:.0f} MB currently available on this device "
            f"(device_total_mb={memory.device_total_mb} MB). Lower memory_budget_mb in "
            "config/model_config.yaml, free up memory, or switch to a smaller/"
            "lower-precision model before loading."
        )


def check_process_memory(memory: MemorySection, process_rss_mb: float, stage: str) -> None:
    """Fail fast if this process has already grown past the configured budget.

    Call this right after `VLMLoader.load()`: if the model alone already
    exceeds the budget, later inference calls -- which add activation
    memory on top -- are close to guaranteed to hit the device's
    OOM-killer.

    Args:
        memory: `memory:` section of the config.
        process_rss_mb: This process's resident set size, in MiB (see
            `process_resident_memory_mb`).
        stage: Human-readable checkpoint name included in the error message
            (e.g. `"after model load"`).

    Raises:
        MemoryBudgetExceededError: If `process_rss_mb` exceeds the configured budget.
    """
    if process_rss_mb > memory.memory_budget_mb:
        raise MemoryBudgetExceededError(
            f"Process memory {stage} is {process_rss_mb:.0f} MB, exceeding "
            f"memory_budget_mb={memory.memory_budget_mb} MB configured in "
            "config/model_config.yaml (sized for an 8 GB Jetson Orin Nano with "
            "headroom for the OS). Use a smaller model, switch precision to fp16, "
            "or raise memory_budget_mb if more RAM is actually available."
        )
