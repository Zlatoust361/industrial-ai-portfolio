"""Shared fixtures/helpers for modern_detectors/tests/.

Run from the repo root (`PYTHONPATH=. pytest -q modern_detectors/tests/`,
same root-qualified convention `classical_cv/`/`compliance/`/`sla/`'s
tests use -- `modern_detectors/` imports `classical_cv.defect_images`/
`src.evaluation.*`/`src.synthetic.*` directly, so it needs the repo root
on `PYTHONPATH`, not its own directory).

`assert_reaches_network_boundary` is the same helper (same reasoning)
`classical_cv/tests/conftest.py`/`captioning/tests/conftest.py`/
`peft/tests/conftest.py` already define.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script_module(filename: str) -> ModuleType:
    """Import a `modern_detectors/scripts/*.py` file by path.

    Same helper, same reasoning, as every other week's
    `tests/conftest.py::load_script_module`.

    Args:
        filename: Script filename under `modern_detectors/scripts/`.

    Returns:
        The imported module (running its top-level code, but not
        `main()` -- guarded behind `if __name__ == "__main__":`).
    """
    module_name = filename.removesuffix(".py").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def assert_reaches_network_boundary(callable_under_test: object) -> None:
    """Call `callable_under_test`, and assert it fails on a REAL network/HF resolution error.

    Args:
        callable_under_test: A zero-argument callable expected to raise.

    Raises:
        AssertionError: If `callable_under_test` doesn't raise at all, or
            raises something that doesn't look like a network/HF-resolution
            failure (most likely a real bug in this week's own code).
    """
    import pytest

    with pytest.raises(Exception) as exc_info:  # noqa: PT011 -- deliberately broad, see docstring
        callable_under_test()

    exc = exc_info.value
    assert not isinstance(exc, (TypeError, AttributeError, KeyError, NameError)), (
        f"Expected a network/HF-resolution failure, got a likely real bug instead: "
        f"{type(exc).__name__}: {exc}"
    )
    message = str(exc).lower()
    type_name = type(exc).__name__.lower()
    looks_like_network_failure = (
        isinstance(exc, OSError)
        or "proxy" in type_name
        or "connect" in type_name
        or "403" in message
        or "forbidden" in message
        or "owlvit" in message
        or "huggingface" in message
    )
    assert looks_like_network_failure, (
        f"Expected a network/HF-resolution failure, got: {type(exc).__name__}: {exc}"
    )
