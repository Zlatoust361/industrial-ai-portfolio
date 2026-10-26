"""Shared fixtures/helpers for rag/tests/.

Run from the repo root (`PYTHONPATH=. pytest -q rag/tests/`, same
root-qualified convention `fleet/`/`captioning/`/`active_learning/`'s
tests use -- `rag/` imports `src.config`/`src.models.*` directly in its
one real-network call site.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script_module(filename: str) -> ModuleType:
    """Import a `rag/scripts/*.py` file by path. Same helper/reasoning as
    `peft/`/`fleet/`/`captioning/`/`active_learning/`'s `tests/conftest.py`."""
    module_name = filename.removesuffix(".py").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def assert_reaches_network_boundary(callable_under_test: object) -> None:
    """Call `callable_under_test`, and assert it fails on a REAL network/HF resolution error.

    Same helper/reasoning as `peft/`/`captioning/`/`active_learning/`'s
    `tests/conftest.py`.
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
        or "openai/clip" in message
        or "huggingface" in message
    )
    assert looks_like_network_failure, (
        f"Expected a network/HF-resolution failure, got: {type(exc).__name__}: {exc}"
    )
