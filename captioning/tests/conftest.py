"""Shared fixtures/helpers for captioning/tests/.

Run from the repo root (`PYTHONPATH=. pytest -q captioning/tests/`, same
root-qualified convention `fleet/`'s tests use -- `captioning/` imports
`src.config`/`src.models.*` directly, so it needs the repo root on
`PYTHONPATH`, not its own directory).

`assert_reaches_network_boundary` is the same helper (same reasoning)
`peft/tests/conftest.py` already defines: this sandbox's installed
`transformers`/`huggingface_hub` wraps network denial in `httpx2.ProxyError`,
not the plain `OSError` an older version would raise -- tests assert the
SHAPE of the failure (a real network/HF-resolution error), not one
hardcoded exception class.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script_module(filename: str) -> ModuleType:
    """Import a `captioning/scripts/*.py` file by path.

    Not strictly required for this filename (`run_captioning_demo.py` is
    a valid module name) -- used anyway so the test exercises the exact
    same `sys.path.insert`/top-level code `python3 scripts/
    run_captioning_demo.py` would run. Same helper, same reasoning, as
    `peft/tests/conftest.py`/`fleet/tests/conftest.py`/
    `tracking/tests/conftest.py`.

    Args:
        filename: Script filename under `captioning/scripts/`, e.g.
            `"run_captioning_demo.py"`.

    Returns:
        The imported module (running its top-level code, but not `main()`
        -- guarded behind `if __name__ == "__main__":`).
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
        or "smolvlm" in message
        or "huggingface" in message
    )
    assert looks_like_network_failure, (
        f"Expected a network/HF-resolution failure, got: {type(exc).__name__}: {exc}"
    )
