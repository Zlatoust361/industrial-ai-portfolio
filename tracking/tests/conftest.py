"""Shared fixtures/helpers for tracking/tests/.

Run from `tracking/` itself (`PYTHONPATH=. pytest -q tests/`, same
convention `quantization/`/`peft/`'s self-contained-subsystem tests use --
`tracking/` has no `src.config`/repo-root dependency, so there's no
reason to run it root-qualified like `fleet/`/`assistant/`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script_module(filename: str) -> ModuleType:
    """Import a `tracking/scripts/*.py` file by path.

    Not strictly required for this filename (`run_tracking_demo.py` is a
    valid module name) -- used anyway so the test exercises the exact
    same `sys.path.insert`/top-level code `python3 scripts/
    run_tracking_demo.py` would run, not a package import that might
    resolve paths differently. Same helper, same reasoning, as
    `peft/tests/conftest.py`/`fleet/tests/conftest.py`.

    Args:
        filename: Script filename under `tracking/scripts/`, e.g.
            `"run_tracking_demo.py"`.

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
