"""Shared fixtures/helpers for compliance/tests/.

Run from the repo root (`PYTHONPATH=. pytest -q compliance/tests/`, same
root-qualified convention `fleet/`/`captioning/`/`rag/`/`explainability/`'s
tests use). Unlike those weeks, nothing in `compliance/` touches a real
model or the network at all -- `sbom.py`/`license_audit.py`/`checklist.py`
read local package metadata and static text; `adversarial_demo.py` runs
entirely offline (see its own module docstring) -- so there is no
`assert_reaches_network_boundary` helper here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script_module(filename: str) -> ModuleType:
    """Import a `compliance/scripts/*.py` file by path.

    Same helper, same reasoning, as every other week's
    `tests/conftest.py::load_script_module` -- exercises the exact same
    `sys.path.insert`/top-level code `python3 scripts/run_license_audit.py`
    would run.

    Args:
        filename: Script filename under `compliance/scripts/`, e.g.
            `"run_license_audit.py"`.

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
