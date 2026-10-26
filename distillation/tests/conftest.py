"""Shared fixtures/helpers for distillation/tests/.

Run from the repo root (`PYTHONPATH=. pytest -q distillation/tests/`,
same root-qualified convention `modern_detectors/`/`classical_cv/`'s
tests use). No network boundary in this week -- every model is trained
from scratch, nothing is downloaded -- so there's no
`assert_reaches_network_boundary` helper here, same as
`predictive_maintenance/tests/conftest.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script_module(filename: str) -> ModuleType:
    """Import a `distillation/scripts/*.py` file by path.

    Same helper, same reasoning, as every other week's
    `tests/conftest.py::load_script_module`.
    """
    module_name = filename.removesuffix(".py").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
