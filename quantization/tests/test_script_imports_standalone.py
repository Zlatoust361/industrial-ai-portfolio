"""Regression test: every `scripts/*.py` must be runnable standalone, with
no `PYTHONPATH` set -- exactly how a real user runs them (see
`../README.md`: "no PYTHONPATH= juggling is needed").

This caught a real bug during week 09 day 3's work: `07_evaluate_quality.py`
only added `quantization/` to `sys.path`, not the repo root -- its (new)
import of `utils.power_mode`, which itself does `from src.evaluation.jetson
import parse_nvpmodel_query`, then failed with `ModuleNotFoundError: No
module named 'src'` for a real standalone invocation. Every
`load_script_module`-based test in this directory (the usual way scripts
are tested here) missed it, because root `pyproject.toml`'s `[tool.pytest.
ini_options] pythonpath = ["."]` puts the repo root on `sys.path` for every
pytest run regardless of cwd -- a real user invoking `python3 scripts/07_...py`
directly gets no such help. `--help` is enough to catch this: every
module-level import runs before `argparse` ever prints anything, so a
clean `--help` exit proves the import wiring is self-sufficient without
touching Jetson/`torch`/`tensorrt` at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((QUANTIZATION_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script_path", SCRIPTS, ids=lambda p: p.name)
def test_script_help_runs_without_external_pythonpath(script_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=QUANTIZATION_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"{script_path.name} --help failed with no PYTHONPATH set:\n{result.stderr}"
    )
    assert "ModuleNotFoundError" not in result.stderr
