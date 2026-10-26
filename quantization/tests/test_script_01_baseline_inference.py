"""Script-level tests for `scripts/01_baseline_inference.py`.

`torch` isn't installed in this sandbox (same boundary every other week
hits -- see `../README.md`), so the real success path (a written
`baseline_results.json`) isn't reachable here. What *is* verified for
real: argument parsing, config loading, and that `main()` reaches exactly
`CLIPLoader.load()`'s `import torch` -- not some earlier, unrelated failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config

baseline_inference = load_script_module("01_baseline_inference.py")


def test_main_reaches_torch_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["01_baseline_inference.py", "--config", str(config_path)])

    with pytest.raises(ModuleNotFoundError, match="torch"):
        baseline_inference.main()


def test_build_inference_constructs_without_torch(tmp_path: Path) -> None:
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    # CLIPInference/CLIPLoader construction itself doesn't touch torch --
    # only encode_image/encode_text (via _ensure_loaded -> loader.load())
    # does. Building the inference object is a real, exercised step of
    # main() that doesn't require torch installed.
    inference = baseline_inference.build_inference(config)

    assert inference.loader.config.name == config.model.name
    assert inference.loader.config.precision == "fp16"
