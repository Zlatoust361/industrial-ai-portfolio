"""Script-level tests for `scripts/02_export_onnx.py`.

`torch` isn't installed in this sandbox (same boundary as `01`/every other
week) -- the real export/smoke-test path isn't reachable here. What's
verified for real: argument parsing, config loading, and that `main()`
reaches exactly `CLIPLoader.load()`'s `import torch`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config

export_onnx = load_script_module("02_export_onnx.py")


def test_main_reaches_torch_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["02_export_onnx.py", "--config", str(config_path)])

    with pytest.raises(ModuleNotFoundError, match="torch"):
        export_onnx.main()


def test_main_with_skip_smoke_test_still_reaches_torch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["02_export_onnx.py", "--config", str(config_path), "--skip-smoke-test"]
    )

    with pytest.raises(ModuleNotFoundError, match="torch"):
        export_onnx.main()


def test_run_smoke_test_accepts_fp16_dummy_input(tmp_path: Path) -> None:
    """Regression test for a real bug found on a Jetson Orin Nano.

    `run_smoke_test` used to cast the ONNX RUNTIME INPUT to fp32
    unconditionally (`dummy_input.float().cpu().numpy()`), but the graph it
    runs against was traced/exported with whatever dtype `dummy_input` had
    at `export_to_onnx()` time -- fp16, when `model.precision: "fp16"`
    (`CLIPLoader` honors fp16 regardless of device, see `src/models/clip.py`).
    ONNX Runtime rejected the mismatch outright: `InvalidArgument: Unexpected
    input data type. Actual: (tensor(float)), expected: (tensor(float16))`.
    Uses a tiny synthetic linear layer, not real CLIP -- this needs no
    network/HF download, unlike `test_main_reaches_torch_boundary` above.
    """
    import torch
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    class TinyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.fc(pixel_values)

    encoder = TinyEncoder().half().eval()
    dummy_input = torch.zeros(1, 4, dtype=torch.float16)
    onnx_path = tmp_path / "tiny.onnx"

    export_onnx.export_to_onnx(encoder, dummy_input, onnx_path, config)
    # Must not raise onnxruntime's InvalidArgument dtype-mismatch error.
    export_onnx.run_smoke_test(encoder, onnx_path, dummy_input, config)
