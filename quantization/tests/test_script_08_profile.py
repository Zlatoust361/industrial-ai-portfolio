"""Script-level tests for `scripts/08_profile.py`.

Unlike every other `test_script_*.py` in this directory, this one is NOT
smoke-tested to a hardware import boundary -- `08_profile.py` only reads
already-written JSON and renders Markdown, so it runs for real, end to
end, in this sandbox.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config

profile_script = load_script_module("08_profile.py")

_FP16_REPORT = {
    "metadata": {"label": "fp16", "power_mode": {"mode_name": "15W", "mode_id": 1}},
    "latency": {
        "batch_size": 1,
        "mean_ms": 10.0,
        "p50_ms": 9.5,
        "p95_ms": 12.0,
        "throughput_ips": 100.0,
    },
    "temperatures": {"mean_temps_c": {"GPU": 45.0}, "max_temps_c": {"GPU": 50.0}},
}
_INT4_REPORT = {
    "metadata": {"label": "int4", "power_mode": {"mode_name": "15W", "mode_id": 1}},
    "latency": {
        "batch_size": 1,
        "mean_ms": 5.0,
        "p50_ms": 4.8,
        "p95_ms": 6.0,
        "throughput_ips": 200.0,
    },
    "temperatures": {"mean_temps_c": {"GPU": 46.0}, "max_temps_c": {"GPU": 52.0}},
}
_QUALITY_REPORT = {"mean_cosine_similarity": 0.995, "passed": True, "num_images": 64}


def _write_inference_reports(outputs_dir: Path) -> None:
    inference_dir = outputs_dir / "trt_inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    (inference_dir / "fp16_15W.json").write_text(json.dumps(_FP16_REPORT))
    (inference_dir / "int4_15W.json").write_text(json.dumps(_INT4_REPORT))


def test_main_writes_real_summary_from_real_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    _write_inference_reports(tmp_path / "outputs")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["08_profile.py", "--config", str(config_path)])

    profile_script.main()

    summary = (tmp_path / "outputs" / "summary.md").read_text()
    assert "# Week 09" in summary
    assert "2.00x" in summary
    assert "No `07_evaluate_quality.py` report found" in summary


def test_main_includes_quality_report_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    outputs_dir = tmp_path / "outputs"
    _write_inference_reports(outputs_dir)
    quality_dir = outputs_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "quality_report.json").write_text(json.dumps(_QUALITY_REPORT))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["08_profile.py", "--config", str(config_path)])

    profile_script.main()

    summary = (tmp_path / "outputs" / "summary.md").read_text()
    assert "0.9950" in summary
    assert "PASSED" in summary


def test_main_raises_clear_error_without_inference_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["08_profile.py", "--config", str(config_path)])

    with pytest.raises(FileNotFoundError, match="06_trt_inference"):
        profile_script.main()


def test_main_respects_cli_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_config(tmp_path)
    custom_inference_dir = tmp_path / "elsewhere" / "trt_inference"
    custom_inference_dir.mkdir(parents=True)
    (custom_inference_dir / "fp16_15W.json").write_text(json.dumps(_FP16_REPORT))
    (custom_inference_dir / "int4_15W.json").write_text(json.dumps(_INT4_REPORT))
    custom_output = tmp_path / "custom_summary.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_profile.py",
            "--config",
            str(config_path),
            "--inference-dir",
            str(custom_inference_dir),
            "--output",
            str(custom_output),
        ],
    )

    profile_script.main()

    assert custom_output.is_file()
    assert "# Week 09" in custom_output.read_text()
