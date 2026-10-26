"""Tests for `captioning.utils.config.load_config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from captioning.utils.config import load_config

_CONFIG_TEMPLATE = """
model:
  name: HuggingFaceTB/SmolVLM-256M-Instruct
  type: vlm-captioning
  precision: fp32
  device: cpu
  seed: 42
  cache_dir: {cache_dir}
memory:
  memory_budget_mb: 2048
  device_total_mb: 8192
prompting:
  allowed_phrases: ["scratch", "crack"]
  max_new_tokens: 40
trigger:
  mode: {trigger_mode}
paths:
  outputs_dir: {outputs_dir}
  augmented_report_path: {outputs_dir}/augmented_verdicts.json
  evaluation_report_path: {outputs_dir}/evaluation_report.json
"""


def _write_config(tmp_path: Path, *, trigger_mode: str = "on_defect_confirmed") -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            cache_dir=tmp_path / ".cache",
            outputs_dir=tmp_path / "outputs",
            trigger_mode=trigger_mode,
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_config_parses_every_section(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    assert config.model.name == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert config.memory.memory_budget_mb == 2048
    assert config.prompting.allowed_phrases == ("scratch", "crack")
    assert config.prompting.max_new_tokens == 40
    assert config.trigger.mode == "on_defect_confirmed"
    assert config.paths.outputs_dir == tmp_path / "outputs"


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Captioning config file not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_rejects_unknown_trigger_mode(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, trigger_mode="on_demand")
    with pytest.raises(ValueError, match="trigger.mode must be one of"):
        load_config(config_path)
