"""Tests for `src.config.environment` (week 12, capstone)."""

from __future__ import annotations

import pytest

from src.config.environment import Environment, is_jetson, resolve_environment


def test_resolve_environment_defaults_to_dev_with_no_override_or_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLM_ENVIRONMENT", raising=False)
    assert resolve_environment() is Environment.DEV


def test_resolve_environment_reads_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_ENVIRONMENT", "jetson")
    assert resolve_environment() is Environment.JETSON


def test_resolve_environment_explicit_override_wins_over_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLM_ENVIRONMENT", "jetson")
    assert resolve_environment("dev") is Environment.DEV


def test_resolve_environment_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        resolve_environment("staging")


def test_environment_values_match_src_config_targets() -> None:
    from src.config import TARGETS

    assert {e.value for e in Environment} == set(TARGETS)


def test_is_jetson() -> None:
    assert is_jetson(Environment.JETSON) is True
    assert is_jetson(Environment.DEV) is False
