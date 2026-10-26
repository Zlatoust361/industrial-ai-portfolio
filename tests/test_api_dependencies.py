"""Unit tests for `src.api.dependencies.get_config`'s path/env-var resolution.

`get_config` is `@lru_cache`d (one config load per process, shared by
every request) -- `get_config.cache_clear()` before each call here so
env-var/argument changes between tests actually take effect instead of
returning a memoized result from an earlier test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.dependencies import DEFAULT_CONFIG_PATH, get_config


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_get_config_defaults_to_default_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONFIG_PATH", raising=False)

    config = get_config()

    assert config.model.name  # DEFAULT_CONFIG_PATH loaded successfully


def test_get_config_reads_config_path_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jetson_config = tmp_path / "model_config.jetson.yaml"
    jetson_config.write_text(Path(DEFAULT_CONFIG_PATH).read_text().replace('"fp32"', '"fp16"'))
    monkeypatch.setenv("CONFIG_PATH", str(jetson_config))

    config = get_config()

    assert config.model.precision == "fp16"


def test_get_config_explicit_argument_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # Points $CONFIG_PATH at a file that doesn't exist -- if the explicit
    # argument didn't win, load_config would raise FileNotFoundError.
    monkeypatch.setenv("CONFIG_PATH", "/does/not/exist.yaml")

    config = get_config(DEFAULT_CONFIG_PATH)

    assert config.model.name
