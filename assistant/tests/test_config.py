"""Unit tests for `assistant.utils.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.utils.config import load_bridges_config, load_chat_config, load_parser_config


def test_load_parser_config_parses_the_real_config_file() -> None:
    config = load_parser_config("assistant/config/parser_config.yaml")

    assert config.rate_limit.max_calls_per_window > 0
    assert config.rate_limit.window_seconds > 0
    assert config.rate_limit.max_queue_size > 0
    assert config.rate_limit.max_wait_seconds >= 0
    assert config.claude_api.model
    assert config.claude_api.api_key_env == "ANTHROPIC_API_KEY"
    assert config.claude_api.timeout_s > 0


def test_load_parser_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_parser_config("assistant/config/does_not_exist.yaml")


_MINIMAL_YAML = """
rate_limit:
  max_calls_per_window: {max_calls_per_window}
  window_seconds: {window_seconds}
  max_queue_size: {max_queue_size}
  max_wait_seconds: {max_wait_seconds}
claude_api:
  model: claude-sonnet-5
  api_key_env: SOME_KEY
  timeout_s: {timeout_s}
"""


def _write_config(
    tmp_path: Path,
    max_calls_per_window: float = 10,
    window_seconds: float = 60.0,
    max_queue_size: float = 20,
    max_wait_seconds: float = 5.0,
    timeout_s: float = 15.0,
) -> Path:
    path = tmp_path / "parser_config.yaml"
    path.write_text(
        _MINIMAL_YAML.format(
            max_calls_per_window=max_calls_per_window,
            window_seconds=window_seconds,
            max_queue_size=max_queue_size,
            max_wait_seconds=max_wait_seconds,
            timeout_s=timeout_s,
        )
    )
    return path


def test_load_parser_config_roundtrips_values(tmp_path: Path) -> None:
    path = _write_config(tmp_path, max_calls_per_window=3, window_seconds=30.0)
    config = load_parser_config(path)
    assert config.rate_limit.max_calls_per_window == 3
    assert config.rate_limit.window_seconds == 30.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_calls_per_window", 0),
        ("window_seconds", 0),
        ("max_queue_size", 0),
        ("max_wait_seconds", -1),
    ],
)
def test_load_parser_config_rejects_invalid_rate_limit_values(
    tmp_path: Path, field: str, value: float
) -> None:
    path = _write_config(tmp_path, **{field: value})
    with pytest.raises(ValueError, match="rate_limit"):
        load_parser_config(path)


def test_load_parser_config_rejects_non_positive_timeout(tmp_path: Path) -> None:
    path = _write_config(tmp_path, timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        load_parser_config(path)


def test_load_parser_config_allows_zero_max_wait_seconds(tmp_path: Path) -> None:
    path = _write_config(tmp_path, max_wait_seconds=0)
    config = load_parser_config(path)
    assert config.rate_limit.max_wait_seconds == 0.0


# -- load_chat_config ----------------------------------------------------------------


def test_load_chat_config_parses_the_real_config_file() -> None:
    config = load_chat_config("assistant/config/chat_config.yaml")

    assert config.db_path
    assert config.host
    assert 1 <= config.port <= 65535


def test_load_chat_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_chat_config("assistant/config/does_not_exist.yaml")


def test_load_chat_config_roundtrips_values(tmp_path: Path) -> None:
    path = tmp_path / "chat_config.yaml"
    path.write_text("chat:\n  db_path: some/path.db\n  host: 127.0.0.1\n  port: 9000\n")

    config = load_chat_config(path)

    assert config.db_path == "some/path.db"
    assert config.host == "127.0.0.1"
    assert config.port == 9000


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_load_chat_config_rejects_out_of_range_port(tmp_path: Path, port: int) -> None:
    path = tmp_path / "chat_config.yaml"
    path.write_text(f"chat:\n  db_path: x.db\n  host: 0.0.0.0\n  port: {port}\n")

    with pytest.raises(ValueError, match="port"):
        load_chat_config(path)


# -- load_bridges_config -------------------------------------------------------------


def test_load_bridges_config_parses_the_real_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM_INFERENCE_URL", raising=False)
    config = load_bridges_config("assistant/config/bridges_config.yaml")

    assert config.vision_bridge.base_url
    assert config.vision_bridge.timeout_s > 0
    assert config.ros_interface.simulate is True
    assert config.opc_ua_bridge.read_only is True


def test_load_bridges_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_bridges_config("assistant/config/does_not_exist.yaml")


def test_load_bridges_config_base_url_defaults_to_the_yaml_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLM_INFERENCE_URL", raising=False)
    config = load_bridges_config("assistant/config/bridges_config.yaml")
    assert config.vision_bridge.base_url == "http://localhost:8000"


def test_load_bridges_config_vlm_inference_url_env_var_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The exact k3s scenario (week 12 day 4): chat-backend and
    # vlm-inference are separate Pods, only reachable via the Service's
    # in-cluster DNS name -- bridges_config.yaml's own
    # http://localhost:8000 default is wrong there.
    monkeypatch.setenv(
        "VLM_INFERENCE_URL", "http://vlm-inference.vlm-edge.svc.cluster.local:8000"
    )
    config = load_bridges_config("assistant/config/bridges_config.yaml")
    assert config.vision_bridge.base_url == "http://vlm-inference.vlm-edge.svc.cluster.local:8000"
