"""Typed loaders for `assistant/config/*.yaml` (week 11).

Same "typed YAML section -> frozen dataclass" shape as `src/config.py`/
`pipeline/utils/config.py`/`quantization/utils/config.py`. Three loaders,
three separate config files: `load_parser_config` (day 2, `llm_parser.py`'s
own rate-limit/Claude-API settings), `load_bridges_config` (day 3,
`action_executor.py`'s three domain bridges), `load_chat_config` (day 4,
`chat_app.py`'s SQLite path/host/port) -- kept apart because they're
unrelated concerns that just happen to all belong to `assistant/`, not
because any of them would grow to need another's settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RateLimitConfig:
    """`rate_limit:` section -- see `assistant.rate_limiter`.

    Attributes:
        max_calls_per_window: Token-bucket capacity -- at most this many
            external API calls per `window_seconds`.
        window_seconds: Refill window, in seconds.
        max_queue_size: `ExternalCallQueue`'s backlog cap before
            `submit()` raises `QueueFullError` instead of enqueuing.
        max_wait_seconds: How long `LLMParser.parse()` waits for a queued
            external call before returning a `"pending"` result instead
            of blocking further.
    """

    max_calls_per_window: int
    window_seconds: float
    max_queue_size: int
    max_wait_seconds: float


@dataclass(frozen=True)
class ClaudeAPIConfig:
    """`claude_api:` section -- see `assistant.llm_parser.call_claude_api`.

    Attributes:
        model: Claude model id to request.
        api_key_env: Environment variable name holding the real API key
            (never the key itself -- nothing here is a secret).
        timeout_s: HTTP request timeout, in seconds.
    """

    model: str
    api_key_env: str
    timeout_s: float


@dataclass(frozen=True)
class ParserConfig:
    """Root config object composed of every section in `parser_config.yaml`."""

    rate_limit: RateLimitConfig
    claude_api: ClaudeAPIConfig


def load_parser_config(path: str | Path = "assistant/config/parser_config.yaml") -> ParserConfig:
    """Load and validate `llm_parser.py`'s config from a YAML file.

    Args:
        path: Path to a YAML file matching `parser_config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `ParserConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field is missing.
        ValueError: If any numeric field is non-positive (`max_wait_seconds`
            may be `0` -- "don't wait at all, always return pending
            immediately" is a valid, if aggressive, choice).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Parser config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    rate_limit_raw = raw["rate_limit"]
    rate_limit = RateLimitConfig(
        max_calls_per_window=int(rate_limit_raw["max_calls_per_window"]),
        window_seconds=float(rate_limit_raw["window_seconds"]),
        max_queue_size=int(rate_limit_raw["max_queue_size"]),
        max_wait_seconds=float(rate_limit_raw["max_wait_seconds"]),
    )
    if rate_limit.max_calls_per_window <= 0:
        raise ValueError("rate_limit.max_calls_per_window must be positive.")
    if rate_limit.window_seconds <= 0:
        raise ValueError("rate_limit.window_seconds must be positive.")
    if rate_limit.max_queue_size <= 0:
        raise ValueError("rate_limit.max_queue_size must be positive.")
    if rate_limit.max_wait_seconds < 0:
        raise ValueError("rate_limit.max_wait_seconds must be >= 0.")

    claude_api_raw = raw["claude_api"]
    claude_api = ClaudeAPIConfig(
        model=str(claude_api_raw["model"]),
        api_key_env=str(claude_api_raw["api_key_env"]),
        timeout_s=float(claude_api_raw["timeout_s"]),
    )
    if claude_api.timeout_s <= 0:
        raise ValueError("claude_api.timeout_s must be positive.")

    return ParserConfig(rate_limit=rate_limit, claude_api=claude_api)


@dataclass(frozen=True)
class VisionBridgeConfig:
    """`vision_bridge:` section -- see `assistant.vision_bridge`.

    Attributes:
        base_url: The already-deployed FastAPI service (week 08) this
            calls into over HTTP -- never a second in-process model copy.
        timeout_s: HTTP request timeout, in seconds.
        prompt_style: `config/prompts.yaml` style name for
            `classify_defect`'s ok-vs-defect reference phrases (week 02).
        detector_backend: One of
            `ros2_ws/src/vision_bridge/vision_bridge/detector_factory.BACKENDS`.
        detector_model_name: HuggingFace Hub id of the OWL-ViT checkpoint.
        detector_score_threshold: Minimum detection confidence to report.
    """

    base_url: str
    timeout_s: float
    prompt_style: str
    detector_backend: str
    detector_model_name: str
    detector_score_threshold: float


@dataclass(frozen=True)
class ROSInterfaceConfig:
    """`ros_interface:` section -- see `assistant.ros_interface`.

    Attributes:
        text_query_topic: Matches `ros2_ws/src/vision_bridge`'s own
            `text_query_topic` launch default.
        detections_topic: Matches its `detections_topic` launch default.
        training_control_topic: Where a real training-orchestration node
            would listen (no such node exists in this repo today).
        node_name: This bridge's own ROS2 node name.
        simulate: `True` runs an in-memory pub/sub bus instead of a real
            `rclpy` node -- see `ROSInterface`'s own docstring.
    """

    text_query_topic: str
    detections_topic: str
    training_control_topic: str
    node_name: str
    simulate: bool


@dataclass(frozen=True)
class OPCUABridgeConfig:
    """`opc_ua_bridge:` section -- see `assistant.opc_ua_bridge`.

    Attributes:
        endpoint: OPC-UA server endpoint URL.
        namespace: Placeholder KKS-style tag namespace -- see
            `course/modules/ot-integration-for-cv-engineer/day04-plc-scada-integration-kks.md`
            for what a real tag scheme looks like; this is intentionally
            not that.
        read_only: When `True`, `OPCUABridge.write_tag` always raises --
            writing to a PLC is safety-relevant, opt-in must be explicit.
        simulate: `True` runs an in-memory tag store instead of a real
            `asyncua` client.
    """

    endpoint: str
    namespace: str
    read_only: bool
    simulate: bool


@dataclass(frozen=True)
class BridgesConfig:
    """Root config object composed of every section in `bridges_config.yaml`."""

    vision_bridge: VisionBridgeConfig
    ros_interface: ROSInterfaceConfig
    opc_ua_bridge: OPCUABridgeConfig


def load_bridges_config(path: str | Path = "assistant/config/bridges_config.yaml") -> BridgesConfig:
    """Load and validate `action_executor.py`'s three bridge configs from a YAML file.

    Args:
        path: Path to a YAML file matching `bridges_config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `BridgesConfig`. `vision_bridge.base_url`
        resolves `$VLM_INFERENCE_URL` over the YAML value when the env
        var is set -- same "env var overrides YAML default" order
        `src/api/dependencies.py::get_config`'s `$CONFIG_PATH` already
        established (week 08), needed here because
        `bridges_config.yaml`'s own `http://localhost:8000` default is
        only correct when this process and the deployed CLIP service
        share a network namespace (docker-compose, a bare host process)
        -- on k3s (week 12 day 4) they're separate Pods, reachable only
        via the Service's in-cluster DNS name
        (`k3s/chat-backend-deployment.yaml` sets this env var to exactly
        that).

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field is missing.
        ValueError: If `vision_bridge.timeout_s`/`detector_score_threshold`
            is non-positive, or `vision_bridge.detector_backend` isn't a
            known `detector_factory.BACKENDS` value.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Bridges config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    vision_raw = raw["vision_bridge"]
    vision = VisionBridgeConfig(
        base_url=os.environ.get("VLM_INFERENCE_URL", str(vision_raw["base_url"])),
        timeout_s=float(vision_raw["timeout_s"]),
        prompt_style=str(vision_raw["prompt_style"]),
        detector_backend=str(vision_raw["detector_backend"]),
        detector_model_name=str(vision_raw["detector_model_name"]),
        detector_score_threshold=float(vision_raw["detector_score_threshold"]),
    )
    if vision.timeout_s <= 0:
        raise ValueError("vision_bridge.timeout_s must be positive.")
    if vision.detector_score_threshold <= 0:
        raise ValueError("vision_bridge.detector_score_threshold must be positive.")
    known_backends = ("auto", "tensorrt", "pytorch", "stub")
    if vision.detector_backend not in known_backends:
        raise ValueError(
            f"vision_bridge.detector_backend must be one of {known_backends}, "
            f"got {vision.detector_backend!r}."
        )

    ros_raw = raw["ros_interface"]
    ros = ROSInterfaceConfig(
        text_query_topic=str(ros_raw["text_query_topic"]),
        detections_topic=str(ros_raw["detections_topic"]),
        training_control_topic=str(ros_raw["training_control_topic"]),
        node_name=str(ros_raw["node_name"]),
        simulate=bool(ros_raw["simulate"]),
    )

    opc_ua_raw = raw["opc_ua_bridge"]
    opc_ua = OPCUABridgeConfig(
        endpoint=str(opc_ua_raw["endpoint"]),
        namespace=str(opc_ua_raw["namespace"]),
        read_only=bool(opc_ua_raw["read_only"]),
        simulate=bool(opc_ua_raw["simulate"]),
    )

    return BridgesConfig(vision_bridge=vision, ros_interface=ros, opc_ua_bridge=opc_ua)


@dataclass(frozen=True)
class ChatConfig:
    """`chat:` section -- see `assistant.chat_app`.

    Attributes:
        db_path: Where `assistant.database.Database`'s SQLite file lives,
            repo-root-relative.
        host: Interface `uvicorn` binds to.
        port: Port `uvicorn` binds to.
    """

    db_path: str
    host: str
    port: int


def load_chat_config(path: str | Path = "assistant/config/chat_config.yaml") -> ChatConfig:
    """Load and validate `chat_app.py`'s config from a YAML file.

    Args:
        path: Path to a YAML file matching `chat_config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `ChatConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field is missing.
        ValueError: If `port` is not in `[1, 65535]`.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Chat config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    chat_raw = raw["chat"]
    chat = ChatConfig(
        db_path=str(chat_raw["db_path"]),
        host=str(chat_raw["host"]),
        port=int(chat_raw["port"]),
    )
    if not (1 <= chat.port <= 65535):
        raise ValueError(f"chat.port must be in [1, 65535], got {chat.port}.")

    return chat
