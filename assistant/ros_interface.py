"""ros_interface.py -- bridges chat commands to/from ROS2 (week 11, day 3).

Real `rclpy` isn't installed in the sandbox this was written in (no ROS2
on the dev machine -- see `README.md`'s "Week 11" section), so this
module runs in one of two modes, chosen by `config.simulate`:

- **`simulate=True`** (default, `assistant/config/bridges_config.yaml`):
  an in-memory publish/subscribe bus (`_SimulatedBus`) with the exact
  same topic names week 03's `ros2_ws/src/vision_bridge` launch file
  uses by default -- real pub/sub semantics (multiple subscribers, FIFO
  delivery, no rclpy), for CI/dev/testing without a ROS2 install.
- **`simulate=False`**: a real `rclpy` node. Lazy-imported (same reason
  `torch`/`diffusers`/`asyncua` are lazy-imported elsewhere in this repo
  -- a large, environment-specific dependency nobody needs just to run
  this repo's test suite) and verified in this sandbox only to the exact
  `ModuleNotFoundError` boundary, the same "smoke test to the boundary"
  pattern used throughout for GPU/ROS2/PLC-dependent code.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant.utils.config import ROSInterfaceConfig, load_bridges_config


class ROSInterfaceError(RuntimeError):
    """Raised for real-rclpy connectivity failures (never raised in simulate mode)."""


@dataclass(frozen=True)
class ROSMessage:
    """One message published/received on a topic -- the same shape in simulated and real mode."""

    topic: str
    data: object


class _SimulatedBus:
    """In-memory pub/sub: real fan-out to every subscriber, no rclpy.

    Not a mock of `rclpy` -- a genuinely separate, real implementation of
    the same publish/subscribe contract, so `ROSInterface`'s callers
    (`action_executor.py`) exercise real multi-subscriber delivery
    ordering/semantics in tests, not a stand-in that always trivially
    "works."
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[ROSMessage], None]]] = {}
        self._history: dict[str, list[ROSMessage]] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, data: object) -> None:
        message = ROSMessage(topic=topic, data=data)
        with self._lock:
            self._history.setdefault(topic, []).append(message)
            subscribers = list(self._subscribers.get(topic, ()))
        for callback in subscribers:
            callback(message)

    def subscribe(self, topic: str, callback: Callable[[ROSMessage], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def history(self, topic: str) -> list[ROSMessage]:
        with self._lock:
            return list(self._history.get(topic, ()))


@dataclass
class _CollectingSubscription:
    """A subscriber that just buffers every message it receives, for `wait_for_message`."""

    messages: list[ROSMessage] = field(default_factory=list)
    _event: threading.Event = field(default_factory=threading.Event)

    def __call__(self, message: ROSMessage) -> None:
        self.messages.append(message)
        self._event.set()

    def wait(self, timeout_s: float) -> bool:
        return self._event.wait(timeout=timeout_s)


class ROSInterface:
    """Publishes text queries / training-control commands, subscribes to detections."""

    def __init__(self, config: ROSInterfaceConfig | None = None) -> None:
        """
        Args:
            config: Defaults to `load_bridges_config().ros_interface`.
        """
        self._config = config if config is not None else load_bridges_config().ros_interface
        self._bus: _SimulatedBus | None = _SimulatedBus() if self._config.simulate else None
        self._node: object | None = None

    def _ensure_real_node(self) -> object:
        """Lazily builds (and caches) a real `rclpy` node.

        Raises:
            ImportError: If `rclpy` isn't installed (this sandbox --
                nothing here has a code path around that; a real
                deployment needs real ROS2).
        """
        if self._node is not None:
            return self._node
        import rclpy
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node(self._config.node_name)
        self._text_query_pub = node.create_publisher(String, self._config.text_query_topic, 10)
        self._training_control_pub = node.create_publisher(
            String, self._config.training_control_topic, 10
        )
        self._node = node
        return node

    def publish_text_query(self, text: str) -> None:
        """Publish `text` on `config.text_query_topic`.

        Week 03's `vision_bridge_node.py` subscribes here.

        Args:
            text: The open-vocabulary query text.

        Raises:
            ImportError: In real (non-simulated) mode, if `rclpy` isn't installed.
        """
        if self._bus is not None:
            self._bus.publish(self._config.text_query_topic, text)
            return
        # _ensure_real_node() first, message-type import second: if
        # neither rclpy nor std_msgs is installed, the node's own absence
        # is the more fundamental problem and should surface first.
        node = self._ensure_real_node()
        from std_msgs.msg import String

        self._text_query_pub.publish(String(data=text))
        node.get_logger().info(f"published text_query: {text!r}")

    def trigger_training_restart(self, reason: str = "operator command") -> ROSMessage:
        """Publish a training-restart command on `config.training_control_topic`.

        No real training-orchestration ROS2 node subscribes to this topic
        anywhere in this repo today -- weeks 07/09's training scripts are
        offline CLI batch jobs (`scripts/train_dr_yolo.py`,
        `quantization/scripts/04_quantize_model.py`), not a live service.
        This demonstrates the command-dispatch/action-triggering path a
        real trainer would listen on, not a wired trainer.

        Args:
            reason: Human-readable justification, carried in the message
                for audit/logging.

        Returns:
            The `ROSMessage` that was published (simulate mode) or a
            `ROSMessage` describing what was sent (real mode).

        Raises:
            ImportError: In real mode, if `rclpy` isn't installed.
        """
        payload = {"command": "restart_training", "reason": reason}
        if self._bus is not None:
            self._bus.publish(self._config.training_control_topic, payload)
            return ROSMessage(topic=self._config.training_control_topic, data=payload)

        node = self._ensure_real_node()
        import json

        from std_msgs.msg import String

        self._training_control_pub.publish(String(data=json.dumps(payload)))
        node.get_logger().info(f"published training_control: {payload}")
        return ROSMessage(topic=self._config.training_control_topic, data=payload)

    def subscribe_detections(self, callback: Callable[[ROSMessage], None]) -> None:
        """Register `callback` to run on every message published to `config.detections_topic`.

        Args:
            callback: Called with each `ROSMessage` as it arrives.

        Raises:
            ImportError: In real mode, if `rclpy` isn't installed.
        """
        if self._bus is not None:
            self._bus.subscribe(self._config.detections_topic, callback)
            return
        from vision_bridge_msgs.msg import DetectionArray  # noqa: F401 -- real-mode only; the

        # import itself is the point here (proves the message package is
        # importable), actual subscription wiring is real-deployment work
        # beyond this sandbox's ability to verify.
        raise ROSInterfaceError(
            "Real (non-simulated) ROS2 subscription wiring needs a live rclpy executor "
            "spinning this node -- not implemented for the sandbox this was written in."
        )

    def wait_for_detection(self, timeout_s: float) -> ROSMessage | None:
        """Block up to `timeout_s` for the next message on `config.detections_topic`.

        Simulate mode only.

        Args:
            timeout_s: Max seconds to wait.

        Returns:
            The first matching `ROSMessage`, or `None` on timeout.

        Raises:
            ROSInterfaceError: In real (non-simulated) mode -- use
                `subscribe_detections` with a real executor instead.
        """
        if self._bus is None:
            raise ROSInterfaceError("wait_for_detection is only supported in simulate mode.")
        collector = _CollectingSubscription()
        self._bus.subscribe(self._config.detections_topic, collector)
        if collector.wait(timeout_s) and collector.messages:
            return collector.messages[0]
        return None

    def get_status(self) -> dict[str, object]:
        """Connectivity/health summary -- one input to `action_executor.py`'s `get_status` handler.

        Returns:
            `{"mode": "simulate"|"real", "connected": bool, ...}`. Never
            raises -- a real-mode connectivity failure becomes
            `connected=False` with an `error` field, not an exception.
        """
        if self._bus is not None:
            return {
                "mode": "simulate",
                "connected": True,
                "node_name": self._config.node_name,
                "text_query_topic": self._config.text_query_topic,
                "detections_topic": self._config.detections_topic,
                "training_control_topic": self._config.training_control_topic,
            }
        try:
            self._ensure_real_node()
        except Exception as exc:  # noqa: BLE001 -- connectivity check, never raises
            return {"mode": "real", "connected": False, "error": str(exc)}
        return {"mode": "real", "connected": True, "node_name": self._config.node_name}

    def shutdown(self) -> None:
        """Tear down the real `rclpy` node, if one was created. No-op in simulate mode."""
        if self._node is None:
            return
        import rclpy

        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._node = None

    def __enter__(self) -> ROSInterface:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
