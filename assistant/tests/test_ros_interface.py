"""Tests for `assistant.ros_interface`.

Simulate mode is exercised for real -- a real `_SimulatedBus` with real
multi-subscriber fan-out, real threading (`wait_for_detection`), no
mocking. Real (non-simulated) mode is checked against this sandbox's REAL
absence of `rclpy` (no ROS2 install here -- see README.md).
"""

from __future__ import annotations

import threading

import pytest

from assistant.ros_interface import ROSInterface, ROSInterfaceError, ROSMessage
from assistant.utils.config import ROSInterfaceConfig


def _config(simulate: bool = True) -> ROSInterfaceConfig:
    return ROSInterfaceConfig(
        text_query_topic="/test/text_query",
        detections_topic="/test/detections",
        training_control_topic="/test/training_control",
        node_name="test_node",
        simulate=simulate,
    )


# -- simulate mode: real pub/sub -------------------------------------------------


def test_publish_text_query_delivers_to_subscribers() -> None:
    with ROSInterface(config=_config()) as ros:
        received = []
        ros._bus.subscribe(ros._config.text_query_topic, received.append)  # noqa: SLF001

        ros.publish_text_query("a scratch")

        assert len(received) == 1
        assert received[0] == ROSMessage(topic="/test/text_query", data="a scratch")


def test_publish_text_query_fans_out_to_multiple_subscribers() -> None:
    with ROSInterface(config=_config()) as ros:
        first, second = [], []
        ros._bus.subscribe(ros._config.text_query_topic, first.append)  # noqa: SLF001
        ros._bus.subscribe(ros._config.text_query_topic, second.append)  # noqa: SLF001

        ros.publish_text_query("hello")

        assert len(first) == 1
        assert len(second) == 1


def test_trigger_training_restart_publishes_a_structured_command() -> None:
    with ROSInterface(config=_config()) as ros:
        message = ros.trigger_training_restart(reason="test reason")

        assert message.topic == "/test/training_control"
        assert message.data == {"command": "restart_training", "reason": "test reason"}


def test_subscribe_detections_then_publish_delivers_the_message() -> None:
    with ROSInterface(config=_config()) as ros:
        received = []
        ros.subscribe_detections(received.append)

        ros._bus.publish(ros._config.detections_topic, {"label": "crack", "score": 0.9})  # noqa: SLF001

        assert len(received) == 1
        assert received[0].data == {"label": "crack", "score": 0.9}


def test_wait_for_detection_returns_the_next_message() -> None:
    with ROSInterface(config=_config()) as ros:

        def publish_soon() -> None:
            ros._bus.publish(ros._config.detections_topic, "arrived")  # noqa: SLF001

        threading.Timer(0.05, publish_soon).start()

        message = ros.wait_for_detection(timeout_s=2.0)

        assert message is not None
        assert message.data == "arrived"


def test_wait_for_detection_times_out_cleanly_when_nothing_arrives() -> None:
    with ROSInterface(config=_config()) as ros:
        assert ros.wait_for_detection(timeout_s=0.05) is None


def test_get_status_simulate_mode() -> None:
    with ROSInterface(config=_config()) as ros:
        status = ros.get_status()

    assert status["mode"] == "simulate"
    assert status["connected"] is True
    assert status["text_query_topic"] == "/test/text_query"


def test_shutdown_is_a_noop_in_simulate_mode() -> None:
    ros = ROSInterface(config=_config())
    ros.shutdown()  # must not raise
    ros.shutdown()  # idempotent


# -- real mode: this sandbox genuinely has no rclpy -----------------------------


def test_get_status_real_mode_reports_missing_rclpy_without_raising() -> None:
    ros = ROSInterface(config=_config(simulate=False))
    status = ros.get_status()

    assert status["mode"] == "real"
    assert status["connected"] is False
    assert "rclpy" in status["error"]


def test_publish_text_query_real_mode_raises_importerror() -> None:
    ros = ROSInterface(config=_config(simulate=False))
    with pytest.raises(ModuleNotFoundError, match="rclpy"):
        ros.publish_text_query("x")


def test_subscribe_detections_real_mode_raises_before_rclpy_even_needed() -> None:
    # Real check: the vision_bridge_msgs package import is attempted
    # first and fails the same way rclpy would (neither is installed) --
    # confirms this code path is real, not a silent pass-through.
    ros = ROSInterface(config=_config(simulate=False))
    with pytest.raises((ModuleNotFoundError, ROSInterfaceError)):
        ros.subscribe_detections(lambda _msg: None)


def test_wait_for_detection_real_mode_raises_ros_interface_error() -> None:
    ros = ROSInterface(config=_config(simulate=False))
    with pytest.raises(ROSInterfaceError, match="simulate mode"):
        ros.wait_for_detection(timeout_s=0.01)
