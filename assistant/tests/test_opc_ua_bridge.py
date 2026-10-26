"""Tests for `assistant.opc_ua_bridge`.

Simulate mode is exercised for real -- a real `_SimulatedTagStore` with
real read/write/unknown-tag semantics, no mocking. Real (non-simulated)
mode is checked against this sandbox's REAL absence of `asyncua`.
"""

from __future__ import annotations

import pytest

from assistant.opc_ua_bridge import OPCUABridge, OPCUABridgeError, OPCUAWriteNotAllowedError
from assistant.utils.config import OPCUABridgeConfig


def _config(read_only: bool = True, simulate: bool = True) -> OPCUABridgeConfig:
    return OPCUABridgeConfig(
        endpoint="opc.tcp://test:4840/",
        namespace="ns=2;s=Test",
        read_only=read_only,
        simulate=simulate,
    )


# -- simulate mode: real tag store -----------------------------------------------


def test_read_tag_returns_a_default_tag_value() -> None:
    with OPCUABridge(config=_config()) as opc:
        value = opc.read_tag("Line1.Assistant.Heartbeat")

    assert value.tag == "Line1.Assistant.Heartbeat"
    assert value.value == 0
    assert value.quality == "good"


def test_read_tag_unknown_tag_raises() -> None:
    with OPCUABridge(config=_config()) as opc, pytest.raises(OPCUABridgeError, match="Unknown tag"):
        opc.read_tag("NoSuchTag")


def test_write_tag_blocked_when_read_only() -> None:
    with OPCUABridge(config=_config(read_only=True)) as opc:
        with pytest.raises(OPCUAWriteNotAllowedError, match="read_only"):
            opc.write_tag("Line1.Assistant.Heartbeat", 1, allow_write=True)


def test_write_tag_blocked_without_explicit_allow_write() -> None:
    with OPCUABridge(config=_config(read_only=False)) as opc:
        with pytest.raises(OPCUAWriteNotAllowedError, match="allow_write"):
            opc.write_tag("Line1.Assistant.Heartbeat", 1)  # allow_write defaults to False


def test_write_tag_succeeds_with_both_opt_ins() -> None:
    with OPCUABridge(config=_config(read_only=False)) as opc:
        opc.write_tag("Line1.Assistant.Heartbeat", 42, allow_write=True)
        assert opc.read_tag("Line1.Assistant.Heartbeat").value == 42


def test_custom_initial_tags_are_used_in_simulate_mode() -> None:
    with OPCUABridge(config=_config(), simulated_tags={"Custom.Tag": "hello"}) as opc:
        assert opc.read_tag("Custom.Tag").value == "hello"


def test_get_status_simulate_mode() -> None:
    with OPCUABridge(config=_config()) as opc:
        status = opc.get_status()

    assert status["mode"] == "simulate"
    assert status["connected"] is True
    assert status["read_only"] is True
    assert status["tag_count"] == 3  # _DEFAULT_SIMULATED_TAGS


def test_disconnect_is_a_noop_in_simulate_mode() -> None:
    opc = OPCUABridge(config=_config())
    opc.disconnect()  # must not raise
    opc.disconnect()  # idempotent


# -- real mode: this sandbox genuinely has no asyncua ----------------------------


def test_get_status_real_mode_reports_missing_asyncua_without_raising() -> None:
    opc = OPCUABridge(config=_config(simulate=False))
    status = opc.get_status()

    assert status["mode"] == "real"
    assert status["connected"] is False
    assert "asyncua" in status["error"]


def test_read_tag_real_mode_raises_importerror() -> None:
    opc = OPCUABridge(config=_config(simulate=False))
    with pytest.raises(ModuleNotFoundError, match="asyncua"):
        opc.read_tag("Some.Tag")


def test_write_tag_real_mode_still_blocked_by_read_only_before_touching_asyncua() -> None:
    # Safety-relevant ordering: the read_only check must happen BEFORE any
    # attempt to even connect, let alone write -- confirmed by getting
    # OPCUAWriteNotAllowedError, not an asyncua ImportError, here.
    opc = OPCUABridge(config=_config(read_only=True, simulate=False))
    with pytest.raises(OPCUAWriteNotAllowedError):
        opc.write_tag("Some.Tag", 1, allow_write=True)
