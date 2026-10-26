"""Tests for `src.robot` (week 12, capstone) -- composition, not new logic."""

from __future__ import annotations

import assistant.opc_ua_bridge as opc_ua_bridge_module
import assistant.ros_interface as ros_interface_module
import src.robot as robot


def test_ros_reexports_are_the_real_week11_objects() -> None:
    assert robot.ROSInterface is ros_interface_module.ROSInterface
    assert robot.ROSInterfaceError is ros_interface_module.ROSInterfaceError
    assert robot.ROSMessage is ros_interface_module.ROSMessage


def test_opc_ua_reexports_are_the_real_week11_objects() -> None:
    assert robot.OPCUABridge is opc_ua_bridge_module.OPCUABridge
    assert robot.OPCUABridgeError is opc_ua_bridge_module.OPCUABridgeError
    assert robot.OPCUAWriteNotAllowedError is opc_ua_bridge_module.OPCUAWriteNotAllowedError
    assert robot.TagValue is opc_ua_bridge_module.TagValue


def test_ros_interface_actually_works_through_the_layer() -> None:
    ros = robot.ROSInterface()
    try:
        status = ros.get_status()
        assert status["mode"] == "simulate"
        assert status["connected"] is True
    finally:
        ros.shutdown()


def test_opc_ua_bridge_actually_works_through_the_layer() -> None:
    opc_ua = robot.OPCUABridge()
    try:
        status = opc_ua.get_status()
        assert status["mode"] == "simulate"
        assert status["read_only"] is True
    finally:
        opc_ua.disconnect()
