"""Robot/OT layer -- the capstone's single entry point for "act in the physical world" (week 12).

Composition, not new code: a re-export of week 11's already-built,
already-tested ROS2 and OPC-UA/PLC bridges. Both bundled under one
`robot` layer rather than split into separate packages -- together
they're the two channels this repo has for a decision made in software
to reach actual physical equipment (a ROS2-controlled device, a PLC
tag), which is what a capstone-level "robot" layer means here more than
"ROS2 specifically."

- `ROSInterface` (week 11 day 3): publishes/subscribes on week 03's
  `vision_bridge` ROS2 node's own topic names; a real in-memory pub/sub
  bus when `simulate: true` (the default, no `rclpy` needed), a real
  `rclpy` node otherwise.
- `OPCUABridge` (week 11 day 3): read/write PLC tags, deliberately a
  stub per `course/modules/ot-integration-for-cv-engineer/`, read-only
  by default with a double opt-in required to write.

`manipulator.py` (week 12, day 3) is genuinely new code, not a
re-export: SO-ARM 101 control, the same config-driven `simulate`
split as the two bridges above, but nothing in weeks 1-11 built
manipulator control at all -- see that module's own docstring.
"""

from __future__ import annotations

from assistant.opc_ua_bridge import (
    OPCUABridge,
    OPCUABridgeError,
    OPCUAWriteNotAllowedError,
    TagValue,
)
from assistant.ros_interface import ROSInterface, ROSInterfaceError, ROSMessage
from src.robot.manipulator import (
    JOINT_NAMES,
    JointPosition,
    Manipulator,
    ManipulatorConfig,
    ManipulatorError,
    load_manipulator_config,
)

__all__ = [
    "ROSInterface",
    "ROSInterfaceError",
    "ROSMessage",
    "OPCUABridge",
    "OPCUABridgeError",
    "OPCUAWriteNotAllowedError",
    "TagValue",
    "Manipulator",
    "ManipulatorConfig",
    "ManipulatorError",
    "JointPosition",
    "JOINT_NAMES",
    "load_manipulator_config",
]
