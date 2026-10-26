"""Tests for `src.robot.manipulator` (week 12, day 3).

The `simulate=True` path (`_SimulatedManipulatorBackend`) is real,
independently-testable code -- exercised for real (real connect/torque/
position semantics), same reasoning as `ros_interface.py`/
`opc_ua_bridge.py`'s tests. The real serial path
(`_SerialManipulatorBackend`) has no physical SO-ARM 101/serial port in
this sandbox -- verified only to its exact `ModuleNotFoundError`
boundary (no `pyserial` installed here), plus standalone tests of the
packet-framing helpers themselves (pure functions, no I/O, fully
testable without hardware).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.robot.manipulator import (
    _INSTRUCTION_WRITE,
    JOINT_NAMES,
    JointPosition,
    Manipulator,
    ManipulatorConfig,
    ManipulatorError,
    _build_packet,
    _bytes_to_position,
    _checksum,
    _parse_response,
    _position_to_bytes,
    _SerialManipulatorBackend,
    _SimulatedManipulatorBackend,
    load_manipulator_config,
)

# -- JointPosition ------------------------------------------------------------------


def test_joint_position_as_dict_matches_joint_names_order() -> None:
    pos = JointPosition(1, 2, 3, 4, 5, 6)
    assert pos.as_dict() == dict(zip(JOINT_NAMES, [1, 2, 3, 4, 5, 6], strict=True))


def test_joint_position_from_dict_roundtrips() -> None:
    values = {name: i for i, name in enumerate(JOINT_NAMES)}
    pos = JointPosition.from_dict(values)
    assert pos.as_dict() == values


def test_joint_position_from_dict_missing_joint_raises() -> None:
    values = {name: 0 for name in JOINT_NAMES if name != "gripper"}
    with pytest.raises(KeyError):
        JointPosition.from_dict(values)


def test_joint_position_rejects_unknown_field() -> None:
    with pytest.raises(TypeError):
        JointPosition(1, 2, 3, 4, 5, 6, sholder_pan=1)  # type: ignore[call-arg]


# -- load_manipulator_config ----------------------------------------------------------


def test_load_manipulator_config_parses_the_real_config_file() -> None:
    config = load_manipulator_config("config/manipulator.yaml")

    assert config.simulate is True
    assert config.enabled is False
    assert set(config.servo_ids) == set(JOINT_NAMES)
    assert config.servo_ids["shoulder_pan"] == 1
    assert config.servo_ids["gripper"] == 6
    assert config.position_min < config.position_max
    assert config.move_timeout_s > 0


def test_load_manipulator_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_manipulator_config("config/does_not_exist.yaml")


_MINIMAL_YAML = """
manipulator:
  simulate: true
  enabled: false
  port: "/dev/ttyACM0"
  baud_rate: {baud_rate}
  servo_ids:
    shoulder_pan: 1
    shoulder_lift: 2
    elbow_flex: 3
    wrist_flex: 4
    wrist_roll: 5
    gripper: 6
  register_map:
    torque_enable: 40
    goal_position: 42
    present_position: 56
  position_min: {position_min}
  position_max: {position_max}
  home_position:
    shoulder_pan: 2048
    shoulder_lift: 2048
    elbow_flex: 2048
    wrist_flex: 2048
    wrist_roll: 2048
    gripper: 2048
  move_timeout_s: {move_timeout_s}
"""


def _write_config(
    tmp_path: Path,
    baud_rate: int = 1000000,
    position_min: int = 0,
    position_max: int = 4095,
    move_timeout_s: float = 1.0,
) -> Path:
    path = tmp_path / "manipulator.yaml"
    path.write_text(
        _MINIMAL_YAML.format(
            baud_rate=baud_rate,
            position_min=position_min,
            position_max=position_max,
            move_timeout_s=move_timeout_s,
        )
    )
    return path


def test_load_manipulator_config_rejects_position_min_gte_max(tmp_path: Path) -> None:
    path = _write_config(tmp_path, position_min=100, position_max=100)
    with pytest.raises(ValueError, match="position_min"):
        load_manipulator_config(path)


def test_load_manipulator_config_rejects_non_positive_baud_rate(tmp_path: Path) -> None:
    path = _write_config(tmp_path, baud_rate=0)
    with pytest.raises(ValueError, match="baud_rate"):
        load_manipulator_config(path)


def test_load_manipulator_config_rejects_non_positive_timeout(tmp_path: Path) -> None:
    path = _write_config(tmp_path, move_timeout_s=0)
    with pytest.raises(ValueError, match="move_timeout_s"):
        load_manipulator_config(path)


# -- packet framing (pure functions, no I/O) -----------------------------------------


def test_position_to_bytes_and_back_roundtrips() -> None:
    for position in (0, 1, 2048, 4095):
        assert _bytes_to_position(_position_to_bytes(position)) == position


def test_position_to_bytes_is_little_endian() -> None:
    assert _position_to_bytes(0x0102) == bytes([0x02, 0x01])


def test_build_packet_starts_with_the_real_header() -> None:
    packet = _build_packet(1, _INSTRUCTION_WRITE, bytes([42, 0, 8]))
    assert packet[0:2] == b"\xff\xff"
    assert packet[2] == 1  # servo id


def test_build_packet_checksum_is_internally_consistent() -> None:
    packet = _build_packet(3, _INSTRUCTION_WRITE, bytes([42, 0, 8]))
    body = packet[2:-1]
    assert packet[-1] == _checksum(body)


def test_parse_response_extracts_params() -> None:
    body = bytes([1, 4, 0]) + _position_to_bytes(1500)  # id, length, error, params
    response = b"\xff\xff" + body + bytes([_checksum(body)])
    assert _parse_response(response) == _position_to_bytes(1500)


def test_parse_response_rejects_malformed_header() -> None:
    with pytest.raises(ManipulatorError, match="Malformed"):
        _parse_response(b"\x00\x00\x01\x04\x00\x01\x02\x03")


def test_parse_response_rejects_too_short_response() -> None:
    with pytest.raises(ManipulatorError, match="Malformed"):
        _parse_response(b"\xff\xff\x01")


# -- _SimulatedManipulatorBackend: real behavior, not a mock -------------------------


def _home() -> JointPosition:
    return JointPosition(2048, 2048, 2048, 2048, 2048, 2048)


def test_simulated_backend_write_before_connect_raises() -> None:
    backend = _SimulatedManipulatorBackend(_home())
    with pytest.raises(ManipulatorError, match="not connected"):
        backend.write_positions({"gripper": 0})


def test_simulated_backend_write_before_torque_enabled_raises() -> None:
    backend = _SimulatedManipulatorBackend(_home())
    backend.connect()
    with pytest.raises(ManipulatorError, match="torque"):
        backend.write_positions({"gripper": 0})


def test_simulated_backend_write_after_torque_enabled_updates_state() -> None:
    backend = _SimulatedManipulatorBackend(_home())
    backend.connect()
    backend.torque_enable(True)
    backend.write_positions({"gripper": 0})
    assert backend.read_positions()["gripper"] == 0
    assert backend.read_positions()["shoulder_pan"] == 2048  # untouched joints unchanged


def test_simulated_backend_disconnect_disables_torque() -> None:
    backend = _SimulatedManipulatorBackend(_home())
    backend.connect()
    backend.torque_enable(True)
    backend.disconnect()
    backend.connect()  # reconnect without re-enabling torque
    with pytest.raises(ManipulatorError, match="torque"):
        backend.write_positions({"gripper": 0})


# -- Manipulator: the public interface, real end to end in simulate mode -------------


def _manipulator_config(**overrides: object) -> ManipulatorConfig:
    base = dict(
        simulate=True,
        enabled=False,
        port="/dev/ttyACM0",
        baud_rate=1000000,
        servo_ids=dict(zip(JOINT_NAMES, range(1, 7), strict=True)),
        register_map={"torque_enable": 40, "goal_position": 42, "present_position": 56},
        position_min=0,
        position_max=4095,
        home_position=_home(),
        move_timeout_s=0.0,
    )
    base.update(overrides)
    return ManipulatorConfig(**base)  # type: ignore[arg-type]


def test_manipulator_connect_enables_torque_and_move_to_works() -> None:
    with Manipulator(config=_manipulator_config()) as arm:
        arm.move_to(JointPosition(1000, 1000, 1000, 1000, 1000, 1000))
        assert arm.get_positions()["shoulder_pan"] == 1000


def test_manipulator_move_to_out_of_range_raises() -> None:
    with Manipulator(config=_manipulator_config()) as arm:
        with pytest.raises(ManipulatorError, match="outside"):
            arm.move_to(JointPosition(5000, 0, 0, 0, 0, 0))


def test_manipulator_open_close_gripper_only_touches_gripper_joint() -> None:
    with Manipulator(config=_manipulator_config()) as arm:
        arm.move_to(JointPosition(1000, 1000, 1000, 1000, 1000, 1000))
        arm.open_gripper(4095)
        positions = arm.get_positions()
        assert positions["gripper"] == 4095
        assert positions["shoulder_pan"] == 1000  # unrelated joints preserved

        arm.close_gripper(0)
        assert arm.get_positions()["gripper"] == 0


def test_manipulator_home_returns_to_configured_home_position() -> None:
    with Manipulator(config=_manipulator_config()) as arm:
        arm.move_to(JointPosition(1000, 1000, 1000, 1000, 1000, 1000))
        arm.home()
        assert arm.get_positions() == _home().as_dict()


def test_manipulator_get_positions_before_connect_raises() -> None:
    arm = Manipulator(config=_manipulator_config())
    with pytest.raises(ManipulatorError, match="Not connected"):
        arm.get_positions()


def test_manipulator_disconnect_is_safe_to_call_twice() -> None:
    arm = Manipulator(config=_manipulator_config())
    arm.connect()
    arm.disconnect()
    arm.disconnect()  # must not raise


def test_manipulator_get_status_simulate_mode() -> None:
    with Manipulator(config=_manipulator_config()) as arm:
        status = arm.get_status()
    assert status["mode"] == "simulate"
    assert status["enabled"] is False


def test_manipulator_real_mode_blocks_motion_when_not_enabled() -> None:
    config = _manipulator_config(simulate=False, enabled=False)
    backend = _SimulatedManipulatorBackend(config.home_position)
    arm = Manipulator(config=config, backend=backend)
    arm.connect()
    with pytest.raises(ManipulatorError, match="simulate=false but enabled=false"):
        arm.move_to(config.home_position)


def test_manipulator_real_mode_allows_motion_when_enabled() -> None:
    config = _manipulator_config(simulate=False, enabled=True)
    backend = _SimulatedManipulatorBackend(config.home_position)
    arm = Manipulator(config=config, backend=backend)
    arm.connect()
    arm.move_to(JointPosition(1000, 1000, 1000, 1000, 1000, 1000))
    assert arm.get_positions()["shoulder_pan"] == 1000


# -- _SerialManipulatorBackend: verified only to the real ModuleNotFoundError boundary --


def test_serial_backend_connect_reaches_the_real_pyserial_boundary() -> None:
    backend = _SerialManipulatorBackend(_manipulator_config(simulate=False, enabled=True))
    with pytest.raises(ModuleNotFoundError, match="serial"):
        backend.connect()


def test_manipulator_defaults_to_the_real_serial_backend_when_not_simulating() -> None:
    # Real check, not mocked: this sandbox genuinely has no pyserial
    # installed (same standing limitation as torch/rclpy/asyncua
    # elsewhere in this repo).
    config = _manipulator_config(simulate=False, enabled=True)
    arm = Manipulator(config=config)
    assert isinstance(arm._backend, _SerialManipulatorBackend)  # noqa: SLF001
    with pytest.raises(ModuleNotFoundError):
        arm.connect()
