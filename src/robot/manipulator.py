"""SO-ARM 101 manipulator control interface (week 12, day 3).

Genuinely new code, not a re-export like `src/robot/__init__.py`'s other
two bridges (`ROSInterface`/`OPCUABridge`, week 11) -- no prior week
built manipulator control at all. Real facts about the hardware below
(servo model, joint names/IDs, serial bus shape) come from LeRobot's own
published SO-101/SO-ARM101 documentation, not guessed: six Feetech
STS3215 servos (`shoulder_pan`=1, `shoulder_lift`=2, `elbow_flex`=3,
`wrist_flex`=4, `wrist_roll`=5, `gripper`=6) daisy-chained on one
USB-serial bus.

Same config-driven `simulate: true|false` split every other bridge in
this repo uses (`ros_interface.py`/`opc_ua_bridge.py`, week 11): a real,
independently-testable in-memory arm-state simulator by default, a real
serial backend implementing the Feetech SCServo (protocol 0) packet
format for the real hardware path. The register addresses
(`config/manipulator.yaml`'s `register_map`) are the commonly-published
STS/SMS control-table addresses (Feetech's own SCServo SDK's memory
map), NOT independently re-verified against a physical SO-ARM 101 in
this sandbox -- there is no serial port and no arm here. `move_to`
requires BOTH `config.simulate == False` AND `config.enabled == True`
for real hardware to move at all -- the same double-opt-in
`opc_ua_bridge.py::write_tag` already established (ADR-043) for a
different kind of safety-relevant physical action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
"""SO-ARM 101's six joints, in LeRobot's own published order/naming."""

_INSTRUCTION_PING = 0x01
_INSTRUCTION_READ = 0x02
_INSTRUCTION_WRITE = 0x03
_PACKET_HEADER = b"\xff\xff"


class ManipulatorError(RuntimeError):
    """Raised for connectivity failures, malformed responses, and safety-policy violations."""


@dataclass(frozen=True)
class JointPosition:
    """One target (or current) position for all six joints, in raw servo units (0-4095).

    A plain dataclass rather than a `dict[str, int]` so a caller gets a
    typo caught at construction time (`JointPosition(sholder_pan=...)`
    is a `TypeError`, a dict key typo silently does nothing).
    """

    shoulder_pan: int
    shoulder_lift: int
    elbow_flex: int
    wrist_flex: int
    wrist_roll: int
    gripper: int

    def as_dict(self) -> dict[str, int]:
        """This position as `{joint_name: raw_position}`, in `JOINT_NAMES` order."""
        return {name: getattr(self, name) for name in JOINT_NAMES}

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> JointPosition:
        """Build a `JointPosition` from a `{joint_name: raw_position}` mapping.

        Raises:
            KeyError: If any of `JOINT_NAMES` is missing from `values`.
        """
        return cls(**{name: int(values[name]) for name in JOINT_NAMES})


@dataclass(frozen=True)
class ManipulatorConfig:
    """`manipulator:` section of `config/manipulator.yaml`.

    Attributes:
        simulate: `True` runs the in-memory arm simulator; `False` is the
            real serial path.
        enabled: Deployment-level "is real motion allowed at all" gate --
            independent of `simulate`, mirrors `OPCUABridgeConfig.read_only`.
        port: Serial device path (real mode only).
        baud_rate: Serial baud rate (real mode only).
        servo_ids: Joint name -> Feetech servo ID on the shared bus.
        register_map: Control-table addresses this backend writes/reads
            (`torque_enable`/`goal_position`/`present_position`).
        position_min: Minimum valid raw position (servo units).
        position_max: Maximum valid raw position (servo units).
        home_position: The arm's configured home/rest pose.
        move_timeout_s: Serial read timeout, and how long `move_to`
            waits before considering a move settled (see the module
            docstring's honesty note -- no real motion-complete feedback
            exists yet).
    """

    simulate: bool
    enabled: bool
    port: str
    baud_rate: int
    servo_ids: dict[str, int]
    register_map: dict[str, int]
    position_min: int
    position_max: int
    home_position: JointPosition
    move_timeout_s: float


def load_manipulator_config(path: str | Path = "config/manipulator.yaml") -> ManipulatorConfig:
    """Load and validate the manipulator config from a YAML file.

    Args:
        path: Path to a YAML file matching `config/manipulator.yaml`'s schema.

    Returns:
        A fully parsed, immutable `ManipulatorConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field (including any of `JOINT_NAMES` in
            `servo_ids`/`home_position`) is missing.
        ValueError: If `position_min >= position_max`, `baud_rate` isn't
            positive, or `move_timeout_s` isn't positive.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Manipulator config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section = raw["manipulator"]

    servo_ids = {name: int(section["servo_ids"][name]) for name in JOINT_NAMES}
    register_map = {str(k): int(v) for k, v in section["register_map"].items()}
    home_position = JointPosition.from_dict(section["home_position"])

    config = ManipulatorConfig(
        simulate=bool(section["simulate"]),
        enabled=bool(section["enabled"]),
        port=str(section["port"]),
        baud_rate=int(section["baud_rate"]),
        servo_ids=servo_ids,
        register_map=register_map,
        position_min=int(section["position_min"]),
        position_max=int(section["position_max"]),
        home_position=home_position,
        move_timeout_s=float(section["move_timeout_s"]),
    )
    if config.position_min >= config.position_max:
        raise ValueError(
            f"position_min ({config.position_min}) must be < position_max ({config.position_max})."
        )
    if config.baud_rate <= 0:
        raise ValueError("baud_rate must be positive.")
    if config.move_timeout_s <= 0:
        raise ValueError("move_timeout_s must be positive.")
    return config


def _checksum(body: bytes) -> int:
    """Feetech SCServo (protocol 0) checksum: `~sum(ID, Length, Instruction, Params...) & 0xFF`."""
    return (~sum(body)) & 0xFF


def _position_to_bytes(position: int) -> bytes:
    """16-bit position, low byte first -- Feetech SCServo's documented byte order."""
    return bytes([position & 0xFF, (position >> 8) & 0xFF])


def _bytes_to_position(data: bytes) -> int:
    """Inverse of `_position_to_bytes`."""
    return data[0] | (data[1] << 8)


def _build_packet(servo_id: int, instruction: int, params: bytes) -> bytes:
    """Frame one Feetech SCServo packet: header/id/length/instruction/params/checksum."""
    length = len(params) + 2
    body = bytes([servo_id, length, instruction]) + params
    return _PACKET_HEADER + body + bytes([_checksum(body)])


def _parse_response(response: bytes) -> bytes:
    """Extract the parameter bytes from a Feetech SCServo status packet.

    Args:
        response: The raw bytes read back from the serial port.

    Returns:
        The parameter bytes (empty for a write acknowledgment).

    Raises:
        ManipulatorError: If `response` is too short or doesn't start
            with the expected header.
    """
    if len(response) < 6 or response[0:2] != _PACKET_HEADER:
        raise ManipulatorError(f"Malformed response packet: {response!r}")
    length = response[3]
    return response[5 : 5 + length - 2]


class ManipulatorBackend(Protocol):
    """Structural shape `Manipulator` needs -- satisfied by both backends below."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def torque_enable(self, enabled: bool) -> None: ...
    def write_positions(self, positions: dict[str, int]) -> None: ...
    def read_positions(self) -> dict[str, int]: ...


class _SimulatedManipulatorBackend:
    """Real in-memory arm state -- not a mock of a serial backend.

    Same reasoning as `ros_interface.py::_SimulatedBus`/`opc_ua_bridge.py::_SimulatedTagStore`:
    real, deterministic connect/torque/position semantics (writing before
    connecting, or before torque is enabled, really does raise) are
    exercised for real here, without needing a serial port or a
    physical arm.
    """

    def __init__(self, home_position: JointPosition) -> None:
        self._positions: dict[str, int] = home_position.as_dict()
        self._connected = False
        self._torque_enabled = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._torque_enabled = False

    def torque_enable(self, enabled: bool) -> None:
        if not self._connected:
            raise ManipulatorError("Cannot enable/disable torque: not connected.")
        self._torque_enabled = enabled

    def write_positions(self, positions: dict[str, int]) -> None:
        if not self._connected:
            raise ManipulatorError("Cannot write positions: not connected.")
        if not self._torque_enabled:
            raise ManipulatorError("Cannot write positions: torque is not enabled.")
        self._positions.update(positions)

    def read_positions(self) -> dict[str, int]:
        return dict(self._positions)


class _SerialManipulatorBackend:
    """Real Feetech SCServo (protocol 0) packets over a real serial port.

    Packet framing (header/ID/length/instruction/params/checksum) is the
    publicly documented SCServo protocol-0 structure -- stable and
    well-established, the same reason this repo builds real HTTP/OPC-UA
    request bytes by hand elsewhere rather than guessing. What's NOT
    independently verified here: whether `config.register_map`'s
    addresses match this exact physical unit's firmware (see
    `config/manipulator.yaml`'s own comment), and whether
    `serial`(pyserial)'s real I/O behaves as expected against this
    specific USB-to-serial adapter -- neither is checkable without the
    physical SO-ARM 101 and a serial port, neither available in this
    sandbox.
    """

    def __init__(self, config: ManipulatorConfig) -> None:
        self._config = config
        self._serial: object | None = None

    def connect(self) -> None:
        """Open the real serial port.

        Raises:
            ModuleNotFoundError: If `pyserial` isn't installed (this
                sandbox has no `serial` module -- same lazy-import
                pattern as `rclpy`/`asyncua` elsewhere in this repo).
        """
        import serial

        self._serial = serial.Serial(
            self._config.port, self._config.baud_rate, timeout=self._config.move_timeout_s
        )

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _require_connection(self) -> object:
        if self._serial is None:
            raise ManipulatorError("Not connected -- call connect() first.")
        return self._serial

    def torque_enable(self, enabled: bool) -> None:
        serial_conn = self._require_connection()
        address = self._config.register_map["torque_enable"]
        for servo_id in self._config.servo_ids.values():
            packet = _build_packet(servo_id, _INSTRUCTION_WRITE, bytes([address, int(enabled)]))
            serial_conn.write(packet)

    def write_positions(self, positions: dict[str, int]) -> None:
        serial_conn = self._require_connection()
        address = self._config.register_map["goal_position"]
        for joint, position in positions.items():
            if joint not in self._config.servo_ids:
                raise ManipulatorError(f"Unknown joint {joint!r} (not in servo_ids).")
            servo_id = self._config.servo_ids[joint]
            params = bytes([address]) + _position_to_bytes(position)
            packet = _build_packet(servo_id, _INSTRUCTION_WRITE, params)
            serial_conn.write(packet)

    def read_positions(self) -> dict[str, int]:
        serial_conn = self._require_connection()
        address = self._config.register_map["present_position"]
        positions: dict[str, int] = {}
        for joint, servo_id in self._config.servo_ids.items():
            request = _build_packet(servo_id, _INSTRUCTION_READ, bytes([address, 2]))
            serial_conn.write(request)
            response = serial_conn.read(8)  # header(2)+id(1)+len(1)+err(1)+params(2)+checksum(1)
            params = _parse_response(bytes(response))
            if len(params) != 2:
                raise ManipulatorError(
                    f"Expected a 2-byte position response for joint {joint!r}, got {params!r}."
                )
            positions[joint] = _bytes_to_position(params)
        return positions


class Manipulator:
    """SO-ARM 101 control -- `connect`/`move_to`/`open_gripper`/`close_gripper`/`home`/`disconnect`.

    `scenarios/color_sorting.py`/`defect_detection.py` are the intended
    callers, via `src.scenarios.ScenarioContext.manipulator`.
    """

    def __init__(
        self, config: ManipulatorConfig | None = None, backend: ManipulatorBackend | None = None
    ) -> None:
        """
        Args:
            config: Defaults to `load_manipulator_config()`.
            backend: Defaults to `_SimulatedManipulatorBackend` when
                `config.simulate`, else `_SerialManipulatorBackend`.
        """
        self._config = config if config is not None else load_manipulator_config()
        if backend is not None:
            self._backend = backend
        elif self._config.simulate:
            self._backend = _SimulatedManipulatorBackend(self._config.home_position)
        else:
            self._backend = _SerialManipulatorBackend(self._config)
        self._connected = False

    def connect(self) -> None:
        """Connect and enable torque on every joint. Safe to call more than once."""
        if self._connected:
            return
        self._backend.connect()
        self._backend.torque_enable(True)
        self._connected = True

    def _require_real_motion_allowed(self) -> None:
        if not self._config.simulate and not self._config.enabled:
            raise ManipulatorError(
                "This manipulator is configured simulate=false but enabled=false "
                "(config/manipulator.yaml) -- real motion requires BOTH: simulate=false "
                "alone does not arm the arm. Set enabled=true only once you intend for "
                "real hardware to actually move."
            )

    def move_to(self, position: JointPosition) -> None:
        """Command every joint to `position` at once.

        Args:
            position: Target raw servo positions for all six joints.

        Raises:
            ManipulatorError: If not connected, if real motion isn't
                armed (`config.enabled`), or if any target is outside
                `[config.position_min, config.position_max]`.
        """
        self._require_real_motion_allowed()
        if not self._connected:
            raise ManipulatorError("Not connected -- call connect() first.")

        out_of_range = {
            joint: value
            for joint, value in position.as_dict().items()
            if not (self._config.position_min <= value <= self._config.position_max)
        }
        if out_of_range:
            raise ManipulatorError(
                f"Target position(s) outside [{self._config.position_min}, "
                f"{self._config.position_max}]: {out_of_range}."
            )

        self._backend.write_positions(position.as_dict())
        # No real motion-complete feedback exists in this repo yet (see
        # the module docstring) -- move_timeout_s is a fixed settle
        # delay, not a confirmed-arrival wait.
        time.sleep(0 if self._config.simulate else self._config.move_timeout_s)

    def open_gripper(self, position: int = 4095) -> None:
        """Move only the gripper joint to `position` (default: fully open)."""
        current = self.get_positions()
        current["gripper"] = position
        self.move_to(JointPosition.from_dict(current))

    def close_gripper(self, position: int = 0) -> None:
        """Move only the gripper joint to `position` (default: fully closed)."""
        current = self.get_positions()
        current["gripper"] = position
        self.move_to(JointPosition.from_dict(current))

    def home(self) -> None:
        """Move to `config.home_position`."""
        self.move_to(self._config.home_position)

    def get_positions(self) -> dict[str, int]:
        """Current raw positions for all six joints.

        Raises:
            ManipulatorError: If not connected.
        """
        if not self._connected:
            raise ManipulatorError("Not connected -- call connect() first.")
        return self._backend.read_positions()

    def get_status(self) -> dict[str, object]:
        """Connectivity/health summary -- never raises, same contract as the other bridges'."""
        base = {"enabled": self._config.enabled}
        if self._config.simulate:
            return {"mode": "simulate", "connected": self._connected, **base}
        try:
            positions = self.get_positions() if self._connected else None
        except Exception as exc:  # noqa: BLE001 -- connectivity check, never raises
            return {"mode": "real", "connected": False, "error": str(exc), **base}
        return {
            "mode": "real",
            "connected": self._connected,
            "port": self._config.port,
            "positions": positions,
            **base,
        }

    def disconnect(self) -> None:
        """Disable torque and disconnect. Safe to call more than once."""
        if not self._connected:
            return
        try:
            self._backend.torque_enable(False)
        finally:
            self._backend.disconnect()
            self._connected = False

    def __enter__(self) -> Manipulator:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()
