"""opc_ua_bridge.py -- OPC-UA/PLC bridge (week 11, day 3).

Deliberately shallow -- this is a stub, not a real OT integration. See
`course/modules/ot-integration-for-cv-engineer/day04-plc-scada-integration-kks.md`,
which explicitly frames week 11's `opc_ua_bridge.py` this way and defers
real KKS-style tag naming, a real-or-simulated PLC, and OPC-UA-vs-Modbus
tradeoffs to that later, dedicated module. What's real here: a correctly
shaped client interface (`connect`/`read_tag`/`write_tag`/`disconnect`)
against real `asyncua` (lazy-imported -- not installed in this sandbox,
and not currently in this repo's real `requirements-jetson.txt`; see
`README.md`'s "Week 11" section), an explicit read-only-by-default safety
posture (writing to a PLC is safety-relevant -- an operator's chat
command must not silently gain write access), and a simulated in-memory
tag store real enough to test the read/write/safety semantics without a
PLC at all.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from assistant.utils.config import OPCUABridgeConfig, load_bridges_config


class OPCUABridgeError(RuntimeError):
    """Raised for connectivity failures and safety-policy violations (e.g. a blocked write)."""


class OPCUAWriteNotAllowedError(OPCUABridgeError):
    """Raised by `write_tag` when `config.read_only` is `True` and no explicit override is given."""


@dataclass(frozen=True)
class TagValue:
    """One tag's current value, as read from (real or simulated) the PLC."""

    tag: str
    value: object
    quality: str = "good"
    """OPC-UA-style quality indicator -- `"good"`/`"bad"`/`"uncertain"`.
    Always `"good"` in simulate mode (nothing to actually degrade)."""


class _SimulatedTagStore:
    """In-memory tag values -- a real, deterministic stand-in for a PLC's address space.

    Not a mock of `asyncua` -- callers exercise real read/write
    semantics (an unknown tag really does raise `KeyError`, a write
    really does update what a subsequent read returns) against this,
    same reasoning as `ros_interface.py`'s `_SimulatedBus`.
    """

    def __init__(self, initial_tags: dict[str, object] | None = None) -> None:
        self._values: dict[str, object] = dict(initial_tags or {})
        self._lock = threading.Lock()

    def read(self, tag: str) -> object:
        with self._lock:
            if tag not in self._values:
                raise KeyError(f"Unknown tag: {tag!r}")
            return self._values[tag]

    def write(self, tag: str, value: object) -> None:
        with self._lock:
            self._values[tag] = value

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._values)


_DEFAULT_SIMULATED_TAGS: dict[str, object] = {
    "Line1.Assistant.Heartbeat": 0,
    "Line1.Assistant.DetectorEnabled": True,
    "Line1.Assistant.LastDefectCount": 0,
}
"""Placeholder tag names, matching `config/bridges_config.yaml`'s
`namespace` prefix loosely -- NOT a real PLC's actual KKS tag scheme, see
this module's own docstring."""


class OPCUABridge:
    """Read (and, if explicitly allowed, write) OPC-UA tags on the line's PLC."""

    def __init__(
        self,
        config: OPCUABridgeConfig | None = None,
        simulated_tags: dict[str, object] | None = None,
    ) -> None:
        """
        Args:
            config: Defaults to `load_bridges_config().opc_ua_bridge`.
            simulated_tags: Initial tag values for simulate mode. Defaults
                to `_DEFAULT_SIMULATED_TAGS`. Ignored in real mode.
        """
        self._config = config if config is not None else load_bridges_config().opc_ua_bridge
        self._store: _SimulatedTagStore | None = (
            _SimulatedTagStore(simulated_tags or _DEFAULT_SIMULATED_TAGS)
            if self._config.simulate
            else None
        )
        self._client: object | None = None

    def _ensure_real_client(self) -> object:
        """Lazily connects (and caches) a real `asyncua` sync client.

        Raises:
            ImportError: If `asyncua` isn't installed (this sandbox --
                and not currently in this repo's real
                `requirements-jetson.txt` either).
            OPCUABridgeError: If connecting to `config.endpoint` fails.
        """
        if self._client is not None:
            return self._client
        from asyncua.sync import Client

        client = Client(url=self._config.endpoint)
        try:
            client.connect()
        except Exception as exc:  # noqa: BLE001 -- real connection failures all become
            raise OPCUABridgeError(f"Could not connect to {self._config.endpoint}: {exc}") from exc
        self._client = client
        return client

    def read_tag(self, tag: str) -> TagValue:
        """Read one tag's current value.

        Args:
            tag: Tag name (simulate mode) or a real node identifier
                under `config.namespace` (real mode).

        Returns:
            The tag's current value.

        Raises:
            OPCUABridgeError: If the tag is unknown (simulate mode) or
                the read fails (real mode).
            ImportError: In real mode, if `asyncua` isn't installed.
        """
        if self._store is not None:
            try:
                value = self._store.read(tag)
            except KeyError as exc:
                raise OPCUABridgeError(str(exc)) from exc
            return TagValue(tag=tag, value=value)

        client = self._ensure_real_client()
        node = client.get_node(f"{self._config.namespace};{tag}")
        try:
            value = node.read_value()
        except Exception as exc:  # noqa: BLE001
            raise OPCUABridgeError(f"Failed to read tag {tag!r}: {exc}") from exc
        return TagValue(tag=tag, value=value)

    def write_tag(self, tag: str, value: object, allow_write: bool = False) -> None:
        """Write one tag's value -- a safety-relevant action, opt-in required.

        Args:
            tag: Tag name (simulate mode) or a real node identifier
                under `config.namespace` (real mode).
            value: New value to write.
            allow_write: Must be `True` AND `config.read_only` must be
                `False` for this to actually write anything -- two
                independent opt-ins (the deployment's own policy, and
                THIS call's explicit intent), neither implied by the
                other. This is deliberately more friction than a single
                flag: an operator's chat command reaching this far should
                never silently gain write access to a PLC.

        Raises:
            OPCUAWriteNotAllowedError: If `config.read_only` is `True`,
                or `allow_write` is `False`.
            OPCUABridgeError: If the write fails (real mode).
            ImportError: In real mode, if `asyncua` isn't installed.
        """
        if self._config.read_only:
            raise OPCUAWriteNotAllowedError(
                "This bridge is configured read_only=true (assistant/config/bridges_config.yaml) "
                "-- writing to a PLC is a safety-relevant action, not enabled by default."
            )
        if not allow_write:
            raise OPCUAWriteNotAllowedError(
                "write_tag requires allow_write=True -- an explicit, per-call opt-in, "
                "even when the bridge itself isn't configured read_only."
            )

        if self._store is not None:
            self._store.write(tag, value)
            return

        client = self._ensure_real_client()
        node = client.get_node(f"{self._config.namespace};{tag}")
        try:
            node.write_value(value)
        except Exception as exc:  # noqa: BLE001
            raise OPCUABridgeError(f"Failed to write tag {tag!r}: {exc}") from exc

    def get_status(self) -> dict[str, object]:
        """Connectivity/health summary -- one input to `action_executor.py`'s `get_status` handler.

        Returns:
            `{"mode": "simulate"|"real", "connected": bool, "read_only": bool, ...}`.
            Never raises -- a real-mode connectivity failure becomes
            `connected=False` with an `error` field, not an exception.
        """
        base = {"read_only": self._config.read_only}
        if self._store is not None:
            return {
                "mode": "simulate",
                "connected": True,
                "endpoint": self._config.endpoint,
                "tag_count": len(self._store.snapshot()),
                **base,
            }
        try:
            self._ensure_real_client()
        except Exception as exc:  # noqa: BLE001 -- connectivity check, never raises
            return {"mode": "real", "connected": False, "error": str(exc), **base}
        return {"mode": "real", "connected": True, "endpoint": self._config.endpoint, **base}

    def disconnect(self) -> None:
        """Disconnect the real `asyncua` client, if one was created. No-op in simulate mode."""
        if self._client is None:
            return
        self._client.disconnect()
        self._client = None

    def __enter__(self) -> OPCUABridge:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()
