"""action_executor.py -- dispatches a validated `ParsedCommand` to its domain (week 11, day 3).

One handler per `assistant.schemas.KNOWN_INTENTS` entry, each calling
into exactly the bridge that owns its domain:

- `list_defects` -> a `DetectionLogProtocol` implementation: day 3's
  in-memory `DetectionLog` by default, or day 4's SQLite-backed
  `assistant.database.Database` when the caller wires one in (see
  `assistant/chat_app.py`).
- `get_status` -> all three bridges' `get_status()`, combined into one report.
- `restart_training` -> `ros_interface.trigger_training_restart` (no live
  trainer exists in this repo to actually restart -- see that method's
  own docstring).
- `help` -> purely local, no bridge.

`ActionExecutor.execute` re-validates `command.parameters` against
`assistant.schemas.INTENT_PARAM_SCHEMAS` before dispatch (the same
validation `llm_parser.py`'s two tiers already ran) -- defense in depth
for a caller that constructs a `ParsedCommand` by hand rather than
through `LLMParser.parse()`, not because a well-formed command should
ever fail this twice.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from assistant.opc_ua_bridge import OPCUABridge
from assistant.ros_interface import ROSInterface
from assistant.schemas import DEFECT_TYPES, KNOWN_INTENTS, ParsedCommand, validate_parameters
from assistant.vision_bridge import VisionBridge

HELP_TEXT: dict[str, str] = {
    "list_defects": (
        "Показать последние обнаруженные дефекты. Параметры: defect_type "
        f"({'/'.join(DEFECT_TYPES)}), min_confidence (0-1), limit (1-500)."
    ),
    "get_status": "Показать состояние: сервис инференса, ROS2, OPC-UA/PLC.",
    "restart_training": "Отправить команду перезапуска обучения.",
    "help": "Показать список доступных команд.",
    "ask_runbook": (
        "Спросить про регламент/runbook (например, 'что делать при обрыве камеры'). "
        "Параметры: question."
    ),
}
"""Human-readable descriptions -- `help`'s own output, and a real (not
placeholder) reason `action_executor.py` needs `assistant.schemas.KNOWN_INTENTS`
rather than a hardcoded list of its own."""


@dataclass(frozen=True)
class DetectionRecord:
    """One logged detection -- what `list_defects` filters/sorts/limits over.

    Attributes:
        defect_type: One of `assistant.schemas.DEFECT_TYPES`.
        confidence: In `[0, 1]`.
        detected_at: Unix timestamp (seconds).
        source: Free-text provenance (e.g. `"vision_bridge.detect_objects"`,
            `"demo"`), for audit -- not interpreted by `DetectionLog`.
    """

    defect_type: str
    confidence: float
    detected_at: float
    source: str


class DetectionLog:
    """In-memory detection history -- day 3's honest stand-in, kept as the tests' default.

    Real, tested filtering/sorting/limiting logic operating on whatever's
    been `record()`ed since the process started, not a claim that it
    persists across restarts. Day 4 added `assistant.database.Database`,
    a real SQLite-backed store behind this exact same `record()`/`query()`/
    `__len__()` shape (`DetectionLogProtocol`, below) -- `assistant/chat_app.py`
    wires that one in instead; this in-memory version stays as the
    zero-setup default for `ActionExecutor()`/tests that don't need
    persistence.
    """

    def __init__(self) -> None:
        self._records: list[DetectionRecord] = []

    def record(self, record: DetectionRecord) -> None:
        """Append one detection to the log."""
        self._records.append(record)

    def query(
        self,
        defect_type: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
    ) -> list[DetectionRecord]:
        """Most-recent-first, filtered by `defect_type`/`min_confidence`, capped at `limit`.

        Args:
            defect_type: Only records matching this type, or all types if `None`.
            min_confidence: Only records at or above this confidence, or
                no floor if `None`.
            limit: Maximum records to return.

        Returns:
            Matching records, newest (`detected_at`) first.
        """
        matches = [
            r
            for r in self._records
            if (defect_type is None or r.defect_type == defect_type)
            and (min_confidence is None or r.confidence >= min_confidence)
        ]
        matches.sort(key=lambda r: r.detected_at, reverse=True)
        return matches[:limit]

    def __len__(self) -> int:
        return len(self._records)


class DetectionLogProtocol(Protocol):
    """Structural shape `ActionExecutor.detection_log` actually needs.

    Day 3's in-memory `DetectionLog` (above) and day 4's SQLite-backed
    `assistant.database.Database` both satisfy this without either module
    importing the other -- `database.py` imports `DetectionRecord` from
    this module already; importing `Database` back here for a type hint
    would make that circular for no benefit, since `ActionExecutor` only
    ever calls these three methods on whatever it's given.
    """

    def record(self, record: DetectionRecord) -> None: ...

    def query(
        self,
        defect_type: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
    ) -> list[DetectionRecord]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of dispatching one `ParsedCommand` -- what a chat UI (day 4) renders.

    Attributes:
        intent: Echoes `ParsedCommand.intent`.
        success: `False` for anything from "not actionable" through a
            handler-raised exception -- `message` explains why either way.
        message: Human-readable summary, safe to show an operator directly.
        data: Structured payload (e.g. `list_defects`' matching records,
            `get_status`'s per-bridge report) -- `{}` when there's nothing
            beyond `message`.
    """

    intent: str
    success: bool
    message: str
    data: dict[str, object] = field(default_factory=dict)


_HandlerFn = Callable[[dict[str, object]], ExecutionResult]


def _default_rag_answer_fn(question: str) -> object:
    """Lazy default `rag_answer_fn` -- imports `rag` only on first actual call.

    A plain function (not a bound import at module level) so that
    importing `action_executor.py` itself never requires `rag/`'s own
    dependencies (`numpy`, eventually `src.models.clip`'s `torch`/
    `transformers`) to be installed -- same lazy-import convention this
    whole repository already follows for heavy dependencies.
    """
    from rag.integration import answer_question_with_default_index

    return answer_question_with_default_index(question)


class ActionExecutor:
    """Dispatches a `ParsedCommand` to its domain handler.

    One of `vision_bridge`/`ros_interface`/`opc_ua_bridge`, depending on intent.
    """

    def __init__(
        self,
        vision: VisionBridge | None = None,
        ros: ROSInterface | None = None,
        opc_ua: OPCUABridge | None = None,
        detection_log: DetectionLogProtocol | None = None,
        rag_answer_fn: Callable[[str], object] | None = None,
    ) -> None:
        """
        Args:
            vision: Defaults to `VisionBridge()`.
            ros: Defaults to `ROSInterface()`.
            opc_ua: Defaults to `OPCUABridge()`.
            detection_log: Defaults to a fresh, empty in-memory
                `DetectionLog`. Pass an `assistant.database.Database` for
                persistent, SQLite-backed history instead (day 4) -- both
                satisfy `DetectionLogProtocol`.
            rag_answer_fn: `question: str -> rag.answer.RAGAnswer` (week
                19). Defaults to a LAZY wrapper around
                `rag.integration.answer_question_with_default_index` --
                `rag/`'s real CLIP-backed index is only built on the
                FIRST actual `ask_runbook` call, never at
                `ActionExecutor` construction time (same lazy-load
                convention every `VLMLoader` in this repository already
                follows -- constructing this object must never force a
                network call before one is actually needed). Tests
                inject a fake instead, same injectable-call pattern as
                `vision`/`ros`/`opc_ua` above.
        """
        self.vision = vision if vision is not None else VisionBridge()
        self.ros = ros if ros is not None else ROSInterface()
        self.opc_ua = opc_ua if opc_ua is not None else OPCUABridge()
        self.detection_log = detection_log if detection_log is not None else DetectionLog()
        self.rag_answer_fn = rag_answer_fn if rag_answer_fn is not None else _default_rag_answer_fn
        self._handlers: dict[str, _HandlerFn] = {
            "list_defects": self._handle_list_defects,
            "get_status": self._handle_get_status,
            "restart_training": self._handle_restart_training,
            "help": self._handle_help,
            "ask_runbook": self._handle_ask_runbook,
        }

    def record_detection(
        self, defect_type: str, confidence: float, source: str = "vision_bridge"
    ) -> None:
        """Append one detection to `self.detection_log`, timestamped now.

        The real integration point for `vision_bridge.detect_objects`'
        results (mapping its open-vocabulary `Detection.label` to a
        canonical `assistant.schemas.DEFECT_TYPES` name is caller policy,
        not this method's concern) -- kept separate from `execute()`
        since recording a detection and executing a chat command are two
        independent things that happen to share this log.

        Args:
            defect_type: One of `assistant.schemas.DEFECT_TYPES`.
            confidence: In `[0, 1]`.
            source: Free-text provenance, for audit.
        """
        self.detection_log.record(
            DetectionRecord(
                defect_type=defect_type,
                confidence=confidence,
                detected_at=time.time(),
                source=source,
            )
        )

    def execute(self, command: ParsedCommand) -> ExecutionResult:
        """Validate and dispatch `command` to its handler.

        Args:
            command: Typically `LLMParser.parse()`'s output.

        Returns:
            An `ExecutionResult`. Never raises -- a handler exception, an
            unknown intent, or a validation failure all become
            `success=False` results, not propagated exceptions (a chat
            request should always get an answer, never a 500).
        """
        if not command.is_actionable:
            return ExecutionResult(
                intent=command.intent,
                success=False,
                message=command.error or f"Command not actionable (intent={command.intent!r}).",
            )

        try:
            validated_params = validate_parameters(command.intent, command.parameters).model_dump()
        except KeyError:
            return ExecutionResult(
                intent=command.intent,
                success=False,
                message=f"Unknown intent: {command.intent!r} (not in {KNOWN_INTENTS}).",
            )
        except ValidationError as exc:
            return ExecutionResult(
                intent=command.intent, success=False, message=f"Invalid parameters: {exc}"
            )

        # command.intent is in KNOWN_INTENTS (checked above), so a handler is registered.
        handler = self._handlers[command.intent]
        try:
            return handler(validated_params)
        except Exception as exc:  # noqa: BLE001 -- a handler failure must produce a result,
            return ExecutionResult(  # never propagate and abort the whole chat turn
                intent=command.intent, success=False, message=f"Execution failed: {exc}"
            )

    def _handle_list_defects(self, params: dict[str, object]) -> ExecutionResult:
        records = self.detection_log.query(
            defect_type=params.get("defect_type"),
            min_confidence=params.get("min_confidence"),
            limit=params["limit"],
        )
        return ExecutionResult(
            intent="list_defects",
            success=True,
            message=(
                f"Found {len(records)} matching defect(s) "
                f"(log has {len(self.detection_log)} total)."
            ),
            data={"records": [r.__dict__ for r in records]},
        )

    def _handle_get_status(self, params: dict[str, object]) -> ExecutionResult:
        vision_status = self.vision.health_check()
        ros_status = self.ros.get_status()
        opc_ua_status = self.opc_ua.get_status()
        all_connected = (
            vision_status.get("reachable", False)
            and ros_status.get("connected", False)
            and opc_ua_status.get("connected", False)
        )
        message = "All systems reachable." if all_connected else "One or more systems unreachable."
        return ExecutionResult(
            intent="get_status",
            success=True,
            message=message,
            data={"vision": vision_status, "ros": ros_status, "opc_ua": opc_ua_status},
        )

    def _handle_restart_training(self, params: dict[str, object]) -> ExecutionResult:
        message = self.ros.trigger_training_restart(reason="operator chat command")
        return ExecutionResult(
            intent="restart_training",
            success=True,
            message=f"Training-restart command published on {message.topic!r}.",
            data={"topic": message.topic, "payload": message.data},
        )

    def _handle_help(self, params: dict[str, object]) -> ExecutionResult:
        return ExecutionResult(
            intent="help",
            success=True,
            message="Доступные команды: " + ", ".join(KNOWN_INTENTS),
            data={"commands": dict(HELP_TEXT)},
        )

    def _handle_ask_runbook(self, params: dict[str, object]) -> ExecutionResult:
        """Answer a free-text question from the real `docs/runbook.md`/`docs/architecture.md`.

        `success=True` regardless of whether an answer was actually
        grounded -- "the regulation doesn't cover this" (see
        `rag.answer.RAGAnswer.is_grounded`) is a legitimate, successful
        response to a question, not a failed command execution. The
        distinction a caller needs (chat UI, audit log) is carried in
        `data["is_grounded"]`/`data["sources"]`, not in `success`.
        """
        answer = self.rag_answer_fn(str(params["question"]))
        return ExecutionResult(
            intent="ask_runbook",
            success=True,
            message=answer.text,
            data={"sources": answer.sources, "is_grounded": answer.is_grounded},
        )

    def shutdown(self) -> None:
        """Tear down the ROS2/OPC-UA bridges' real connections, if any were opened."""
        self.ros.shutdown()
        self.opc_ua.disconnect()

    def __enter__(self) -> ActionExecutor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
