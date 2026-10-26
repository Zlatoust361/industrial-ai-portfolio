"""Tests for `assistant.action_executor`.

`ActionExecutor` is exercised against real `VisionBridge`/`ROSInterface`/
`OPCUABridge` instances (simulate mode where relevant) -- no mocking of
the executor's own dispatch logic, since none of that needs a real
network/ROS2/PLC to test for real.
"""

from __future__ import annotations

import time

import pytest

from assistant.action_executor import (
    ActionExecutor,
    DetectionLog,
    DetectionRecord,
    ExecutionResult,
)
from assistant.opc_ua_bridge import OPCUABridge
from assistant.ros_interface import ROSInterface
from assistant.schemas import KNOWN_INTENTS, ParsedCommand
from assistant.vision_bridge import VisionBridge


def _command(intent: str, parameters: dict | None = None, **overrides: object) -> ParsedCommand:
    defaults = {
        "intent": intent,
        "parameters": parameters or {},
        "source": "local",
        "raw_text": "test",
        "confidence": 1.0,
    }
    defaults.update(overrides)
    return ParsedCommand(**defaults)


@pytest.fixture
def executor():
    with ActionExecutor() as ex:
        yield ex


# -- DetectionLog -----------------------------------------------------------------


def test_detection_log_query_filters_by_type_and_confidence() -> None:
    log = DetectionLog()
    log.record(DetectionRecord("crack", 0.95, detected_at=1.0, source="x"))
    log.record(DetectionRecord("scratch", 0.6, detected_at=2.0, source="x"))
    log.record(DetectionRecord("crack", 0.5, detected_at=3.0, source="x"))

    results = log.query(defect_type="crack", min_confidence=0.9)

    assert len(results) == 1
    assert results[0].confidence == 0.95


def test_detection_log_query_sorted_newest_first() -> None:
    log = DetectionLog()
    log.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))
    log.record(DetectionRecord("crack", 0.9, detected_at=3.0, source="x"))
    log.record(DetectionRecord("crack", 0.9, detected_at=2.0, source="x"))

    results = log.query()

    assert [r.detected_at for r in results] == [3.0, 2.0, 1.0]


def test_detection_log_query_respects_limit() -> None:
    log = DetectionLog()
    for i in range(10):
        log.record(DetectionRecord("crack", 0.9, detected_at=float(i), source="x"))

    assert len(log.query(limit=3)) == 3


def test_detection_log_query_no_filters_returns_everything_up_to_limit() -> None:
    log = DetectionLog()
    log.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))
    log.record(DetectionRecord("pit", 0.1, detected_at=2.0, source="x"))

    assert len(log.query(limit=20)) == 2


def test_detection_log_len() -> None:
    log = DetectionLog()
    assert len(log) == 0
    log.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))
    assert len(log) == 1


# -- ActionExecutor.execute: dispatch/validation/error-handling contract --------


def test_execute_not_actionable_command_returns_failure_without_dispatch() -> None:
    with ActionExecutor() as ex:
        command = _command("pending", error="still processing")
        result = ex.execute(command)

    assert result.success is False
    assert result.message == "still processing"


def test_execute_unknown_intent_is_reported_not_raised() -> None:
    # Bypasses is_actionable's own KNOWN_INTENTS check by constructing a
    # ParsedCommand with error=None but an intent execute() must still
    # independently reject -- defense in depth, not redundant.
    with ActionExecutor() as ex:
        command = ParsedCommand(
            intent="delete_everything",
            parameters={},
            source="local",
            raw_text="x",
            confidence=1.0,
        )
        # is_actionable is False here too (intent not in KNOWN_INTENTS),
        # so this also exercises the not-actionable path -- both guards
        # agree on rejecting it.
        result = ex.execute(command)

    assert result.success is False


def test_execute_invalid_parameters_reported_as_failure() -> None:
    with ActionExecutor() as ex:
        command = _command("list_defects", {"limit": -5})
        result = ex.execute(command)

    assert result.success is False
    assert "Invalid parameters" in result.message


def test_execute_handler_exception_becomes_a_failure_result() -> None:
    with ActionExecutor() as ex:

        def broken_handler(params: dict) -> ExecutionResult:
            raise RuntimeError("boom")

        ex._handlers["help"] = broken_handler  # noqa: SLF001 -- injecting a failure deliberately
        result = ex.execute(_command("help"))

    assert result.success is False
    assert "boom" in result.message


# -- list_defects ------------------------------------------------------------------


def test_list_defects_end_to_end(executor: ActionExecutor) -> None:
    executor.record_detection("crack", 0.95, source="demo")
    executor.record_detection("scratch", 0.5, source="demo")

    result = executor.execute(_command("list_defects", {"defect_type": "crack"}))

    assert result.success is True
    assert len(result.data["records"]) == 1
    assert result.data["records"][0]["defect_type"] == "crack"


def test_list_defects_defaults_apply_when_no_filters_given(executor: ActionExecutor) -> None:
    executor.record_detection("crack", 0.5, source="demo")

    result = executor.execute(_command("list_defects"))

    assert result.success is True
    assert len(result.data["records"]) == 1


def test_record_detection_timestamps_with_real_time(executor: ActionExecutor) -> None:
    before = time.time()
    executor.record_detection("pit", 0.7)
    after = time.time()

    (record,) = executor.detection_log.query(limit=1)
    assert before <= record.detected_at <= after


# -- get_status ---------------------------------------------------------------------


def test_get_status_combines_all_three_bridges(executor: ActionExecutor) -> None:
    result = executor.execute(_command("get_status"))

    assert result.success is True
    assert set(result.data) == {"vision", "ros", "opc_ua"}
    assert result.data["ros"]["mode"] == "simulate"
    assert result.data["opc_ua"]["mode"] == "simulate"


def test_get_status_reports_unreachable_vision_service_without_failing(
    executor: ActionExecutor,
) -> None:
    # No FastAPI service is actually running in this test -- vision_bridge
    # correctly reports unreachable rather than raising.
    result = executor.execute(_command("get_status"))
    assert result.data["vision"]["reachable"] is False
    # The status CHECK succeeded; the target being down is data, not a failure.
    assert result.success is True


# -- restart_training ----------------------------------------------------------------


def test_restart_training_publishes_via_ros_interface(executor: ActionExecutor) -> None:
    result = executor.execute(_command("restart_training"))

    assert result.success is True
    assert result.data["payload"]["command"] == "restart_training"


# -- ask_runbook (week 19) ---------------------------------------------------------------


class _FakeRAGAnswer:
    def __init__(self, text: str, sources: list, is_grounded: bool) -> None:
        self.text = text
        self.sources = sources
        self.is_grounded = is_grounded


def test_ask_runbook_uses_injected_rag_answer_fn() -> None:
    seen_questions = []

    def fake_rag_answer_fn(question: str):
        seen_questions.append(question)
        return _FakeRAGAnswer(
            text="Camera troubleshooting steps.",
            sources=[("docs/runbook.md", "1. Camera / video source loss")],
            is_grounded=True,
        )

    question = "что делать при обрыве камеры?"
    with ActionExecutor(rag_answer_fn=fake_rag_answer_fn) as executor:
        result = executor.execute(_command("ask_runbook", {"question": question}))

    assert result.success is True
    assert result.intent == "ask_runbook"
    assert result.message == "Camera troubleshooting steps."
    assert result.data["is_grounded"] is True
    assert result.data["sources"] == [("docs/runbook.md", "1. Camera / video source loss")]
    assert seen_questions == [question]


def test_ask_runbook_reports_ungrounded_answers_as_success_too() -> None:
    def fake_rag_answer_fn(question: str):
        return _FakeRAGAnswer(text="Не найдено.", sources=[], is_grounded=False)

    with ActionExecutor(rag_answer_fn=fake_rag_answer_fn) as executor:
        result = executor.execute(_command("ask_runbook", {"question": "какая погода?"}))

    # Not finding a relevant section is a legitimate, successful answer --
    # not a failed command execution (see _handle_ask_runbook's docstring).
    assert result.success is True
    assert result.data["is_grounded"] is False


def test_ask_runbook_empty_question_is_rejected_by_the_schema(executor: ActionExecutor) -> None:
    result = executor.execute(_command("ask_runbook", {"question": ""}))
    assert result.success is False
    assert "Invalid parameters" in result.message


# -- help -------------------------------------------------------------------------------


def test_help_lists_all_known_intents(executor: ActionExecutor) -> None:
    result = executor.execute(_command("help"))

    assert result.success is True
    assert set(result.data["commands"]) == set(KNOWN_INTENTS)


# -- full end-to-end wiring: LLMParser -> ActionExecutor -------------------------------


def test_full_pipeline_from_raw_text_to_execution_result() -> None:
    from assistant.llm_parser import LLMParser

    with LLMParser() as parser, ActionExecutor() as executor:
        executor.record_detection("crack", 0.95, source="demo")

        parsed = parser.parse("покажи последние 50 дефектов crack с conf>0.9")
        result = executor.execute(parsed)

    assert parsed.source == "local"
    assert result.intent == "list_defects"
    assert result.success is True
    assert len(result.data["records"]) == 1


def test_action_executor_accepts_injected_bridges() -> None:
    vision = VisionBridge()
    ros = ROSInterface()
    opc_ua = OPCUABridge()
    log = DetectionLog()

    executor = ActionExecutor(vision=vision, ros=ros, opc_ua=opc_ua, detection_log=log)
    assert executor.vision is vision
    assert executor.ros is ros
    assert executor.opc_ua is opc_ua
    assert executor.detection_log is log
    executor.shutdown()
