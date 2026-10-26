"""Unit + integration tests for `assistant.llm_parser`.

`match_locally` is exercised for real -- plain regex, no mocking needed.
`call_claude_api` is exercised against a real `httpx.MockTransport`-backed
client (real HTTP-layer code -- request construction, headers, tool-use
response parsing, schema validation -- against a fake network), same
pattern as `src/synthetic/diffusion_pipeline.py`'s `StabilityAPIBackend`
tests (week 05). `LLMParser` end to end uses that same injected client, so
the full local-miss -> queue -> rate-limit -> API -> validate path runs
for real, just against a fake transport instead of the real Claude API
(no key, no network access to it, and not authorized to spend a real API
budget from this sandbox).
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from assistant.llm_parser import (
    LLMParser,
    ParseStats,
    call_claude_api,
    match_locally,
)
from assistant.utils.config import ClaudeAPIConfig, ParserConfig, RateLimitConfig


def _config(
    max_calls_per_window: int = 10,
    window_seconds: float = 60.0,
    max_queue_size: int = 5,
    max_wait_seconds: float = 2.0,
    api_key_env: str = "TEST_LLM_PARSER_API_KEY",
) -> ParserConfig:
    return ParserConfig(
        rate_limit=RateLimitConfig(
            max_calls_per_window=max_calls_per_window,
            window_seconds=window_seconds,
            max_queue_size=max_queue_size,
            max_wait_seconds=max_wait_seconds,
        ),
        claude_api=ClaudeAPIConfig(
            model="claude-sonnet-5", api_key_env=api_key_env, timeout_s=5.0
        ),
    )


# -- match_locally (tier 1) ------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_intent, expected_params",
    [
        (
            "покажи последние 50 дефектов crack с conf>0.9",
            "list_defects",
            {"defect_type": "crack", "min_confidence": 0.9, "limit": 50},
        ),
        (
            "list defects with confidence > 0.8",
            "list_defects",
            {"defect_type": None, "min_confidence": 0.8, "limit": 20},
        ),
        (
            "покажи 10 дефектов pit",
            "list_defects",
            {"defect_type": "pit", "min_confidence": None, "limit": 10},
        ),
        ("статус", "get_status", {}),
        ("what's the status", "get_status", {}),
        ("перезапусти обучение", "restart_training", {}),
        ("restart training please", "restart_training", {}),
        ("помощь", "help", {}),
        ("help", "help", {}),
    ],
)
def test_match_locally_recognizes_common_commands(
    text: str, expected_intent: str, expected_params: dict
) -> None:
    result = match_locally(text)
    assert result is not None
    assert result.intent == expected_intent
    assert result.parameters == expected_params
    assert result.source == "local"
    assert result.confidence == 1.0
    assert result.is_actionable


def test_match_locally_returns_none_for_free_form_text() -> None:
    assert match_locally("почему стало больше брака на третьей смене?") is None


def test_match_locally_limit_pattern_does_not_pick_up_confidence_digits() -> None:
    # Regression: an earlier version's limit regex matched ANY bare digit
    # anywhere in the text, including the leading "0" of "0.8" -- this
    # command has no real limit at all.
    result = match_locally("покажи дефекты с conf>0.8")
    assert result is not None
    assert result.parameters["limit"] == 20  # schema default, not 0
    assert result.parameters["min_confidence"] == 0.8


def test_match_locally_restart_training_checked_before_list_defects() -> None:
    # "перезапусти обучение" doesn't contain "дефект"/"defect" so this
    # isn't really an ordering hazard, but confirms restart_training wins
    # outright rather than falling through.
    result = match_locally("перезапусти обучение по дефектам")
    assert result is not None
    assert result.intent == "restart_training"


@pytest.mark.parametrize(
    "text",
    [
        "что делать при обрыве камеры?",
        "why did the quality gate trigger?",
        "диск переполнен, что делать",
        "ошибка TensorRT после обновления JetPack",
        "перегрев, троттлинг на Jetson",
    ],
)
def test_match_locally_recognizes_runbook_topics(text: str) -> None:
    result = match_locally(text)
    assert result is not None
    assert result.intent == "ask_runbook"
    assert result.parameters == {"question": text}


def test_match_locally_ask_runbook_checked_after_more_specific_intents() -> None:
    # "покажи дефекты" alone doesn't mention a runbook topic, so this is
    # list_defects, not ask_runbook -- confirms the loose trigger doesn't
    # shadow a real list_defects command.
    result = match_locally("покажи дефекты")
    assert result is not None
    assert result.intent == "list_defects"


# -- call_claude_api (tier 2) -----------------------------------------------------


def _tool_use_response(intent: str, parameters: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "content": [
                {
                    "type": "tool_use",
                    "name": "emit_parsed_command",
                    "input": {"intent": intent, "parameters": parameters},
                }
            ]
        },
    )


def test_call_claude_api_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_LLM_PARSER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_LLM_PARSER_API_KEY"):
        call_claude_api("что делать", _config())


def test_call_claude_api_success_against_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        assert request.headers["x-api-key"] == "fake-key"
        assert request.headers["anthropic-version"]
        body = json.loads(request.content)
        assert body["tool_choice"] == {"type": "tool", "name": "emit_parsed_command"}
        return _tool_use_response("get_status", {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = call_claude_api("как дела на линии", _config(), client=client)

    assert len(captured_requests) == 1
    assert result.intent == "get_status"
    assert result.parameters == {}
    assert result.source == "api"


def test_call_claude_api_extracts_and_validates_list_defects_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return _tool_use_response(
            "list_defects", {"defect_type": "crack", "min_confidence": 0.95, "limit": 30}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = call_claude_api("что там с трещинами", _config(), client=client)

    assert result.parameters == {"defect_type": "crack", "min_confidence": 0.95, "limit": 30}


def test_call_claude_api_unknown_intent_raises_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return _tool_use_response("delete_database", {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(KeyError):
        call_claude_api("do something dangerous", _config(), client=client)


def test_call_claude_api_invalid_params_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return _tool_use_response("list_defects", {"limit": -5})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        call_claude_api("покажи -5 дефектов", _config(), client=client)


def test_call_claude_api_raises_on_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        call_claude_api("x", _config(), client=client)


def test_call_claude_api_missing_tool_use_block_raises_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "no tool use"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(KeyError):
        call_claude_api("x", _config(), client=client)


# -- ParseStats ---------------------------------------------------------------


def test_parse_stats_summary_computes_fractions() -> None:
    stats = ParseStats()
    for _ in range(3):
        stats.record("local")
    stats.record("api")
    summary = stats.summary()
    assert summary["total"] == 4
    assert summary["local_count"] == 3
    assert summary["local_fraction"] == pytest.approx(0.75)
    assert summary["api_count"] == 1
    assert summary["api_fraction"] == pytest.approx(0.25)


def test_parse_stats_summary_zero_total_has_zero_fractions() -> None:
    assert ParseStats().summary()["local_fraction"] == 0.0


# -- LLMParser end to end --------------------------------------------------------


def test_llm_parser_local_hit_never_touches_the_queue() -> None:
    with LLMParser(config=_config()) as parser:
        result = parser.parse("статус")
        assert result.intent == "get_status"
        assert result.source == "local"
        assert parser.stats.local_count == 1
        assert parser.stats.api_count == 0


def test_llm_parser_falls_through_to_api_on_local_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return _tool_use_response("help", {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with LLMParser(config=_config(), api_client=client) as parser:
        result = parser.parse("что вообще ты умеешь делать длинной фразой", max_wait_s=2.0)
        assert result.intent == "help"
        assert result.source == "api"
        assert result.is_actionable
        assert parser.stats.api_count == 1
        assert parser.stats.local_count == 0


def test_llm_parser_reports_error_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_LLM_PARSER_API_KEY", raising=False)
    with LLMParser(config=_config()) as parser:
        result = parser.parse("свободная фраза без ключа", max_wait_s=2.0)
        assert result.intent == "error"
        assert not result.is_actionable
        assert result.error is not None
        assert parser.stats.error_count == 1


def test_llm_parser_returns_pending_when_wait_budget_too_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        release.wait(timeout=5.0)
        return _tool_use_response("help", {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with LLMParser(config=_config(), api_client=client) as parser:
        result = parser.parse("свободная фраза, ответ придёт нескоро", max_wait_s=0.05)
        assert result.intent == "pending"
        assert not result.is_actionable
        assert parser.stats.pending_count == 1
        release.set()


def test_llm_parser_queue_full_reports_unavailable_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_PARSER_API_KEY", "fake-key")
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        release.wait(timeout=5.0)
        return _tool_use_response("help", {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = _config(max_queue_size=1)
    with LLMParser(config=config, api_client=client) as parser:
        # First call occupies the one worker slot (blocked in the handler).
        first_thread_started = threading.Event()

        def call_first() -> None:
            first_thread_started.set()
            parser.parse("первая свободная фраза, надолго", max_wait_s=0.01)

        t = threading.Thread(target=call_first)
        t.start()
        first_thread_started.wait()
        time.sleep(0.05)  # let the worker actually dequeue the first item

        # Second call fills the 1-slot backlog.
        second_result_holder: list = []

        def call_second() -> None:
            second_result_holder.append(
                parser.parse("вторая свободная фраза, тоже надолго", max_wait_s=0.01)
            )

        t2 = threading.Thread(target=call_second)
        t2.start()
        t2.join(timeout=2.0)
        time.sleep(0.05)

        # Third call should be rejected immediately -- queue is full.
        start = time.monotonic()
        third = parser.parse("третья фраза должна быть отклонена немедленно")
        elapsed = time.monotonic() - start

        release.set()
        t.join(timeout=2.0)

        assert third.intent == "unavailable"
        assert not third.is_actionable
        assert elapsed < 0.5  # rejected fast, not after waiting on the network
        assert parser.stats.rejected_count == 1


def test_llm_parser_save_stats_writes_valid_json(tmp_path) -> None:
    with LLMParser(config=_config()) as parser:
        parser.parse("статус")
        stats_path = tmp_path / "stats.json"
        parser.save_stats(stats_path)
        summary = json.loads(stats_path.read_text())
        assert summary["local_count"] == 1


def test_llm_parser_log_summary_returns_the_same_shape_as_stats_summary() -> None:
    with LLMParser(config=_config()) as parser:
        parser.parse("статус")
        assert parser.log_summary() == parser.stats.summary()


def test_llm_parser_shutdown_stops_the_background_thread() -> None:
    parser = LLMParser(config=_config())
    parser.shutdown()
    assert not parser._queue.is_running  # noqa: SLF001 -- verifying real cleanup, not the public API
