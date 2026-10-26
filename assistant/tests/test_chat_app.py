"""Tests for `assistant.chat_app` -- FastAPI + WebSocket chat backend (week 11, day 4).

Real end-to-end: `TestClient` drives the actual `lifespan` (real
`Database`/`ActionExecutor`/`LLMParser` construction and teardown, real
background threads started/stopped), and `websocket_connect` is a real
WebSocket handshake + message round trip against the real `/ws/chat`
handler, the same pattern `test_vision_bridge.py`'s
`test_health_check_against_the_real_deployed_app` established for
`src/api/app.py`. No mocking of `chat_app.py`'s own code -- the only
thing not exercised for real is `vision_bridge`'s HTTP call (nothing is
listening on its configured `base_url` in this sandbox, exactly like
`test_action_executor.py::test_get_status_reports_unreachable_vision_service_without_failing`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistant.chat_app import app
from assistant.database import Database


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(tmp_path / "detections.db"))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ws/chat" in response.text


def test_lifespan_opens_a_real_sqlite_database_at_the_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "detections.db"
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(db_path))

    with TestClient(app):
        assert db_path.exists()


def test_ws_chat_help_command_round_trips(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("помощь")
        result = json.loads(websocket.receive_text())

    assert result["intent"] == "help"
    assert result["success"] is True
    assert "commands" in result["data"]


def test_ws_chat_list_defects_reads_from_the_real_sqlite_backed_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "detections.db"
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(db_path))

    with TestClient(app) as test_client:
        test_client.app.state.executor.record_detection("crack", 0.95, source="test")

        with test_client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("покажи последние 10 дефектов crack")
            result = json.loads(websocket.receive_text())

    assert result["intent"] == "list_defects"
    assert result["success"] is True
    assert len(result["data"]["records"]) == 1

    # And it's real SQLite, not swept away with the process -- reopening
    # the same file directly (no chat_app involved) sees the same row.
    with Database(db_path) as reopened:
        assert len(reopened) == 1


def test_ws_chat_handles_multiple_messages_on_one_connection(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("статус")
        first = json.loads(websocket.receive_text())
        websocket.send_text("помощь")
        second = json.loads(websocket.receive_text())

    assert first["intent"] == "get_status"
    assert second["intent"] == "help"


def test_ws_chat_unrecognized_free_form_text_falls_through_without_crashing(
    client: TestClient,
) -> None:
    # No ANTHROPIC_API_KEY in this sandbox (see assistant/README.md's own
    # honesty note) -- tier 2 correctly fails with a clear error rather
    # than hanging or the connection dying.
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("какая сегодня погода на заводе")
        result = json.loads(websocket.receive_text())

    assert result["success"] is False
