"""FastAPI + WebSocket chat backend (week 11, day 4).

Wires day 2's `LLMParser` and day 3's `ActionExecutor` (now backed by
day 4's SQLite `assistant.database.Database` instead of day 3's
in-memory `DetectionLog`) behind one `WS /ws/chat` endpoint an operator's
browser talks to, plus a minimal HTML/JS chat page served at `GET /`
(`assistant/static/index.html`).

Same structured-logging convention as `src/api/app.py` (week 08): one
JSON line per HTTP request via `configure_logging`, reused rather than a
second copy of that formatter.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from assistant.action_executor import ActionExecutor
from assistant.database import Database
from assistant.llm_parser import LLMParser
from src.api.logging_config import configure_logging

configure_logging(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO))
logger = logging.getLogger("assistant_chat")

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DB_PATH = "outputs/assistant_demo/detections.db"


def _resolve_db_path() -> str:
    """`$ASSISTANT_DB_PATH` if set, else `DEFAULT_DB_PATH`.

    Read lazily (inside `lifespan`, not at import time) so tests can
    `monkeypatch.setenv` before a `TestClient` triggers startup and get
    an isolated database file per test, without this module needing a
    reload.
    """
    return os.environ.get("ASSISTANT_DB_PATH", DEFAULT_DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the real SQLite `Database`, `ActionExecutor`, `LLMParser` for the process lifetime.

    Torn down in reverse order on shutdown -- `LLMParser.shutdown()`
    stops its `ExternalCallQueue` worker thread, `ActionExecutor.shutdown()`
    closes the ROS2/OPC-UA bridges' connections (real or simulated),
    `Database.close()` closes the SQLite connection. Skipping this would
    leak a real background thread per test/run.
    """
    db = Database(_resolve_db_path())
    executor = ActionExecutor(detection_log=db)
    parser = LLMParser()
    app.state.db = db
    app.state.executor = executor
    app.state.parser = parser
    try:
        yield
    finally:
        parser.shutdown()
        executor.shutdown()
        db.close()


app = FastAPI(title="VLM Assistant Chat", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One structured JSON log line per HTTP request -- same shape as `src/api/app.py`."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe -- same shape as `src/api/app.py::health`."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """The minimal chat page. A static file, not a template engine -- one page, no dynamic data."""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _run_command(app: FastAPI, text: str) -> dict[str, object]:
    """Parse + execute one chat command. Shared by the WebSocket handler (real trigger)."""
    parser: LLMParser = app.state.parser
    executor: ActionExecutor = app.state.executor
    parsed = parser.parse(text)
    outcome = executor.execute(parsed)
    return {
        "text": text,
        "intent": parsed.intent,
        "source": parsed.source,
        "success": outcome.success,
        "message": outcome.message,
        "data": outcome.data,
    }


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """One command in, one JSON result out, repeated for the life of the connection.

    `_run_command` is real, blocking work: `LLMParser.parse()` can wait
    up to `max_wait_seconds` (default 5s, day 2) for a rate-limited
    external call, and `ActionExecutor.execute()` can make a real HTTP
    call to the vision service. Running it inline in this `async def`
    would stall uvicorn's single event loop for every OTHER connection
    for up to that long -- `run_in_threadpool` is the same mechanism
    FastAPI already uses for a plain sync `def` route, applied explicitly
    here since a WebSocket handler must be `async def`.
    """
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            result = await run_in_threadpool(_run_command, websocket.app, text)
            await websocket.send_text(json.dumps(result, ensure_ascii=False))
    except WebSocketDisconnect:
        logger.info("chat websocket disconnected")
