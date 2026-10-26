"""Structured (JSON) logging for the FastAPI service (week 08, day 2).

Plain stdlib `logging` plus a JSON `Formatter` -- one formatter is enough
for "one JSON object per log line to stdout" (the shape a container
log-collector expects); this repo defaults to no dependency without a
genuine reason, and a full structured-logging library isn't one here.
"""

from __future__ import annotations

import json
import logging
import sys

_RESERVED_LOG_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


class JSONFormatter(logging.Formatter):
    """Renders each log record as one JSON object per line.

    Any field passed via `logger.info(..., extra={...})` is merged into
    the output -- this is what request-scoped fields (`request_id`,
    `method`, `duration_ms`, ...) ride on, without a custom `LogRecord`
    subclass.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render `record` as a single-line JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON object (as text): `timestamp`/`level`/`logger`/`message`
            plus every `extra` field, and `exception` if one was attached.
        """
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in vars(record).items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter on the root logger's stdout handler.

    Idempotent: replaces any handlers this already installed instead of
    stacking duplicates, so re-importing/re-configuring (e.g. across
    tests, or an app factory called more than once) doesn't multiply
    log lines.

    Args:
        level: Root logger level.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h for h in root.handlers if not isinstance(h, _JSONStreamHandler)]
    handler = _JSONStreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)


class _JSONStreamHandler(logging.StreamHandler):
    """Marker subclass so `configure_logging` can find and replace its own handler."""
