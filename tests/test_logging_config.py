"""Unit tests for `src.api.logging_config`'s JSON formatter and setup."""

from __future__ import annotations

import json
import logging

from src.api.logging_config import JSONFormatter, configure_logging


def _make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="vlm_api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_json_with_core_fields() -> None:
    record = _make_record()

    payload = json.loads(JSONFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "vlm_api"
    assert payload["message"] == "request"
    assert "timestamp" in payload


def test_json_formatter_merges_extra_fields() -> None:
    record = _make_record(request_id="abc-123", status_code=200, duration_ms=12.5)

    payload = json.loads(JSONFormatter().format(record))

    assert payload["request_id"] == "abc-123"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 12.5


def test_json_formatter_includes_exception_when_present() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="vlm_api",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(JSONFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    configure_logging()

    root = logging.getLogger()
    json_handlers = [h for h in root.handlers if isinstance(h.formatter, JSONFormatter)]
    assert len(json_handlers) == 1


def test_configure_logging_sets_root_level() -> None:
    configure_logging(logging.WARNING)

    assert logging.getLogger().level == logging.WARNING

    configure_logging(logging.INFO)  # restore default for other tests
