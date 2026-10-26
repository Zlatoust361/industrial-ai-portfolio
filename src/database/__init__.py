"""Database layer -- the capstone's single entry point for detection history (week 12).

Composition, not new code: a re-export of week 11 day 4's already-built,
already-tested SQLite-backed detection log, so `scenarios/` (day 3) logs
what it sorts/flags through the same store the chat backend already
uses, instead of a second, scenario-private one.

- `Database` (week 11 day 4): real, file-backed, thread-safe SQLite
  implementation of `DetectionLogProtocol`.
- `DetectionRecord`/`DetectionLog`/`DetectionLogProtocol` (week 11 days
  3-4): the shared record shape and structural interface both `Database`
  and the in-memory `DetectionLog` satisfy -- re-exported from
  `action_executor.py`, where they were first defined, not duplicated
  here.
"""

from __future__ import annotations

from assistant.action_executor import DetectionLog, DetectionLogProtocol, DetectionRecord
from assistant.database import DEFAULT_DB_PATH, Database
from assistant.database_models import CREATE_DETECTIONS_TABLE_SQL, DETECTIONS_TABLE

__all__ = [
    "Database",
    "DEFAULT_DB_PATH",
    "DetectionRecord",
    "DetectionLog",
    "DetectionLogProtocol",
    "DETECTIONS_TABLE",
    "CREATE_DETECTIONS_TABLE_SQL",
]
