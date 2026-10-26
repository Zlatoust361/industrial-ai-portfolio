"""SQLite-backed detection history (week 11, day 4).

Swaps in for day 3's in-memory `assistant.action_executor.DetectionLog`
behind the exact same `record()`/`query()`/`__len__()` shape
(`assistant.action_executor.DetectionLogProtocol`) -- `ActionExecutor`
itself needs no changes to use this instead, only what it's constructed
with. Real, file-backed persistence across restarts, the thing day 3's
`DetectionLog` docstring explicitly said it wasn't.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from assistant.action_executor import DetectionRecord
from assistant.database_models import (
    CREATE_DETECTIONS_INDEX_SQL,
    CREATE_DETECTIONS_TABLE_SQL,
    DETECTIONS_TABLE,
)

DEFAULT_DB_PATH = "outputs/assistant_demo/detections.db"


class Database:
    """One SQLite connection, real file-backed detection history.

    `check_same_thread=False` plus one `threading.Lock` around every
    statement: uvicorn's default single-process/asyncio worker still runs
    sync route bodies and WebSocket handlers in a thread pool (Starlette's
    `run_in_threadpool`), so more than one request can reach this object
    from different OS threads concurrently -- a single connection with an
    explicit lock is simpler and cheap enough at this write volume (one
    INSERT per detection, occasional SELECTs) than a per-request
    connection or a real connection pool would be for one small table on
    one device.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """
        Args:
            db_path: Where the SQLite file lives. Parent directories are
                created if missing. `":memory:"` is accepted for tests
                that want persistence semantics without a real file.
        """
        self.db_path = db_path if str(db_path) == ":memory:" else Path(db_path)
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute(CREATE_DETECTIONS_TABLE_SQL)
            self._conn.execute(CREATE_DETECTIONS_INDEX_SQL)
            self._conn.commit()

    def record(self, record: DetectionRecord) -> None:
        """Insert one detection. Matches `DetectionLog.record`'s signature exactly."""
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {DETECTIONS_TABLE} "  # noqa: S608 -- DETECTIONS_TABLE is a module constant, not input
                "(defect_type, confidence, detected_at, source) VALUES (?, ?, ?, ?)",
                (record.defect_type, record.confidence, record.detected_at, record.source),
            )
            self._conn.commit()

    def query(
        self,
        defect_type: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
    ) -> list[DetectionRecord]:
        """Most-recent-first, filtered by `defect_type`/`min_confidence`, capped at `limit`.

        Same filtering/sorting/limit semantics as `DetectionLog.query` --
        real SQL `WHERE`/`ORDER BY`/`LIMIT` instead of an in-memory
        Python filter+sort, not a behavioral change.
        """
        clauses = ["1 = 1"]
        params: list[object] = []
        if defect_type is not None:
            clauses.append("defect_type = ?")
            params.append(defect_type)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        params.append(limit)

        sql = (
            f"SELECT defect_type, confidence, detected_at, source FROM {DETECTIONS_TABLE} "  # noqa: S608
            f"WHERE {' AND '.join(clauses)} ORDER BY detected_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            DetectionRecord(
                defect_type=row[0], confidence=row[1], detected_at=row[2], source=row[3]
            )
            for row in rows
        ]

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._conn.execute(f"SELECT COUNT(*) FROM {DETECTIONS_TABLE}").fetchone()  # noqa: S608
        return int(count)

    def close(self) -> None:
        """Close the underlying SQLite connection. Safe to call more than once."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
