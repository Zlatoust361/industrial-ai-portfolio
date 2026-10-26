"""SQLite schema for `assistant/`'s persistent detection history (week 11, day 4).

Plain hand-written DDL executed through stdlib `sqlite3`, not an ORM
(SQLAlchemy) -- one small table, the exact same fields day 3's in-memory
`assistant.action_executor.DetectionRecord` already has
(`defect_type`/`confidence`/`detected_at`/`source`), doesn't earn a new
heavy dependency on a memory-constrained Jetson (same reasoning ADR-029
used to skip `prometheus_client` for one small metrics endpoint). See
`assistant/database.py` for the class that actually opens the connection
and runs these statements.
"""

from __future__ import annotations

DETECTIONS_TABLE = "detections"
"""Table name -- referenced by both the DDL below and `database.py`'s queries."""

CREATE_DETECTIONS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DETECTIONS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    defect_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    detected_at REAL NOT NULL,
    source TEXT NOT NULL
)
"""

CREATE_DETECTIONS_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{DETECTIONS_TABLE}_detected_at
ON {DETECTIONS_TABLE} (detected_at DESC)
"""
"""`list_defects` always orders by `detected_at DESC` (see `database.py::Database.query`)
-- an index on exactly that column/direction is what actually helps once
the table has grown past a trivial size."""
