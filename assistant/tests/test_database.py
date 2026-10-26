"""Tests for `assistant.database.Database`.

Same filtering/sorting/limit/len contract as
`tests/test_action_executor.py`'s `DetectionLog` tests -- this is meant
to be a drop-in replacement behind `DetectionLogProtocol`, so it's
exercised against the exact same cases, plus persistence-specific ones
(`DetectionLog` can't have those at all).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.action_executor import ActionExecutor, DetectionRecord
from assistant.database import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    with Database(tmp_path / "detections.db") as database:
        yield database


def test_query_filters_by_type_and_confidence(db: Database) -> None:
    db.record(DetectionRecord("crack", 0.95, detected_at=1.0, source="x"))
    db.record(DetectionRecord("scratch", 0.6, detected_at=2.0, source="x"))
    db.record(DetectionRecord("crack", 0.5, detected_at=3.0, source="x"))

    results = db.query(defect_type="crack", min_confidence=0.9)

    assert len(results) == 1
    assert results[0].confidence == 0.95


def test_query_sorted_newest_first(db: Database) -> None:
    db.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))
    db.record(DetectionRecord("crack", 0.9, detected_at=3.0, source="x"))
    db.record(DetectionRecord("crack", 0.9, detected_at=2.0, source="x"))

    results = db.query()

    assert [r.detected_at for r in results] == [3.0, 2.0, 1.0]


def test_query_respects_limit(db: Database) -> None:
    for i in range(10):
        db.record(DetectionRecord("crack", 0.9, detected_at=float(i), source="x"))

    assert len(db.query(limit=3)) == 3


def test_query_no_filters_returns_everything_up_to_limit(db: Database) -> None:
    db.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))
    db.record(DetectionRecord("pit", 0.1, detected_at=2.0, source="x"))

    assert len(db.query(limit=20)) == 2


def test_len(db: Database) -> None:
    assert len(db) == 0
    db.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))
    assert len(db) == 1


# -- persistence: the actual reason this class exists over DetectionLog ---------


def test_records_survive_a_reconnect_to_the_same_file(tmp_path: Path) -> None:
    db_path = tmp_path / "detections.db"

    with Database(db_path) as first_connection:
        first_connection.record(DetectionRecord("crack", 0.9, detected_at=1.0, source="x"))

    with Database(db_path) as second_connection:
        results = second_connection.query()

    assert len(results) == 1
    assert results[0].defect_type == "crack"


def test_creates_parent_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "detections.db"

    with Database(db_path):
        pass

    assert db_path.exists()


def test_close_is_safe_to_call_twice(tmp_path: Path) -> None:
    db = Database(tmp_path / "detections.db")
    db.close()
    db.close()  # must not raise


# -- drop-in compatibility with ActionExecutor -----------------------------------


def test_action_executor_accepts_a_database_as_its_detection_log(tmp_path: Path) -> None:
    db = Database(tmp_path / "detections.db")
    with ActionExecutor(detection_log=db) as executor:
        executor.record_detection("crack", 0.95, source="demo")
        assert len(executor.detection_log) == 1
    db.close()
