"""Unit tests for `assistant.scripts.measure_memory_footprint`'s pure/local helpers.

The full script spawns two real `uvicorn` subprocesses and measures their
RSS -- verified for real via a direct run
(`PYTHONPATH=. python assistant/scripts/measure_memory_footprint.py`,
see `course/DECISIONS.md`'s week 11 day 4 entry for the real numbers this
produced), not re-run here on every `pytest` invocation: spawning two
full server processes per test run is exactly the kind of slow,
heavy-for-CI cost `run_action_demo.py`/`run_parser_demo.py` (days 2-3)
are also only ever smoke-tested directly, not under pytest. What's
covered here for real: the SQLite seeding helper (real `Database`, real
file) and the readiness-poller's timeout behavior (real socket
connection attempt against a port nothing listens on).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import pytest

from assistant.database import Database
from assistant.scripts.measure_memory_footprint import _rss_mb, _seed_database, _wait_until_ready

UNUSED_PORT = 8399  # nothing listens here in the test environment


def test_seed_database_inserts_the_requested_row_count(tmp_path: Path) -> None:
    db_path = tmp_path / "seed.db"

    _seed_database(db_path, num_rows=25)

    with Database(db_path) as db:
        assert len(db) == 25


def test_seed_database_cycles_through_all_three_defect_types(tmp_path: Path) -> None:
    db_path = tmp_path / "seed.db"

    _seed_database(db_path, num_rows=9)

    with Database(db_path) as db:
        records = db.query(limit=9)
        assert {r.defect_type for r in records} == {"crack", "scratch", "pit"}


def test_rss_mb_returns_a_real_positive_number_for_the_current_process() -> None:
    # The test process itself is a real, already-running PID -- no
    # subprocess needed to exercise psutil.Process(pid).memory_info().rss
    # for real.
    rss = _rss_mb(os.getpid())
    assert rss > 0
    assert rss == pytest.approx(psutil.Process(os.getpid()).memory_info().rss / (1024**2))


def test_wait_until_ready_raises_after_timeout_when_nothing_is_listening() -> None:
    start = time.monotonic()
    with pytest.raises(RuntimeError, match=str(UNUSED_PORT)):
        _wait_until_ready(UNUSED_PORT, timeout_s=0.5)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.5
    assert elapsed < 5.0  # sanity bound -- shouldn't hang far past the requested timeout
