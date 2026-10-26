"""Measure the combined memory footprint: detection service + chat backend + SQLite.

Week 11, day 4.

Real subprocess measurements where this sandbox can actually produce
one, a documented configured ceiling where it can't:

- `src/api/app.py` (weeks 08-09's detection/embedding service) is
  started for real as its own `uvicorn` subprocess and its RSS is
  measured for real via `psutil` -- but `CLIPLoader.load()` only loads
  weights on the FIRST `/v1/embed/*` call (lazy, by design, see
  `src/models/clip.py`), and this sandbox has no `torch` to make that
  call succeed at all. So this script's own measurement of that process
  is honestly labeled `detection_service_idle_rss_mb` -- the floor with
  the model NOT loaded, not the number that matters for a real combined
  budget check.
- The number that DOES matter for the budget check is
  `memory.memory_budget_mb` from `config/model_config.jetson.yaml`
  (6144 MB) -- ADR-002 already defines this as the ceiling for "model
  weights + inference runtime working set" specifically so a real
  Jetson deployment fails clean instead of getting OOM-killed
  (`src/models/memory.py::check_process_memory`); reusing it here
  instead of guessing a second number is the same principle applied one
  level up, at the whole-device combined-footprint question this day's
  task actually asks.
- `assistant/chat_app.py` (this day's own deliverable) is started as its
  own real `uvicorn` subprocess too, against a SQLite file pre-seeded
  with a few thousand detection rows (an operationally realistic
  multi-day history, not an empty database) -- no `torch` dependency
  anywhere in this process, so its RSS IS the real number that matters,
  not a floor.

Both subprocesses are started CONCURRENTLY (not measured one after the
other) -- "запусти оба одновременно" is the literal ask, and it's also
the only way `psutil.virtual_memory()`'s system-wide delta means anything.

Usage:
    PYTHONPATH=. python assistant/scripts/measure_memory_footprint.py \\
        [--output outputs/assistant_demo/memory_footprint.json] \\
        [--seed-detections 2000]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from assistant.action_executor import DetectionRecord
from assistant.database import Database
from src.config import load_config

DETECTION_SERVICE_PORT = 8198
CHAT_BACKEND_PORT = 8199
READY_TIMEOUT_S = 30.0
JETSON_CONFIG_PATH = "config/model_config.jetson.yaml"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `measure_memory_footprint.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/assistant_demo/memory_footprint.json")
    parser.add_argument(
        "--seed-detections",
        type=int,
        default=2000,
        help="Rows pre-inserted into the chat backend's SQLite file before measuring.",
    )
    return parser.parse_args()


def _seed_database(db_path: Path, num_rows: int) -> None:
    """Pre-populate `db_path` with `num_rows` synthetic detections (no client data)."""
    with Database(db_path) as db:
        for i in range(num_rows):
            db.record(
                DetectionRecord(
                    defect_type=("crack", "scratch", "pit")[i % 3],
                    confidence=0.5 + (i % 50) / 100,
                    detected_at=float(i),
                    source="memory_footprint_seed",
                )
            )


def _wait_until_ready(port: int, timeout_s: float) -> None:
    """Poll `GET /health` until it responds or `timeout_s` elapses.

    Raises:
        RuntimeError: If the service never became ready in time.
    """
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Service on port {port} did not become ready in {timeout_s}s: {last_error}")


def _start_uvicorn(
    app_import_path: str, port: int, env: dict[str, str] | None = None
) -> subprocess.Popen:
    """Launch `app_import_path` as its own real `uvicorn` subprocess.

    Args:
        app_import_path: e.g. `"src.api.app:app"`.
        port: Port to bind, on `127.0.0.1` only (never exposed off-host).
        env: Extra/overriding environment variables for the subprocess.

    Returns:
        The running `subprocess.Popen` -- caller is responsible for
        terminating it once measurements are done.
    """
    import os

    full_env = {**os.environ, **(env or {})}
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            app_import_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=full_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _rss_mb(pid: int) -> float:
    """This PID's resident set size, in MiB -- same unit `src/models/memory.py` uses."""
    return psutil.Process(pid).memory_info().rss / (1024**2)


def main() -> None:
    """Start both services concurrently, measure real RSS, combine with the configured ceiling."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_path = output_path.parent / "memory_footprint_seed.db"
    db_path.unlink(missing_ok=True)
    print(f"Seeding {args.seed_detections} synthetic detections into {db_path} ...")
    _seed_database(db_path, args.seed_detections)

    system_before_mb = psutil.virtual_memory().used / (1024**2)

    print("Starting src.api.app (detection service) and assistant.chat_app (chat backend) ...")
    detection_proc = _start_uvicorn("src.api.app:app", DETECTION_SERVICE_PORT)
    chat_proc = _start_uvicorn(
        "assistant.chat_app:app",
        CHAT_BACKEND_PORT,
        env={"ASSISTANT_DB_PATH": str(db_path)},
    )

    try:
        _wait_until_ready(DETECTION_SERVICE_PORT, READY_TIMEOUT_S)
        _wait_until_ready(CHAT_BACKEND_PORT, READY_TIMEOUT_S)
        # Let both processes finish their own startup-time allocations
        # (import machinery, FastAPI route registration) before sampling.
        time.sleep(1.0)

        detection_idle_rss_mb = _rss_mb(detection_proc.pid)
        chat_rss_mb = _rss_mb(chat_proc.pid)
        system_after_mb = psutil.virtual_memory().used / (1024**2)
    finally:
        detection_proc.terminate()
        chat_proc.terminate()
        detection_proc.wait(timeout=10)
        chat_proc.wait(timeout=10)

    jetson_config = load_config(JETSON_CONFIG_PATH)
    memory_budget_mb = jetson_config.memory.memory_budget_mb
    device_total_mb = jetson_config.memory.device_total_mb

    modeled_combined_worst_case_mb = memory_budget_mb + chat_rss_mb
    os_headroom_mb = device_total_mb - modeled_combined_worst_case_mb
    # ADR-002's original ~2 GB OS-headroom assumption, before this day's second process.
    baseline_os_headroom_mb = device_total_mb - memory_budget_mb
    within_device_budget = modeled_combined_worst_case_mb <= device_total_mb

    report = {
        "measured": {
            "note": (
                "Real psutil RSS of two concurrently-running uvicorn subprocesses on "
                "this sandbox's CPU/x86_64 host, not a Jetson -- see per-field notes."
            ),
            "detection_service_idle_rss_mb": round(detection_idle_rss_mb, 1),
            "detection_service_idle_note": (
                "FastAPI/uvicorn process up, CLIP weights NOT loaded (lazy-loaded on "
                "first /v1/embed/* call; this sandbox has no torch to make that call "
                "succeed) -- a floor, not the number the budget check below uses."
            ),
            "chat_backend_rss_mb": round(chat_rss_mb, 1),
            "chat_backend_note": (
                f"FastAPI+WebSocket+SQLite ({args.seed_detections} pre-seeded detection "
                "rows), no torch/CLIP dependency at all -- this IS the real number the "
                "budget check below uses."
            ),
            "system_used_memory_delta_mb": round(system_after_mb - system_before_mb, 1),
        },
        "configured": {
            "source": JETSON_CONFIG_PATH,
            "memory_budget_mb": memory_budget_mb,
            "memory_budget_mb_note": (
                "ADR-002's ceiling for the detection service's model weights + "
                "inference runtime working set on a real Jetson -- used here instead "
                "of a guessed post-load number, since this sandbox cannot load the "
                "real model to measure it."
            ),
            "device_total_mb": device_total_mb,
        },
        "combined_budget_check": {
            "modeled_combined_worst_case_mb": round(modeled_combined_worst_case_mb, 1),
            "modeled_combined_worst_case_formula": (
                "memory_budget_mb (detection, configured) + chat_backend_rss_mb (measured)"
            ),
            "device_total_mb": device_total_mb,
            "os_headroom_mb": round(os_headroom_mb, 1),
            "baseline_os_headroom_mb_before_chat_backend": round(baseline_os_headroom_mb, 1),
            "within_device_budget": within_device_budget,
        },
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"detection_service_idle_rss_mb = {detection_idle_rss_mb:.1f}")
    print(f"chat_backend_rss_mb = {chat_rss_mb:.1f}")
    print(f"configured memory_budget_mb (detection ceiling) = {memory_budget_mb}")
    print(f"modeled_combined_worst_case_mb = {modeled_combined_worst_case_mb:.1f}")
    print(f"device_total_mb = {device_total_mb}")
    print(
        f"os_headroom_mb = {os_headroom_mb:.1f} "
        f"(was {baseline_os_headroom_mb:.1f} before the chat backend)"
    )
    print(f"within_device_budget = {within_device_budget}")
    print(f"Report written to '{output_path}'.")


if __name__ == "__main__":
    main()
