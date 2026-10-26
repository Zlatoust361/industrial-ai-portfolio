"""Jetson power-mode benchmark data model and `tegrastats` parsing.

This module is hardware-free and unit-testable: process orchestration that
needs the real device (`sudo nvpmodel -m ...`, the `tegrastats` binary)
lives in `scripts/jetson_benchmark.py`, which depends on this module but
not the other way around.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.evaluation.benchmark import percentile

NVPMODEL_MODES: dict[int, str] = {
    0: "MAXN_SUPER",
    1: "15W",
    2: "7W",
}
"""`nvpmodel -m <id>` -> human-readable power-mode name (Jetson Orin Nano preset table)."""

_TEGRASTATS_SENSOR_PATTERN = re.compile(r"(\w+)@(-?\d+(?:\.\d+)?)C")
_TEGRASTATS_RAM_PATTERN = re.compile(r"RAM (\d+)/(\d+)MB")
_TEGRASTATS_GR3D_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%")
_NVPMODEL_QUERY_NAME_PATTERN = re.compile(r"NV Power Mode:\s*(\S+)")
_NVPMODEL_QUERY_ID_PATTERN = re.compile(r"^\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ModeBenchmarkResult:
    """Latency + thermal result for one `nvpmodel` power mode.

    Attributes:
        mode_id: `nvpmodel -m` argument (0, 1 or 2).
        mode_name: Human-readable mode name, from `NVPMODEL_MODES`.
        iterations: Number of timed inference calls.
        batch_size: Batch size used for each inference call.
        latency_mean_ms: Mean per-call latency, in milliseconds.
        latency_p50_ms: Median per-call latency, in milliseconds.
        latency_p95_ms: 95th-percentile per-call latency, in milliseconds.
        mean_temps_c: Mean temperature per `tegrastats` sensor (e.g. `"GPU"`,
            `"CPU"`, `"tj"`), in Celsius.
        max_temps_c: Peak temperature per `tegrastats` sensor, in Celsius.
    """

    mode_id: int
    mode_name: str
    iterations: int
    batch_size: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    mean_temps_c: dict[str, float]
    max_temps_c: dict[str, float]


def parse_tegrastats_temps(lines: list[str]) -> dict[str, list[float]]:
    """Extract per-sensor temperature readings from raw `tegrastats` output lines.

    `tegrastats` prints entries like `GPU@40.5C`, `CPU@41C`, `tj@41.5C` on
    every line; this pulls all of them out regardless of JetPack version,
    since the exact sensor set varies by board/release.

    Args:
        lines: Raw stdout lines produced by `tegrastats`.

    Returns:
        Mapping of sensor name to the list of readings observed, in Celsius,
        in the order they appeared. Empty if no sensor readings were found.
    """
    readings: dict[str, list[float]] = {}
    for line in lines:
        for sensor, value in _TEGRASTATS_SENSOR_PATTERN.findall(line):
            readings.setdefault(sensor, []).append(float(value))
    return readings


def parse_tegrastats_ram_mb(line: str) -> tuple[int, int] | None:
    """Extract used/total system RAM from one `tegrastats` output line.

    Added for week 10's monitoring exporter (`monitoring/jetson_exporter/`)
    -- reused from the same `RAM (\\d+)/(\\d+)MB` shape
    `ros2_ws/scripts/resource_monitor.py` already parses independently
    (that module duplicates it on purpose, being a separate top-level
    package -- see its own docstring); this is the one hardware-free,
    shared copy every same-repo caller should import instead.

    Args:
        line: A raw `tegrastats` stdout line, e.g. `"RAM 3241/7620MB (lfb ...) ..."`.

    Returns:
        `(used_mb, total_mb)`, or `None` if the line doesn't match the expected pattern.
    """
    match = _TEGRASTATS_RAM_PATTERN.search(line)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_tegrastats_gpu_utilization(line: str) -> float | None:
    """Extract GPU utilization (`GR3D_FREQ`) from one `tegrastats` output line.

    Args:
        line: A raw `tegrastats` stdout line, e.g. `"... GR3D_FREQ 12%@[420] ..."`
            (the `@[freq]`/`@freq` suffix, when present, is ignored -- only
            the percentage before `%` is captured, since its exact shape
            varies by JetPack version).

    Returns:
        GPU utilization as a percentage (`0`-`100`), or `None` if the line
        doesn't match the expected pattern.
    """
    match = _TEGRASTATS_GR3D_PATTERN.search(line)
    return float(match.group(1)) if match else None


def parse_nvpmodel_query(output: str) -> dict[str, str | int | None]:
    """Parse `nvpmodel -q` stdout into a small structured summary.

    Typical `nvpmodel -q` output looks like:

        NV Power Mode: MAXN_SUPER
        0

    Args:
        output: Raw stdout of `nvpmodel -q`.

    Returns:
        `{"mode_name": <str | None>, "mode_id": <int | None>}`. Both are
        `None` if the expected patterns aren't found (unexpected/empty
        `nvpmodel` output).
    """
    name_match = _NVPMODEL_QUERY_NAME_PATTERN.search(output)
    id_match = _NVPMODEL_QUERY_ID_PATTERN.search(output)
    return {
        "mode_name": name_match.group(1) if name_match else None,
        "mode_id": int(id_match.group(1)) if id_match else None,
    }


def summarize_mode_result(
    mode_id: int,
    iterations: int,
    batch_size: int,
    durations_ms: list[float],
    temp_readings: dict[str, list[float]],
) -> ModeBenchmarkResult:
    """Combine timing samples and tegrastats readings into one `ModeBenchmarkResult`.

    Args:
        mode_id: `nvpmodel -m` argument this run used.
        iterations: Number of timed inference calls (typically `len(durations_ms)`).
        batch_size: Batch size used for each inference call.
        durations_ms: Per-call latencies, in milliseconds, any order.
        temp_readings: Output of `parse_tegrastats_temps` for the same time window.

    Returns:
        A `ModeBenchmarkResult` with latency percentiles and per-sensor temperature stats.

    Raises:
        KeyError: If `mode_id` is not one of `NVPMODEL_MODES`.
    """
    sorted_durations = sorted(durations_ms)
    mean_ms = sum(sorted_durations) / len(sorted_durations) if sorted_durations else 0.0

    return ModeBenchmarkResult(
        mode_id=mode_id,
        mode_name=NVPMODEL_MODES[mode_id],
        iterations=iterations,
        batch_size=batch_size,
        latency_mean_ms=mean_ms,
        latency_p50_ms=percentile(sorted_durations, 50),
        latency_p95_ms=percentile(sorted_durations, 95),
        mean_temps_c={
            sensor: sum(values) / len(values) for sensor, values in temp_readings.items()
        },
        max_temps_c={sensor: max(values) for sensor, values in temp_readings.items()},
    )
