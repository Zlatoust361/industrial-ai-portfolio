"""Simple in-process metrics for the FastAPI service (week 08, day 4).

Four pieces, matching the day's ask:

- `MetricsCollector` -- per-endpoint request latency (p50/p95/mean) and
  throughput, fed by `app.py`'s existing request-logging middleware.
- `gpu_memory_mb` -- `torch.cuda` allocated/reserved memory, when CUDA is
  available (it isn't in this sandbox -- returns `None` cleanly, the same
  boundary every other `torch`-dependent piece of this repo hits).
- `TegrastatsSampler` -- a background `tegrastats` reader for temperature,
  kept running for the process's lifetime (unlike
  `scripts/jetson_benchmark.py`'s `TegrastatsMonitor`, which only runs
  around one timed measurement block) and reusing
  `src.evaluation.jetson.parse_tegrastats_temps` rather than
  re-implementing the sensor regex a third time.
- `render_prometheus` -- hand-rolled Prometheus text-exposition format.
  No `prometheus_client` dependency: a "simple metrics system" for one
  small service doesn't need a client library, and hand-rolling keeps the
  exact same numbers driving both `/metrics` and `/metrics.json` (see
  `app.py`) instead of two independently-computed views.

Not a real Prometheus histogram: p50/p95 here are computed once per
`/metrics` scrape from a bounded rolling window (`window_size`, default
1000 samples), then exposed as plain gauges -- not raw bucket counters
for `histogram_quantile()` to compute server-side. That's the standard
Prometheus-native approach and is more correct under scraping/aggregation
across replicas, but is meaningfully more code for a single-instance
Jetson deployment; documented here as a deliberate simplification, not an
oversight.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from src.evaluation.benchmark import percentile
from src.evaluation.jetson import parse_tegrastats_temps

THROTTLE_TEMP_C = 85.0
"""Conservative default thermal-warning threshold, in Celsius.

Not read from the device's actual DVFS/thermal trip points (`cat
/sys/devices/virtual/thermal/thermal_zone*/trip_point_*_temp` on the
device itself would give the real, board-specific numbers) -- this repo
has no physical Jetson Orin Nano to read those from, or to confirm this
threshold against a real throttling event. Treat `thermal_warning=True`
(see `TegrastatsSnapshot`) as "worth checking `tegrastats`/`jetson_clocks`
directly", not as an authoritative throttle signal. Override via
`TegrastatsSampler(throttle_temp_c=...)` once a real device's trip points
are known.
"""

_ALL_ENDPOINTS_KEY = "_all"
"""Synthetic endpoint key aggregating every request, alongside per-path stats."""


@dataclass(frozen=True)
class EndpointStats:
    """Latency/throughput stats for one endpoint (or `_ALL_ENDPOINTS_KEY`).

    Attributes:
        count: Total requests recorded for this endpoint since process
            start -- NOT windowed, unlike the percentiles below.
        mean_ms: Mean request duration over the last `window_size` samples.
        p50_ms: Median request duration over the last `window_size` samples.
        p95_ms: 95th-percentile request duration over the last `window_size` samples.
        throughput_rps: `count / uptime_seconds` -- average since process start.
    """

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    throughput_rps: float


@dataclass(frozen=True)
class MetricsSnapshot:
    """A point-in-time read of `MetricsCollector`'s state.

    Attributes:
        uptime_seconds: Seconds since the `MetricsCollector` was constructed.
        endpoints: Per-endpoint stats, plus `_ALL_ENDPOINTS_KEY` for the
            aggregate across every endpoint.
        status_counts: `"{endpoint}:{status_code}"` -> count, since process start.
    """

    uptime_seconds: float
    endpoints: dict[str, EndpointStats]
    status_counts: dict[str, int]


class MetricsCollector:
    """Thread-safe request latency/throughput tracker.

    `record()` is called once per request from `app.py`'s middleware;
    `snapshot()` is called once per `/metrics`/`/metrics.json` scrape.
    """

    def __init__(self, window_size: int = 1000) -> None:
        """Args:
            window_size: Max samples kept per endpoint for percentile
                computation. Bounds memory for a long-running deployment;
                `count`/`throughput_rps` are unaffected (tracked separately,
                unwindowed).
        """
        self._window_size = window_size
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._durations_ms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._counts: dict[str, int] = defaultdict(int)
        self._status_counts: dict[str, int] = defaultdict(int)

    def record(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        """Record one completed request.

        Args:
            endpoint: Request path (e.g. `request.url.path` -- excludes
                the query string, so `?max_frames=...` doesn't fragment
                the per-endpoint stats).
            duration_ms: Wall-clock request duration, in milliseconds.
            status_code: HTTP status code the response was sent with.
        """
        with self._lock:
            for key in (endpoint, _ALL_ENDPOINTS_KEY):
                self._durations_ms[key].append(duration_ms)
                self._counts[key] += 1
            self._status_counts[f"{endpoint}:{status_code}"] += 1

    def snapshot(self) -> MetricsSnapshot:
        """Compute a `MetricsSnapshot` from the current state.

        Returns:
            Per-endpoint latency percentiles/throughput and status-code counts.
        """
        uptime_seconds = max(time.monotonic() - self._start_time, 1e-9)
        with self._lock:
            endpoints = {
                endpoint: _summarize(durations, self._counts[endpoint], uptime_seconds)
                for endpoint, durations in self._durations_ms.items()
            }
            status_counts = dict(self._status_counts)
        return MetricsSnapshot(
            uptime_seconds=uptime_seconds, endpoints=endpoints, status_counts=status_counts
        )


def _summarize(durations: deque[float], count: int, uptime_seconds: float) -> EndpointStats:
    """Build one `EndpointStats` from a window of durations plus the unwindowed total count."""
    sorted_durations = sorted(durations)
    mean_ms = sum(sorted_durations) / len(sorted_durations) if sorted_durations else 0.0
    return EndpointStats(
        count=count,
        mean_ms=mean_ms,
        p50_ms=percentile(sorted_durations, 50),
        p95_ms=percentile(sorted_durations, 95),
        throughput_rps=count / uptime_seconds,
    )


def gpu_memory_mb() -> dict[str, float] | None:
    """Current process CUDA memory usage, in MiB.

    Jetson's unified memory means this overlaps with `tegrastats`' system
    RAM figure rather than being a separate physical pool -- reported
    anyway since `torch.cuda.memory_allocated`/`memory_reserved` is what
    most GPU dashboards expect, and it still answers "how much of
    `model_config.jetson.yaml`'s memory budget is this process's CUDA
    tensors specifically" (see `src/models/memory.py`).

    Returns:
        `{"allocated_mb": ..., "reserved_mb": ...}`, or `None` if `torch`
        isn't installed or CUDA isn't available (e.g. this sandbox, or a
        CPU dev deployment) -- the same boundary every other `torch`-
        dependent piece of this repo hits.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
    }


@dataclass(frozen=True)
class TegrastatsSnapshot:
    """A point-in-time read of `TegrastatsSampler`'s state.

    Attributes:
        available: Whether the `tegrastats` binary was found on this host.
        latest_temps_c: Most recent reading per sensor, in Celsius.
        max_temps_c: Peak reading per sensor observed since the sampler started.
        thermal_warning: `True` if any sensor's latest reading is at or
            above `throttle_temp_c` -- see `THROTTLE_TEMP_C`'s docstring
            for why this is a configurable proxy, not a hardware signal.
        throttle_temp_c: The threshold `thermal_warning` was computed against.
    """

    available: bool
    latest_temps_c: dict[str, float]
    max_temps_c: dict[str, float]
    thermal_warning: bool
    throttle_temp_c: float


class TegrastatsSampler:
    """Runs `tegrastats` in the background for the life of the process.

    Unlike `scripts/jetson_benchmark.py`'s `TegrastatsMonitor` (starts/stops
    around one timed measurement block and keeps every raw line), this is
    started once at app startup and keeps only the latest + peak reading
    per sensor -- unbounded raw-line accumulation would leak memory over a
    long-running deployment, which is exactly the "long run at MAXN Super"
    scenario this day is about.
    """

    def __init__(self, interval_ms: int = 1000, throttle_temp_c: float = THROTTLE_TEMP_C) -> None:
        """Args:
            interval_ms: Sampling interval passed to `tegrastats --interval`.
            throttle_temp_c: Threshold for `TegrastatsSnapshot.thermal_warning`.
        """
        self._interval_ms = interval_ms
        self._throttle_temp_c = throttle_temp_c
        self._available = shutil.which("tegrastats") is not None
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_temps_c: dict[str, float] = {}
        self._max_temps_c: dict[str, float] = {}

    @property
    def available(self) -> bool:
        """Whether the `tegrastats` binary was found on this host (e.g. `False` off-Jetson)."""
        return self._available

    def start(self) -> None:
        """Start the background `tegrastats` process and reader thread.

        A no-op if `tegrastats` isn't available on this host, or if
        already started -- safe to call unconditionally from `app.py`'s
        `lifespan`.
        """
        if not self._available or self._process is not None:
            return
        self._process = subprocess.Popen(
            ["tegrastats", "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            text=True,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _ingest_line(self, line: str) -> None:
        """Parse one `tegrastats` line and update the latest/peak temperature state."""
        temps = parse_tegrastats_temps([line])
        if not temps:
            return
        with self._lock:
            for sensor, values in temps.items():
                value = values[-1]
                self._latest_temps_c[sensor] = value
                self._max_temps_c[sensor] = max(self._max_temps_c.get(sensor, value), value)

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self._ingest_line(line)

    def stop(self) -> None:
        """Terminate the background process and join the reader thread.

        A no-op if `start()` was never called (or `tegrastats` wasn't available).
        """
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._process = None
        self._thread = None

    def snapshot(self) -> TegrastatsSnapshot:
        """Read the current latest/peak temperatures and compute `thermal_warning`.

        Returns:
            A `TegrastatsSnapshot`. `available=False` (with empty temp
            dicts) if `tegrastats` isn't installed on this host.
        """
        with self._lock:
            latest = dict(self._latest_temps_c)
            peak = dict(self._max_temps_c)
        thermal_warning = any(value >= self._throttle_temp_c for value in latest.values())
        return TegrastatsSnapshot(
            available=self._available,
            latest_temps_c=latest,
            max_temps_c=peak,
            thermal_warning=thermal_warning,
            throttle_temp_c=self._throttle_temp_c,
        )


def _escape_label_value(value: str) -> str:
    """Escape a string for use as a Prometheus label value (backslash/quote/newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_value(value: float | int | bool) -> str:
    """Render a metric value the way Prometheus text format expects."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(round(value, 4))


def _render_metric(
    lines: list[str],
    name: str,
    help_text: str,
    metric_type: str,
    samples: list[tuple[dict, object]],
) -> None:
    """Append one metric's `# HELP`/`# TYPE` header and its samples to `lines`."""
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")
    for labels, value in samples:
        label_str = ",".join(f'{k}="{_escape_label_value(str(v))}"' for k, v in labels.items())
        suffix = f"{{{label_str}}}" if label_str else ""
        lines.append(f"{name}{suffix} {_format_value(value)}")


def render_prometheus(
    metrics: MetricsSnapshot,
    tegrastats: TegrastatsSnapshot,
    gpu_memory: dict[str, float] | None,
    process_rss_mb: float,
) -> str:
    """Render a full Prometheus text-exposition-format scrape body.

    Args:
        metrics: From `MetricsCollector.snapshot()`.
        tegrastats: From `TegrastatsSampler.snapshot()`.
        gpu_memory: From `gpu_memory_mb()`; omitted entirely from the
            output when `None` (CUDA unavailable), rather than emitting a
            metric with no value.
        process_rss_mb: From `src.models.memory.process_resident_memory_mb()`.

    Returns:
        The full scrape body, newline-terminated.
    """
    lines: list[str] = []
    endpoint_items = sorted(metrics.endpoints.items())

    _render_metric(
        lines,
        "vlm_api_requests_total",
        "Total requests served since process start, by endpoint.",
        "counter",
        [({"endpoint": ep}, stats.count) for ep, stats in endpoint_items],
    )
    _render_metric(
        lines,
        "vlm_api_request_duration_mean_ms",
        "Mean request latency in ms, over the last window_size requests.",
        "gauge",
        [({"endpoint": ep}, stats.mean_ms) for ep, stats in endpoint_items],
    )
    _render_metric(
        lines,
        "vlm_api_request_duration_p50_ms",
        "Median request latency in ms, over the last window_size requests.",
        "gauge",
        [({"endpoint": ep}, stats.p50_ms) for ep, stats in endpoint_items],
    )
    _render_metric(
        lines,
        "vlm_api_request_duration_p95_ms",
        "95th-percentile request latency in ms, over the last window_size requests.",
        "gauge",
        [({"endpoint": ep}, stats.p95_ms) for ep, stats in endpoint_items],
    )
    _render_metric(
        lines,
        "vlm_api_throughput_rps",
        "Requests per second, averaged since process start.",
        "gauge",
        [({"endpoint": ep}, stats.throughput_rps) for ep, stats in endpoint_items],
    )
    _render_metric(
        lines,
        "vlm_api_requests_by_status_total",
        "Total requests since process start, by endpoint and HTTP status code -- "
        "the breakdown vlm_api_requests_total doesn't have (week 10 day 2: needed "
        "for an error-rate dashboard panel, e.g. "
        'sum(rate(vlm_api_requests_by_status_total{status_code=~"5.."}[5m])) / '
        "sum(rate(vlm_api_requests_by_status_total[5m]))).",
        "counter",
        [
            ({"endpoint": endpoint, "status_code": status_code}, count)
            for (endpoint, status_code), count in sorted(
                (key.rsplit(":", 1), count) for key, count in metrics.status_counts.items()
            )
        ],
    )
    _render_metric(
        lines,
        "vlm_process_uptime_seconds",
        "Seconds since this process's MetricsCollector was created.",
        "gauge",
        [({}, metrics.uptime_seconds)],
    )
    _render_metric(
        lines,
        "vlm_process_rss_mb",
        "This process's RSS, in MiB (see src/models/memory.py's memory-budget guard).",
        "gauge",
        [({}, process_rss_mb)],
    )
    if gpu_memory is not None:
        _render_metric(
            lines,
            "vlm_gpu_memory_allocated_mb",
            "Current torch.cuda memory allocated by this process, in MiB.",
            "gauge",
            [({}, gpu_memory["allocated_mb"])],
        )
        _render_metric(
            lines,
            "vlm_gpu_memory_reserved_mb",
            "Current torch.cuda memory reserved by this process, in MiB.",
            "gauge",
            [({}, gpu_memory["reserved_mb"])],
        )
    _render_metric(
        lines,
        "vlm_jetson_tegrastats_available",
        "1 if the tegrastats binary was found on this host, else 0.",
        "gauge",
        [({}, tegrastats.available)],
    )
    _render_metric(
        lines,
        "vlm_jetson_temperature_celsius",
        "Latest tegrastats sensor temperature reading.",
        "gauge",
        [
            ({"sensor": sensor}, value)
            for sensor, value in sorted(tegrastats.latest_temps_c.items())
        ],
    )
    _render_metric(
        lines,
        "vlm_jetson_temperature_max_celsius",
        "Peak tegrastats sensor temperature observed since process start.",
        "gauge",
        [({"sensor": sensor}, value) for sensor, value in sorted(tegrastats.max_temps_c.items())],
    )
    _render_metric(
        lines,
        "vlm_jetson_thermal_warning",
        f"1 if any tegrastats sensor is at/above throttle_temp_c={tegrastats.throttle_temp_c}, "
        "else 0 -- a configurable proxy threshold, not a real DVFS throttle signal "
        "(see THROTTLE_TEMP_C's docstring in src/api/metrics.py).",
        "gauge",
        [({}, tegrastats.thermal_warning)],
    )
    return "\n".join(lines) + "\n"
