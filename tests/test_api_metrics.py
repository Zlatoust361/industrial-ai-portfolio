"""Unit tests for `src.api.metrics` -- MetricsCollector, gpu_memory_mb,
TegrastatsSampler, render_prometheus.

`gpu_memory_mb`/`TegrastatsSampler.available` are exercised for real (not
mocked) against this sandbox's actual state: no CUDA, no `tegrastats`
binary -- both correctly report "unavailable" rather than raising, the
same boundary every other hardware-dependent piece of this repo hits.
"""

from __future__ import annotations

from src.api.metrics import (
    MetricsCollector,
    TegrastatsSampler,
    gpu_memory_mb,
    render_prometheus,
)


def test_metrics_collector_computes_percentiles_and_mean() -> None:
    collector = MetricsCollector(window_size=100)
    for duration_ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
        collector.record("/health", duration_ms, 200)

    snapshot = collector.snapshot()
    stats = snapshot.endpoints["/health"]
    assert stats.count == 5
    assert stats.mean_ms == 30.0
    assert stats.p50_ms == 30.0
    assert stats.p95_ms == 50.0


def test_metrics_collector_tracks_all_endpoints_aggregate() -> None:
    collector = MetricsCollector(window_size=100)
    collector.record("/v1/embed/image", 10.0, 200)
    collector.record("/v1/embed/text", 20.0, 200)

    snapshot = collector.snapshot()
    assert snapshot.endpoints["_all"].count == 2
    assert snapshot.endpoints["/v1/embed/image"].count == 1
    assert snapshot.endpoints["/v1/embed/text"].count == 1


def test_metrics_collector_count_is_unwindowed_but_percentiles_are_windowed() -> None:
    collector = MetricsCollector(window_size=3)
    for duration_ms in [1.0, 2.0, 3.0, 4.0, 5.0]:
        collector.record("/health", duration_ms, 200)

    stats = collector.snapshot().endpoints["/health"]
    assert stats.count == 5  # unwindowed: every record() call counted
    assert stats.p50_ms == 4.0  # windowed: only the last 3 samples (3, 4, 5)


def test_metrics_collector_status_counts() -> None:
    collector = MetricsCollector()
    collector.record("/v1/embed/image", 10.0, 200)
    collector.record("/v1/embed/image", 10.0, 400)
    collector.record("/v1/embed/image", 10.0, 200)

    status_counts = collector.snapshot().status_counts
    assert status_counts["/v1/embed/image:200"] == 2
    assert status_counts["/v1/embed/image:400"] == 1


def test_metrics_collector_throughput_is_count_over_uptime() -> None:
    collector = MetricsCollector()
    collector.record("/health", 5.0, 200)
    snapshot = collector.snapshot()
    assert snapshot.uptime_seconds > 0
    assert snapshot.endpoints["/health"].throughput_rps == (1 / snapshot.uptime_seconds)


def test_gpu_memory_mb_is_none_without_cuda() -> None:
    # This sandbox has no torch install and no CUDA -- real verification,
    # not a mock, of the "returns None cleanly" contract.
    assert gpu_memory_mb() is None


def test_tegrastats_sampler_unavailable_in_this_sandbox() -> None:
    # No `tegrastats` binary here either -- same real-boundary check.
    sampler = TegrastatsSampler()
    assert sampler.available is False


def test_tegrastats_sampler_start_stop_are_safe_noops_when_unavailable() -> None:
    sampler = TegrastatsSampler()
    sampler.start()
    sampler.start()  # idempotent
    sampler.stop()
    sampler.stop()  # idempotent, even though start() never actually spawned anything


def test_tegrastats_sampler_tracks_latest_and_peak_temps() -> None:
    sampler = TegrastatsSampler(throttle_temp_c=50.0)
    sampler._ingest_line("RAM 1000/4000MB (lfb 1x1) GPU@40.0C CPU@45.0C")
    sampler._ingest_line("RAM 1100/4000MB (lfb 1x1) GPU@55.0C CPU@42.0C")

    snapshot = sampler.snapshot()
    assert snapshot.latest_temps_c == {"GPU": 55.0, "CPU": 42.0}
    assert snapshot.max_temps_c == {"GPU": 55.0, "CPU": 45.0}


def test_tegrastats_sampler_thermal_warning_triggers_at_threshold() -> None:
    sampler = TegrastatsSampler(throttle_temp_c=50.0)
    sampler._ingest_line("GPU@49.0C")
    assert sampler.snapshot().thermal_warning is False

    sampler._ingest_line("GPU@50.0C")
    assert sampler.snapshot().thermal_warning is True


def test_tegrastats_sampler_ignores_lines_without_sensor_readings() -> None:
    sampler = TegrastatsSampler()
    sampler._ingest_line("this line has no sensor readings in it")
    assert sampler.snapshot().latest_temps_c == {}


def test_render_prometheus_includes_metric_names_and_help_type() -> None:
    collector = MetricsCollector()
    collector.record("/v1/embed/image", 12.0, 200)
    tegrastats_sampler = TegrastatsSampler(throttle_temp_c=50.0)
    tegrastats_sampler._ingest_line("GPU@40.0C")

    body = render_prometheus(
        collector.snapshot(), tegrastats_sampler.snapshot(), gpu_memory=None, process_rss_mb=123.4
    )

    assert "# HELP vlm_api_requests_total" in body
    assert "# TYPE vlm_api_requests_total counter" in body
    assert 'vlm_api_requests_total{endpoint="/v1/embed/image"} 1' in body
    assert 'vlm_jetson_temperature_celsius{sensor="GPU"} 40.0' in body
    assert "vlm_jetson_thermal_warning 0" in body
    assert "vlm_process_rss_mb 123.4" in body
    # gpu_memory=None -> the GPU metrics are omitted entirely, not emitted with a null value.
    assert "vlm_gpu_memory_allocated_mb" not in body


def test_render_prometheus_includes_status_code_breakdown() -> None:
    collector = MetricsCollector()
    collector.record("/health", 5.0, 200)
    collector.record("/health", 5.0, 200)
    collector.record("/v1/embed/image", 12.0, 400)

    body = render_prometheus(
        collector.snapshot(), TegrastatsSampler().snapshot(), gpu_memory=None, process_rss_mb=0.0
    )

    assert "# HELP vlm_api_requests_by_status_total" in body
    assert "# TYPE vlm_api_requests_by_status_total counter" in body
    assert 'vlm_api_requests_by_status_total{endpoint="/health",status_code="200"} 2' in body
    assert (
        'vlm_api_requests_by_status_total{endpoint="/v1/embed/image",status_code="400"} 1'
        in body
    )


def test_render_prometheus_includes_gpu_memory_when_available() -> None:
    collector = MetricsCollector()
    tegrastats_sampler = TegrastatsSampler()

    body = render_prometheus(
        collector.snapshot(),
        tegrastats_sampler.snapshot(),
        gpu_memory={"allocated_mb": 512.0, "reserved_mb": 1024.0},
        process_rss_mb=1.0,
    )

    assert "vlm_gpu_memory_allocated_mb 512.0" in body
    assert "vlm_gpu_memory_reserved_mb 1024.0" in body


def test_render_prometheus_escapes_label_values() -> None:
    collector = MetricsCollector()
    collector.record('/weird"path', 1.0, 200)

    body = render_prometheus(
        collector.snapshot(), TegrastatsSampler().snapshot(), gpu_memory=None, process_rss_mb=0.0
    )

    assert 'endpoint="/weird\\"path"' in body
