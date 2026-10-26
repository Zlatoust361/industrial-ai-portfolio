"""Latency/throughput benchmarking for `VLMInference` image encoding."""

from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image

from src.models.base import VLMInference


@dataclass(frozen=True)
class LatencyStats:
    """Latency/throughput statistics for one batch size.

    Attributes:
        batch_size: Batch size these stats were measured at.
        mean_ms: Mean wall-clock latency per batch, in milliseconds.
        p50_ms: Median latency per batch, in milliseconds.
        p95_ms: 95th-percentile latency per batch, in milliseconds.
        throughput_ips: Images processed per second (`batch_size / mean latency`).
    """

    batch_size: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    throughput_ips: float


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Args:
        sorted_values: Values sorted in ascending order.
        pct: Percentile to compute, in `[0, 100]`.

    Returns:
        The value at the given percentile, or `0.0` for an empty input.
    """
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[index]


def benchmark_image_encoder(
    inference: VLMInference,
    sample_image: Image.Image,
    batch_sizes: list[int],
    warmup_iters: int,
    num_iters: int,
) -> list[LatencyStats]:
    """Measure image-encoding latency/throughput across batch sizes.

    Args:
        inference: A `VLMInference` implementation to benchmark.
        sample_image: One representative image, replicated to fill each batch.
        batch_sizes: Batch sizes to sweep.
        warmup_iters: Untimed iterations run before measurement, per batch size.
        num_iters: Timed iterations per batch size.

    Returns:
        One `LatencyStats` entry per batch size, in the order given.
    """
    results: list[LatencyStats] = []
    for batch_size in batch_sizes:
        batch = [sample_image] * batch_size

        for _ in range(warmup_iters):
            inference.encode_image(batch)

        durations_ms: list[float] = []
        for _ in range(num_iters):
            start = time.perf_counter()
            inference.encode_image(batch)
            durations_ms.append((time.perf_counter() - start) * 1000)
        durations_ms.sort()

        mean_ms = sum(durations_ms) / len(durations_ms)
        throughput = batch_size / (mean_ms / 1000) if mean_ms > 0 else 0.0
        results.append(
            LatencyStats(
                batch_size=batch_size,
                mean_ms=mean_ms,
                p50_ms=percentile(durations_ms, 50),
                p95_ms=percentile(durations_ms, 95),
                throughput_ips=throughput,
            )
        )
    return results
