"""Unit tests for `src.evaluation.benchmark`."""

from __future__ import annotations

from PIL import Image

from src.evaluation.benchmark import benchmark_image_encoder, percentile
from tests.fakes import FakeInference, FakeLoader, sample_memory_section, sample_model_section


def test_percentile_of_sorted_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0


def test_percentile_of_empty_list_is_zero() -> None:
    assert percentile([], 50) == 0.0


def test_benchmark_image_encoder_sweeps_all_batch_sizes() -> None:
    loader = FakeLoader(sample_model_section(), sample_memory_section())
    inference = FakeInference(loader)
    sample_image = Image.new("RGB", (8, 8))

    stats = benchmark_image_encoder(
        inference,
        sample_image,
        batch_sizes=[1, 2],
        warmup_iters=1,
        num_iters=3,
    )

    assert [entry.batch_size for entry in stats] == [1, 2]
    assert all(entry.mean_ms >= 0 for entry in stats)
    assert all(entry.throughput_ips >= 0 for entry in stats)
