"""Tests for `scripts/run_tracking_demo.py` -- the week 13 one-command deliverable."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module

run_tracking_demo = load_script_module("run_tracking_demo.py")


def test_generate_synthetic_stream_is_deterministic_for_a_fixed_seed() -> None:
    frames_a, truth_a = run_tracking_demo.generate_synthetic_stream(
        num_frames=20, noise_rate=0.3, seed=7
    )
    frames_b, truth_b = run_tracking_demo.generate_synthetic_stream(
        num_frames=20, noise_rate=0.3, seed=7
    )
    assert truth_a == truth_b
    assert [[(d.label, d.score, d.bbox) for d in frame] for frame in frames_a] == [
        [(d.label, d.score, d.bbox) for d in frame] for frame in frames_b
    ]


def test_generate_synthetic_stream_different_seeds_diverge() -> None:
    frames_a, _ = run_tracking_demo.generate_synthetic_stream(num_frames=30, noise_rate=0.3, seed=1)
    frames_b, _ = run_tracking_demo.generate_synthetic_stream(num_frames=30, noise_rate=0.3, seed=2)
    labels_a = [d.label for frame in frames_a for d in frame]
    labels_b = [d.label for frame in frames_b for d in frame]
    assert labels_a != labels_b


def test_generate_synthetic_stream_zero_noise_never_flips_label() -> None:
    frames, ground_truth = run_tracking_demo.generate_synthetic_stream(
        num_frames=15, noise_rate=0.0, seed=42
    )
    for frame in frames:
        for detection in frame:
            assert detection.label in ground_truth.values()
    # With no noise, every object's label in every frame matches its own ground truth.
    for frame in frames:
        labels_seen = {d.label for d in frame}
        assert labels_seen == set(ground_truth.values())


def test_run_tracking_demo_produces_one_track_per_object() -> None:
    report = run_tracking_demo.run_tracking_demo(
        num_frames=60, noise_rate=0.3, window_size=5, seed=42
    )
    assert report["num_tracks"] == 2  # DEFAULT_OBJECTS has 2 objects, spatially far apart


def test_run_tracking_demo_smoothing_does_not_increase_total_flips() -> None:
    report = run_tracking_demo.run_tracking_demo(
        num_frames=80, noise_rate=0.3, window_size=5, seed=42
    )
    assert report["total_smoothed_flips"] <= report["total_raw_flips"]


def test_run_tracking_demo_zero_noise_has_zero_flips() -> None:
    report = run_tracking_demo.run_tracking_demo(
        num_frames=30, noise_rate=0.0, window_size=5, seed=42
    )
    assert report["total_raw_flips"] == 0
    assert report["total_smoothed_flips"] == 0


def test_main_writes_report_json(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        run_tracking_demo,
        "parse_args",
        lambda: run_tracking_demo.argparse.Namespace(
            num_frames=20, noise_rate=0.2, window_size=5, seed=1, output=str(output_path)
        ),
    )
    run_tracking_demo.main()

    assert output_path.is_file()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["num_frames"] == 20
    assert "per_track" in report
