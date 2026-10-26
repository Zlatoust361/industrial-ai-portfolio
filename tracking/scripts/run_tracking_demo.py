#!/usr/bin/env python3
"""Week 13's whole non-interactive deliverable, one command.

Runs a deterministic, SYNTHETIC multi-object video stream (two objects
on straight-line motion, one "defect"-truth and one "ok"-truth, each
with injected per-frame label noise simulating a real detector's
blur/glare/angle flicker -- see `generate_synthetic_stream`'s docstring
for exactly why synthetic, not a real video, and why that's still a
real, non-trivial test of this week's actual code) through
`tracking.sort.IoUTracker` + `tracking.smoothing.VerdictSmoother`, and
reports the ONE number the roadmap asks for: how many times each
track's verdict flipped, raw vs smoothed.

Usage:
    PYTHONPATH=. python3 scripts/run_tracking_demo.py [--num-frames 60] \\
        [--noise-rate 0.3] [--window-size 5] [--seed 42] \\
        [--output outputs/tracking_demo/report.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_TRACKING_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TRACKING_ROOT))

from smoothing import VerdictSmoother, count_flips  # noqa: E402
from sort import BBox, Detection, IoUTracker  # noqa: E402

DEFAULT_OUTPUT = "outputs/tracking_demo/report.json"


@dataclass(frozen=True)
class SyntheticObject:
    """One ground-truth object's straight-line motion + true label."""

    name: str
    true_label: str
    x0: float
    y0: float
    dx: float
    dy: float
    width: float
    height: float


DEFAULT_OBJECTS = (
    SyntheticObject("A", "defect", x0=10.0, y0=50.0, dx=2.0, dy=0.0, width=30.0, height=30.0),
    SyntheticObject("B", "ok", x0=250.0, y0=80.0, dx=-1.5, dy=0.4, width=25.0, height=25.0),
)
"""Two objects, spatially far apart throughout -- a real stream would have
near-misses/occlusion too, out of scope for this first pass (see
`tracking/README.md`'s Honesty note)."""


def generate_synthetic_stream(
    num_frames: int,
    noise_rate: float,
    seed: int,
    objects: tuple[SyntheticObject, ...] = DEFAULT_OBJECTS,
) -> tuple[list[list[Detection]], dict[str, str]]:
    """Build a deterministic sequence of per-frame detections with injected label noise.

    Why synthetic, not a real video: this week's actual new code
    (`sort.py`/`smoothing.py`) operates on `Detection` objects, not pixels
    -- feeding it a real video would ALSO require a real detector
    (weeks 01-09's CLIP/YOLO, needing real weights this sandbox can't
    download, see `peft/README.md`'s identical Honesty note for the same
    boundary) just to produce the `Detection`s this demo already needs as
    INPUT. A synthetic stream with known ground truth is a real,
    non-trivial test of tracking continuity + smoothing correctness on
    its own terms -- it just can't speak to real-world detector noise
    characteristics or real camera motion. `adapters.py` plus a real
    detector is the natural next step once real video is available on
    real hardware (see `tracking/README.md`'s Honesty note).

    Args:
        num_frames: Frames to generate.
        noise_rate: Per-frame probability a detection's RAW label is
            flipped away from its object's true label (simulating
            per-frame classifier noise -- blur/glare/angle, not a change
            in the real defect state).
        seed: Deterministic RNG seed -- same seed, same stream, every run.
        objects: Ground-truth objects to simulate.

    Returns:
        `(frames, ground_truth)` -- `frames[i]` is frame `i`'s detections
        (unordered, as a real detector would return them); `ground_truth`
        maps each object's `name` to its true label, for the report.
    """
    rng = random.Random(seed)
    labels = sorted({obj.true_label for obj in objects})

    frames: list[list[Detection]] = []
    for frame_index in range(num_frames):
        detections = []
        for obj in objects:
            x = obj.x0 + obj.dx * frame_index
            y = obj.y0 + obj.dy * frame_index
            raw_label = obj.true_label
            if rng.random() < noise_rate:
                raw_label = next(label for label in labels if label != obj.true_label)
            bbox = BBox(x_min=x, y_min=y, x_max=x + obj.width, y_max=y + obj.height)
            detections.append(Detection(label=raw_label, score=rng.uniform(0.55, 0.95), bbox=bbox))
        frames.append(detections)

    ground_truth = {obj.name: obj.true_label for obj in objects}
    return frames, ground_truth


def run_tracking_demo(
    num_frames: int, noise_rate: float, window_size: int, seed: int
) -> dict[str, object]:
    """Run the full tracker + smoother pipeline over a synthetic stream, report flip counts.

    Returns:
        A JSON-serializable report: per-track raw/smoothed label
        histories and flip counts, plus the aggregate before/after totals
        this week's deliverable is actually about.
    """
    frames, ground_truth = generate_synthetic_stream(num_frames, noise_rate, seed)

    tracker = IoUTracker()
    smoother = VerdictSmoother(window_size=window_size)

    raw_history: dict[int, list[str]] = {}
    smoothed_history: dict[int, list[str]] = {}

    for frame_detections in frames:
        tracked = tracker.update(frame_detections)
        for t in tracked:
            raw_history.setdefault(t.track_id, []).append(t.label)
            smoothed = smoother.update(t.track_id, t.label)
            smoothed_history.setdefault(t.track_id, []).append(smoothed.label)

    per_track = {}
    total_raw_flips = 0
    total_smoothed_flips = 0
    for track_id in sorted(raw_history):
        raw_flips = count_flips(raw_history[track_id])
        smoothed_flips = count_flips(smoothed_history[track_id])
        total_raw_flips += raw_flips
        total_smoothed_flips += smoothed_flips
        per_track[str(track_id)] = {
            "num_frames_tracked": len(raw_history[track_id]),
            "raw_flips": raw_flips,
            "smoothed_flips": smoothed_flips,
        }

    return {
        "num_frames": num_frames,
        "noise_rate": noise_rate,
        "window_size": window_size,
        "seed": seed,
        "ground_truth": ground_truth,
        "num_tracks": len(raw_history),
        "total_raw_flips": total_raw_flips,
        "total_smoothed_flips": total_smoothed_flips,
        "per_track": per_track,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_tracking_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--noise-rate", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the demo, print the before/after summary, write the full report."""
    args = parse_args()
    report = run_tracking_demo(args.num_frames, args.noise_rate, args.window_size, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{report['num_tracks']} track(s) over {report['num_frames']} frames "
        f"(noise_rate={report['noise_rate']}, window_size={report['window_size']}):"
    )
    for track_id, stats in report["per_track"].items():
        print(
            f"  track {track_id}: {stats['raw_flips']} raw flips -> "
            f"{stats['smoothed_flips']} smoothed flips "
            f"(over {stats['num_frames_tracked']} frames)"
        )
    print(
        f"Total: {report['total_raw_flips']} raw flips -> "
        f"{report['total_smoothed_flips']} smoothed flips. "
        f"Report -> {output_path}"
    )


if __name__ == "__main__":
    main()
