"""Frame-sampling strategy: process every Nth frame, not the whole stream.

Named explicitly as one of this week's skills in
`course/ROADMAP-WEEKS-13-24.md`'s week 13 entry: "выбор между 'обработать
каждый N-й кадр' и 'обработать поток целиком' под бюджет Jetson" --
running the detector (let alone detector + tracker + smoother together)
on every single frame of a real video stream may not fit the Jetson's
compute/latency budget alongside everything else already running (week
01's `memory_budget_mb`, same principle every week since has checked,
just the TIME axis instead of the memory one here).
"""

from __future__ import annotations


def select_frame_indices(num_frames: int, stride: int) -> list[int]:
    """Every `stride`-th frame index, 0-based, always including frame 0.

    Args:
        num_frames: Total frames in the stream.
        stride: Process 1 in every `stride` frames (`1` = every frame,
            i.e. no sampling at all).

    Returns:
        Sorted frame indices to actually run the detector on.

    Raises:
        ValueError: If `stride < 1`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride!r}.")
    return list(range(0, num_frames, stride))


def effective_detection_rate(source_fps: float, stride: int) -> float:
    """The detector's actual per-second call rate after striding.

    Args:
        source_fps: The video stream's real frame rate.
        stride: As passed to `select_frame_indices`.

    Returns:
        `source_fps / stride` -- e.g. a 30 fps stream at `stride=3` only
        asks the detector for 10 inferences/second, not 30.

    Raises:
        ValueError: If `stride < 1`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride!r}.")
    return source_fps / stride


def max_stride_for_budget(
    source_fps: float, detector_latency_s: float, headroom: float = 0.8
) -> int:
    """Largest stride that keeps the detector's real-time load under `headroom`.

    Args:
        source_fps: The video stream's real frame rate.
        detector_latency_s: Measured wall-clock time for ONE detector call
            on the target device (e.g. week 01's `jetson_benchmark.json`
            latency figure for the actual deployed model/precision/power
            mode -- not a number this function invents).
        headroom: Fraction of each frame's time budget (`1 / source_fps`)
            the detector is allowed to actually use, leaving the rest for
            the tracker/smoother/everything else sharing the device. A
            bare "fits exactly" (`headroom=1.0`) leaves zero margin for
            jitter -- not recommended, but not blocked either.

    Returns:
        The smallest `stride >= 1` such that
        `detector_latency_s <= (stride / source_fps) * headroom`, i.e.
        the detector's own latency fits within its allotted share of the
        strided frame interval.

    Raises:
        ValueError: If `source_fps <= 0`, `detector_latency_s < 0`, or
            `headroom <= 0`.
    """
    if source_fps <= 0:
        raise ValueError(f"source_fps must be > 0, got {source_fps!r}.")
    if detector_latency_s < 0:
        raise ValueError(f"detector_latency_s must be >= 0, got {detector_latency_s!r}.")
    if headroom <= 0:
        raise ValueError(f"headroom must be > 0, got {headroom!r}.")

    import math

    required_stride = (detector_latency_s * source_fps) / headroom
    return max(1, math.ceil(required_stride))
