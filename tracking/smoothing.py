"""Temporal smoothing of per-track verdicts -- the actual "stop flickering" fix.

A tracker (`sort.py`) gives detections a stable id across frames; it does
NOT fix a model re-flagging the SAME physical object as `"ok"` on one
frame and `"defect"` the next from blur/glare/a bad angle (week 02's own
`ОБ-02` failure-table entry in `course/FAILURE-TABLES.md` names exactly
this class of per-frame noise -- "разделимость низкая" degrading into
flip-flopping right at the decision boundary). This module accumulates a
rolling window of raw verdicts PER TRACK ID and reports a smoothed
(majority-vote) verdict instead of the raw per-frame one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SmoothedVerdict:
    """The majority-vote verdict for one track, over its current window.

    Attributes:
        track_id: As assigned by `sort.IoUTracker`.
        label: Majority-vote label over the window.
        confidence: Fraction of the window's raw verdicts agreeing with
            `label`, in `(0, 1]` -- `1.0` means every frame in the window
            agreed; low values (near `1 / num_distinct_labels_seen`) mean
            the raw signal is genuinely noisy, not that smoothing failed.
        window_size: How many frames of history this verdict is actually
            based on (`< window_size` for a track's first few frames).
    """

    track_id: int
    label: str
    confidence: float
    window_size: int


class VerdictSmoother:
    """Per-track-id rolling-window majority vote over raw verdict labels."""

    def __init__(self, window_size: int = 5) -> None:
        """
        Args:
            window_size: Frames of history kept per track. Larger = more
                stable but slower to react to a REAL verdict change (a
                genuinely new defect appearing partway through a track's
                life) -- the actual flicker-vs-latency tradeoff this
                week's deliverable measures (see
                `scripts/run_tracking_demo.py`'s before/after report).

        Raises:
            ValueError: If `window_size < 1`.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size!r}.")
        self.window_size = window_size
        self._history: dict[int, deque[str]] = {}

    def update(self, track_id: int, raw_label: str) -> SmoothedVerdict:
        """Record one frame's raw verdict for `track_id`, return the smoothed verdict.

        Args:
            track_id: Stable track id from `sort.IoUTracker`.
            raw_label: This frame's raw (unsmoothed) verdict for that track.

        Returns:
            The majority-vote label over the current window, with its
            agreement fraction as `confidence`. Ties broken by whichever
            label is lexicographically first (deterministic, not
            meaningful on its own -- a genuine tie means the window is too
            short or too noisy to trust yet, reflected in the low
            `confidence`, not papered over by the tie-break choice).
        """
        history = self._history.setdefault(track_id, deque(maxlen=self.window_size))
        history.append(raw_label)

        counts: dict[str, int] = {}
        for label in history:
            counts[label] = counts.get(label, 0) + 1
        majority_label = max(sorted(counts), key=counts.get)
        confidence = counts[majority_label] / len(history)

        return SmoothedVerdict(
            track_id=track_id,
            label=majority_label,
            confidence=confidence,
            window_size=len(history),
        )

    def reset(self, track_id: int) -> None:
        """Drop history for `track_id` -- call once its track has aged out of `IoUTracker`.

        Not calling this just leaks a handful of strings per departed
        track, not a correctness bug -- but a long-running process (the
        actual Jetson deployment scenario, not a one-shot demo) should
        call it so `VerdictSmoother`'s own memory stays bounded by
        CURRENTLY active tracks, not every track ever seen.
        """
        self._history.pop(track_id, None)


def count_flips(labels: list[str]) -> int:
    """Count how many times consecutive labels differ -- the "flicker" metric.

    Args:
        labels: A sequence of per-frame verdicts for ONE track, in order.

    Returns:
        Number of adjacent pairs that differ. `0` for a track whose
        verdict never changed (including a track with 0 or 1 frames).
    """
    return sum(1 for a, b in zip(labels, labels[1:], strict=False) if a != b)
