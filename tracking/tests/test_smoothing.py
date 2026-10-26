"""Tests for `smoothing.py` -- `VerdictSmoother` and `count_flips`."""

from __future__ import annotations

import pytest
from smoothing import VerdictSmoother, count_flips


def test_invalid_window_size_raises() -> None:
    with pytest.raises(ValueError, match="window_size"):
        VerdictSmoother(window_size=0)


def test_first_update_returns_that_label_with_full_confidence() -> None:
    smoother = VerdictSmoother(window_size=5)
    verdict = smoother.update(track_id=1, raw_label="defect")
    assert verdict.label == "defect"
    assert verdict.confidence == 1.0
    assert verdict.window_size == 1


def test_majority_vote_over_window() -> None:
    smoother = VerdictSmoother(window_size=5)
    for label in ["defect", "defect", "ok", "defect", "defect"]:
        verdict = smoother.update(track_id=1, raw_label=label)
    assert verdict.label == "defect"
    assert verdict.confidence == pytest.approx(4 / 5)
    assert verdict.window_size == 5


def test_window_slides_once_full() -> None:
    smoother = VerdictSmoother(window_size=3)
    for label in ["defect", "defect", "defect"]:
        smoother.update(track_id=1, raw_label=label)
    # Pushes out the oldest "defect"; window is now ["defect", "ok", "ok"].
    smoother.update(track_id=1, raw_label="ok")
    verdict = smoother.update(track_id=1, raw_label="ok")
    assert verdict.label == "ok"
    assert verdict.window_size == 3
    assert verdict.confidence == pytest.approx(2 / 3)


def test_tracks_are_independent() -> None:
    smoother = VerdictSmoother(window_size=5)
    smoother.update(track_id=1, raw_label="defect")
    smoother.update(track_id=1, raw_label="defect")
    verdict_2 = smoother.update(track_id=2, raw_label="ok")
    assert verdict_2.label == "ok"
    assert verdict_2.window_size == 1  # unaffected by track 1's history


def test_reset_drops_history_for_that_track_only() -> None:
    smoother = VerdictSmoother(window_size=5)
    smoother.update(track_id=1, raw_label="defect")
    smoother.update(track_id=1, raw_label="defect")
    smoother.update(track_id=2, raw_label="ok")

    smoother.reset(track_id=1)
    verdict_1 = smoother.update(track_id=1, raw_label="ok")
    assert verdict_1.window_size == 1  # track 1's history was actually cleared

    verdict_2 = smoother.update(track_id=2, raw_label="ok")
    assert verdict_2.window_size == 2  # track 2 untouched by the reset


def test_reset_unknown_track_id_is_a_no_op() -> None:
    smoother = VerdictSmoother(window_size=5)
    smoother.reset(track_id=999)  # must not raise


def test_tie_breaks_lexicographically_with_low_confidence() -> None:
    smoother = VerdictSmoother(window_size=2)
    smoother.update(track_id=1, raw_label="ok")
    verdict = smoother.update(track_id=1, raw_label="defect")
    assert verdict.label == "defect"  # "defect" < "ok" lexicographically
    assert verdict.confidence == pytest.approx(0.5)


def test_count_flips_on_stable_sequence_is_zero() -> None:
    assert count_flips(["ok", "ok", "ok"]) == 0


def test_count_flips_counts_adjacent_disagreements() -> None:
    assert count_flips(["ok", "defect", "defect", "ok", "ok", "defect"]) == 3


def test_count_flips_on_short_sequences_is_zero() -> None:
    assert count_flips([]) == 0
    assert count_flips(["ok"]) == 0


def test_smoothing_reduces_flip_count_under_noise() -> None:
    """The actual headline claim this week's deliverable makes: smoothing helps."""
    raw = ["defect", "ok", "defect", "defect", "ok", "defect", "defect", "defect", "ok", "defect"]
    smoother = VerdictSmoother(window_size=5)
    smoothed = [smoother.update(track_id=1, raw_label=label).label for label in raw]

    assert count_flips(smoothed) <= count_flips(raw)
