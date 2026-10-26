"""Tests for `sort.py` -- `iou()` and `IoUTracker`.

The `test_track_survives_a_label_flip` case below is the regression test
for the real design bug caught (and fixed) during this week's
implementation, BEFORE any test existed: an earlier version of
`IoUTracker._match` refused to match a detection to a track of a
DIFFERENT label, which would fragment a track's identity on every single
noisy label flip -- defeating the whole point of `smoothing.py`. See
`sort.py`'s own `_match` docstring for the full reasoning.
"""

from __future__ import annotations

import pytest
from sort import BBox, Detection, IoUTracker, iou

# ---------------------------------------------------------------------------
# iou()
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one() -> None:
    box = BBox(0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    a = BBox(0.0, 0.0, 10.0, 10.0)
    b = BBox(100.0, 100.0, 110.0, 110.0)
    assert iou(a, b) == 0.0


def test_iou_half_overlap() -> None:
    a = BBox(0.0, 0.0, 10.0, 10.0)
    b = BBox(5.0, 0.0, 15.0, 10.0)
    # intersection = 5x10=50, union = 100+100-50=150
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_iou_degenerate_box_is_zero() -> None:
    degenerate = BBox(5.0, 5.0, 5.0, 5.0)
    other = BBox(0.0, 0.0, 10.0, 10.0)
    assert iou(degenerate, other) == 0.0


# ---------------------------------------------------------------------------
# IoUTracker construction
# ---------------------------------------------------------------------------


def test_invalid_iou_threshold_raises() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        IoUTracker(iou_threshold=1.5)


def test_invalid_max_age_raises() -> None:
    with pytest.raises(ValueError, match="max_age"):
        IoUTracker(max_age=0)


def test_invalid_min_hits_raises() -> None:
    with pytest.raises(ValueError, match="min_hits"):
        IoUTracker(min_hits=0)


# ---------------------------------------------------------------------------
# IoUTracker.update() -- identity across frames
# ---------------------------------------------------------------------------


def _box_at(x: float, y: float, size: float = 10.0) -> BBox:
    return BBox(x, y, x + size, y + size)


def test_first_frame_creates_one_track_per_detection() -> None:
    tracker = IoUTracker(min_hits=1)
    detections = [
        Detection(label="defect", score=0.9, bbox=_box_at(0.0, 0.0)),
        Detection(label="ok", score=0.8, bbox=_box_at(100.0, 100.0)),
    ]
    tracked = tracker.update(detections)
    assert {t.track_id for t in tracked} == {1, 2}
    assert tracker.active_track_count == 2


def test_same_object_next_frame_keeps_same_track_id() -> None:
    tracker = IoUTracker(min_hits=1)
    first = tracker.update([Detection("defect", 0.9, _box_at(0.0, 0.0))])
    track_id = first[0].track_id

    # Small motion -- still heavily overlapping with the previous box.
    second = tracker.update([Detection("defect", 0.9, _box_at(1.0, 1.0))])
    assert len(second) == 1
    assert second[0].track_id == track_id
    assert second[0].hits == 2


def test_far_away_detection_starts_a_new_track() -> None:
    tracker = IoUTracker(min_hits=1)
    tracker.update([Detection("defect", 0.9, _box_at(0.0, 0.0))])
    second = tracker.update([Detection("defect", 0.9, _box_at(500.0, 500.0))])
    assert second[0].track_id == 2
    assert tracker.active_track_count == 2


def test_track_survives_a_label_flip() -> None:
    """A noisy raw-label flip on the SAME physical object must not fragment its track.

    This is the exact scenario `smoothing.py`'s `VerdictSmoother` exists
    to clean up -- if the tracker itself refused to match across a label
    change, there would be nothing left for the smoother to smooth: the
    "track" would just restart with a new id every time the raw label
    flickered.
    """
    tracker = IoUTracker(min_hits=1)
    first = tracker.update([Detection("defect", 0.9, _box_at(0.0, 0.0))])
    track_id = first[0].track_id

    flipped = tracker.update([Detection("ok", 0.7, _box_at(1.0, 1.0))])
    assert len(flipped) == 1
    assert flipped[0].track_id == track_id
    assert flipped[0].label == "ok"
    assert flipped[0].hits == 2

    back = tracker.update([Detection("defect", 0.85, _box_at(2.0, 2.0))])
    assert back[0].track_id == track_id
    assert back[0].label == "defect"
    assert back[0].hits == 3


def test_two_close_tracks_each_keep_their_own_identity_across_flips() -> None:
    tracker = IoUTracker(min_hits=1, iou_threshold=0.3)
    first = tracker.update(
        [
            Detection("defect", 0.9, _box_at(0.0, 0.0)),
            Detection("ok", 0.9, _box_at(200.0, 200.0)),
        ]
    )
    id_a = next(t.track_id for t in first if t.bbox.x_min == 0.0)
    id_b = next(t.track_id for t in first if t.bbox.x_min == 200.0)
    assert id_a != id_b

    second = tracker.update(
        [
            Detection("ok", 0.6, _box_at(1.0, 1.0)),  # A flipped label
            Detection("defect", 0.6, _box_at(201.0, 201.0)),  # B flipped label
        ]
    )
    label_by_id = {t.track_id: t.label for t in second}
    assert label_by_id[id_a] == "ok"
    assert label_by_id[id_b] == "defect"


# ---------------------------------------------------------------------------
# IoUTracker.update() -- aging and dropping
# ---------------------------------------------------------------------------


def test_unmatched_track_is_not_reported_but_kept_until_max_age() -> None:
    tracker = IoUTracker(min_hits=1, max_age=2)
    tracker.update([Detection("defect", 0.9, _box_at(0.0, 0.0))])

    # Two misses in a row: track survives (time_since_update <= max_age).
    assert tracker.update([]) == []
    assert tracker.active_track_count == 1
    assert tracker.update([]) == []
    assert tracker.active_track_count == 1

    # Third miss exceeds max_age=2: track is dropped.
    assert tracker.update([]) == []
    assert tracker.active_track_count == 0


def test_track_reappearing_within_max_age_keeps_its_id() -> None:
    tracker = IoUTracker(min_hits=1, max_age=3)
    first = tracker.update([Detection("defect", 0.9, _box_at(0.0, 0.0))])
    track_id = first[0].track_id

    tracker.update([])  # one miss
    reappeared = tracker.update([Detection("defect", 0.9, _box_at(1.0, 1.0))])
    assert reappeared[0].track_id == track_id
    assert reappeared[0].hits == 2


def test_min_hits_suppresses_unconfirmed_tracks() -> None:
    tracker = IoUTracker(min_hits=2)
    first = tracker.update([Detection("defect", 0.9, _box_at(0.0, 0.0))])
    assert first == []  # hits == 1 < min_hits == 2, not yet reported

    second = tracker.update([Detection("defect", 0.9, _box_at(1.0, 1.0))])
    assert len(second) == 1
    assert second[0].hits == 2


def test_empty_detections_on_empty_tracker_returns_empty() -> None:
    tracker = IoUTracker()
    assert tracker.update([]) == []
    assert tracker.active_track_count == 0
