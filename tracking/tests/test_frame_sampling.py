"""Tests for `frame_sampling.py`."""

from __future__ import annotations

import pytest
from frame_sampling import effective_detection_rate, max_stride_for_budget, select_frame_indices

# ---------------------------------------------------------------------------
# select_frame_indices
# ---------------------------------------------------------------------------


def test_select_frame_indices_stride_one_is_every_frame() -> None:
    assert select_frame_indices(num_frames=5, stride=1) == [0, 1, 2, 3, 4]


def test_select_frame_indices_stride_three() -> None:
    assert select_frame_indices(num_frames=10, stride=3) == [0, 3, 6, 9]


def test_select_frame_indices_always_includes_frame_zero() -> None:
    assert select_frame_indices(num_frames=1, stride=100) == [0]


def test_select_frame_indices_zero_frames_is_empty() -> None:
    assert select_frame_indices(num_frames=0, stride=1) == []


def test_select_frame_indices_rejects_invalid_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        select_frame_indices(num_frames=10, stride=0)


# ---------------------------------------------------------------------------
# effective_detection_rate
# ---------------------------------------------------------------------------


def test_effective_detection_rate_stride_one_is_source_fps() -> None:
    assert effective_detection_rate(source_fps=30.0, stride=1) == pytest.approx(30.0)


def test_effective_detection_rate_scales_down_with_stride() -> None:
    assert effective_detection_rate(source_fps=30.0, stride=3) == pytest.approx(10.0)


def test_effective_detection_rate_rejects_invalid_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        effective_detection_rate(source_fps=30.0, stride=0)


# ---------------------------------------------------------------------------
# max_stride_for_budget
# ---------------------------------------------------------------------------


def test_max_stride_for_budget_fast_detector_needs_no_striding() -> None:
    # 30 fps -> 33.3ms/frame budget; a 5ms detector easily fits at stride=1.
    assert max_stride_for_budget(source_fps=30.0, detector_latency_s=0.005) == 1


def test_max_stride_for_budget_slow_detector_requires_striding() -> None:
    # 30 fps -> 33.3ms/frame; a 100ms detector needs real striding.
    stride = max_stride_for_budget(source_fps=30.0, detector_latency_s=0.1, headroom=0.8)
    assert stride >= 4
    # The chosen stride must actually satisfy the budget constraint.
    frame_interval = stride / 30.0
    assert 0.1 <= frame_interval * 0.8


def test_max_stride_for_budget_never_returns_less_than_one() -> None:
    assert max_stride_for_budget(source_fps=30.0, detector_latency_s=0.0) == 1


def test_max_stride_for_budget_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="source_fps"):
        max_stride_for_budget(source_fps=0.0, detector_latency_s=0.01)
    with pytest.raises(ValueError, match="detector_latency_s"):
        max_stride_for_budget(source_fps=30.0, detector_latency_s=-0.01)
    with pytest.raises(ValueError, match="headroom"):
        max_stride_for_budget(source_fps=30.0, detector_latency_s=0.01, headroom=0.0)
