"""Tests for `adversarial_demo.py` -- runs against the REAL `04_detect.py::StubDetector`."""

from __future__ import annotations

import pytest

from compliance.adversarial_demo import (
    DEFAULT_PATCH_COVERAGES,
    run_adversarial_patch_demo,
)


def test_default_scenario_measures_a_real_per_frame_vulnerability() -> None:
    result = run_adversarial_patch_demo()

    assert result.num_frames == len(DEFAULT_PATCH_COVERAGES)
    # Frames at coverage 0.95 and 1.00 (indices 1, 2) hide enough of the
    # marker to evade the real StubDetector on that frame alone.
    assert result.raw_verdicts == ("defect", "ok", "ok", "defect", "defect")
    assert result.raw_false_negative_rate == pytest.approx(2 / 5)


def test_default_scenario_defense_recovers_the_correct_verdict() -> None:
    result = run_adversarial_patch_demo()

    assert result.smoothed_final_verdict == "defect"
    assert result.defense_recovered_correct_verdict is True


def test_fully_visible_marker_every_frame_is_never_a_false_negative() -> None:
    result = run_adversarial_patch_demo(coverages=(0.0, 0.0, 0.0))

    assert result.raw_verdicts == ("defect", "defect", "defect")
    assert result.raw_false_negative_rate == 0.0
    assert result.defense_recovered_correct_verdict is True


def test_defense_has_a_real_limit_when_most_frames_are_successfully_attacked() -> None:
    """Honest limit, not hidden: if the attacker fools a MAJORITY of the
    window's frames, majority-vote smoothing also reports the wrong
    verdict -- this is documented, not swept under the rug."""
    result = run_adversarial_patch_demo(coverages=(1.0, 1.0, 1.0, 0.0, 0.0))

    assert result.raw_false_negative_rate == pytest.approx(3 / 5)
    assert result.smoothed_final_verdict == "ok"
    assert result.defense_recovered_correct_verdict is False


def test_run_adversarial_patch_demo_rejects_empty_coverages() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_adversarial_patch_demo(coverages=())


def test_run_adversarial_patch_demo_is_deterministic() -> None:
    result_a = run_adversarial_patch_demo()
    result_b = run_adversarial_patch_demo()
    assert result_a == result_b
