"""Unit tests for utils/interpret.py.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_interpret.py
"""

import pytest
from utils.interpret import interpret_detections


def test_interpret_detections_splits_by_threshold() -> None:
    detections = [
        {"image": "a.jpg", "defect_score": 0.9},
        {"image": "b.jpg", "defect_score": 0.1},
        {"image": "c.jpg", "defect_score": 0.5},  # exactly at threshold -> defect
    ]

    verdicts, summary = interpret_detections(detections, defect_score_threshold=0.5)

    assert [v.verdict for v in verdicts] == ["defect", "ok", "defect"]
    assert summary == {
        "total": 3,
        "defect_count": 2,
        "ok_count": 1,
        "defect_rate": pytest.approx(2 / 3),
    }


def test_interpret_detections_empty_input_has_zero_defect_rate() -> None:
    verdicts, summary = interpret_detections([], defect_score_threshold=0.5)

    assert verdicts == []
    assert summary == {"total": 0, "defect_count": 0, "ok_count": 0, "defect_rate": 0.0}


def test_interpret_detections_all_ok() -> None:
    detections = [{"image": "a.jpg", "defect_score": 0.0}, {"image": "b.jpg", "defect_score": 0.2}]
    _, summary = interpret_detections(detections, defect_score_threshold=0.5)
    assert summary["defect_rate"] == 0.0


def test_interpret_detections_all_defect() -> None:
    detections = [{"image": "a.jpg", "defect_score": 1.0}, {"image": "b.jpg", "defect_score": 0.9}]
    _, summary = interpret_detections(detections, defect_score_threshold=0.5)
    assert summary["defect_rate"] == 1.0


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_interpret_detections_rejects_out_of_range_threshold(bad_threshold: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpret_detections([], defect_score_threshold=bad_threshold)
