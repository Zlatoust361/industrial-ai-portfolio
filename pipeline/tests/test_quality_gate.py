"""Unit tests for utils/quality_gate.py.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_quality_gate.py
"""

import pytest
from utils.quality_gate import evaluate_quality_gate


def test_evaluate_quality_gate_green_below_green_threshold() -> None:
    result = evaluate_quality_gate(
        defect_rate=0.01, green_max_defect_rate=0.02, yellow_max_defect_rate=0.10
    )
    assert result.status == "GREEN"


def test_evaluate_quality_gate_yellow_between_thresholds() -> None:
    result = evaluate_quality_gate(
        defect_rate=0.05, green_max_defect_rate=0.02, yellow_max_defect_rate=0.10
    )
    assert result.status == "YELLOW"


def test_evaluate_quality_gate_red_at_or_above_yellow_threshold() -> None:
    result = evaluate_quality_gate(
        defect_rate=0.10, green_max_defect_rate=0.02, yellow_max_defect_rate=0.10
    )
    assert result.status == "RED"


def test_evaluate_quality_gate_boundary_is_exclusive_on_green_side() -> None:
    # exactly at green_max_defect_rate -> not GREEN (strictly below only).
    result = evaluate_quality_gate(
        defect_rate=0.02, green_max_defect_rate=0.02, yellow_max_defect_rate=0.10
    )
    assert result.status == "YELLOW"


@pytest.mark.parametrize("bad_rate", [-0.1, 1.1])
def test_evaluate_quality_gate_rejects_out_of_range_defect_rate(bad_rate: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_quality_gate(bad_rate, 0.02, 0.10)


def test_evaluate_quality_gate_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="must be <"):
        evaluate_quality_gate(
            defect_rate=0.05, green_max_defect_rate=0.10, yellow_max_defect_rate=0.02
        )


def test_evaluate_quality_gate_result_to_dict_shape() -> None:
    result = evaluate_quality_gate(0.01, 0.02, 0.10)
    assert result.to_dict() == {
        "status": "GREEN",
        "defect_rate": 0.01,
        "thresholds": {"green_max_defect_rate": 0.02, "yellow_max_defect_rate": 0.10},
    }
