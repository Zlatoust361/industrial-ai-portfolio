"""Unit tests for utils/report.py.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_report.py
"""

from utils.report import render_report

ENV_REPORT_DEV = {
    "is_jetson": False,
    "l4t_version": None,
    "jetpack_generation": None,
    "available_memory_mb": 12345.6,
    "power_mode": None,
}

ENV_REPORT_JETSON = {
    "is_jetson": True,
    "l4t_version": "R36.4.0",
    "jetpack_generation": "6.x",
    "available_memory_mb": 4096.0,
    "power_mode": "15W",
}

QUALITY_GATE_GREEN = {
    "status": "GREEN",
    "defect_rate": 0.01,
    "thresholds": {"green_max_defect_rate": 0.02, "yellow_max_defect_rate": 0.10},
}


def test_render_report_includes_title_and_status() -> None:
    report = render_report(
        title="Inspection pipeline report",
        env_report=ENV_REPORT_DEV,
        interpret_summary={"total": 10, "defect_count": 1, "defect_rate": 0.1},
        quality_gate=QUALITY_GATE_GREEN,
        verdicts=[],
    )

    assert report.startswith("# Inspection pipeline report")
    assert "**Status: GREEN**" in report
    assert "Total images: 10" in report


def test_render_report_off_jetson_omits_l4t_fields() -> None:
    report = render_report(
        title="t",
        env_report=ENV_REPORT_DEV,
        interpret_summary={"total": 0, "defect_count": 0, "defect_rate": 0.0},
        quality_gate=QUALITY_GATE_GREEN,
        verdicts=[],
    )
    assert "L4T version" not in report


def test_render_report_on_jetson_includes_l4t_fields() -> None:
    report = render_report(
        title="t",
        env_report=ENV_REPORT_JETSON,
        interpret_summary={"total": 0, "defect_count": 0, "defect_rate": 0.0},
        quality_gate=QUALITY_GATE_GREEN,
        verdicts=[],
    )
    assert "R36.4.0" in report
    assert "6.x" in report
    assert "15W" in report


def test_render_report_lists_flagged_images() -> None:
    verdicts = [
        {"image": "a.jpg", "defect_score": 0.9, "verdict": "defect"},
        {"image": "b.jpg", "defect_score": 0.1, "verdict": "ok"},
    ]
    report = render_report(
        title="t",
        env_report=ENV_REPORT_DEV,
        interpret_summary={"total": 2, "defect_count": 1, "defect_rate": 0.5},
        quality_gate=QUALITY_GATE_GREEN,
        verdicts=verdicts,
    )
    assert "Flagged images (1)" in report
    assert "a.jpg" in report
    assert "b.jpg" not in report


def test_render_report_no_defects_says_none() -> None:
    report = render_report(
        title="t",
        env_report=ENV_REPORT_DEV,
        interpret_summary={"total": 2, "defect_count": 0, "defect_rate": 0.0},
        quality_gate=QUALITY_GATE_GREEN,
        verdicts=[{"image": "a.jpg", "defect_score": 0.1, "verdict": "ok"}],
    )
    assert "Flagged images (0)" in report
    assert "None." in report


def test_render_report_truncates_long_defect_list_with_disclosure() -> None:
    verdicts = [
        {"image": f"img{i}.jpg", "defect_score": 0.9, "verdict": "defect"} for i in range(5)
    ]
    report = render_report(
        title="t",
        env_report=ENV_REPORT_DEV,
        interpret_summary={"total": 5, "defect_count": 5, "defect_rate": 1.0},
        quality_gate=QUALITY_GATE_GREEN,
        verdicts=verdicts,
        max_defects_listed=2,
    )
    assert "img0.jpg" in report
    assert "img1.jpg" in report
    assert "img2.jpg" not in report
    assert "3 more not shown" in report
