"""Tests for `scripts/run_license_audit.py` -- the week 21 one-command deliverable."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module

run_license_audit = load_script_module("run_license_audit.py")


def test_run_full_audit_detects_ultralytics_agpl_for_real(tmp_path: Path) -> None:
    """The actual C21.2 requirement: confirm ultralytics/AGPL-3.0 is
    found by the AUTOMATIC run, not asserted from prior knowledge."""
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("ultralytics\n", encoding="utf-8")

    report = run_license_audit.run_full_audit([requirements_path])

    ultralytics_entries = [
        e for e in report["license_audit"]["entries"] if e["name"] == "ultralytics"
    ]
    assert len(ultralytics_entries) == 1
    assert ultralytics_entries[0]["license"] == "AGPL-3.0"
    assert ultralytics_entries[0]["risk"] == "copyleft"


def test_run_full_audit_includes_checklist_and_adversarial_demo(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("pytest\n", encoding="utf-8")

    report = run_license_audit.run_full_audit([requirements_path])

    assert report["checklist"]["summary"]["open"] > 0
    assert len(report["checklist"]["items"]) > 0
    assert report["adversarial_demo"]["numFrames"] > 0
    assert "defenseRecoveredCorrectVerdict" in report["adversarial_demo"]


def test_main_writes_report_and_checklist(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        run_license_audit,
        "parse_args",
        lambda: run_license_audit.argparse.Namespace(
            requirements_files=None, output_dir=str(output_dir)
        ),
    )
    monkeypatch.setattr(
        run_license_audit,
        "run_full_audit",
        lambda requirements_files: {
            "sbom": {"components": [{"name": "a"}], "unresolved": []},
            "license_audit": {
                "risk_counts": {
                    "permissive": 1,
                    "copyleft": 0,
                    "commercial_restricted": 0,
                    "unknown": 0,
                },
                "entries": [
                    {"name": "a", "version": "1.0", "license": "MIT", "risk": "permissive"}
                ],
            },
            "checklist": {"summary": {"closed": 0, "open": 1, "not_applicable": 0}, "items": []},
            "adversarial_demo": {
                "numFrames": 5,
                "rawVerdicts": ["defect", "ok"],
                "rawFalseNegativeRate": 0.4,
                "smoothedFinalVerdict": "defect",
                "defenseRecoveredCorrectVerdict": True,
            },
        },
    )

    run_license_audit.main()

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["sbom"]["components"] == [{"name": "a"}]
    assert (output_dir / "checklist.md").is_file()
