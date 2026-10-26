"""Tests for `license_audit.py` -- deterministic classification, no network."""

from __future__ import annotations

from compliance.license_audit import LicenseRisk, audit_sbom, classify_license


def test_classify_license_none_or_blank_is_unknown() -> None:
    assert classify_license(None) is LicenseRisk.UNKNOWN
    assert classify_license("") is LicenseRisk.UNKNOWN
    assert classify_license("   ") is LicenseRisk.UNKNOWN


def test_classify_license_recognizes_permissive_spdx_ids() -> None:
    for name in ("MIT", "Apache-2.0", "BSD-3-Clause", "ISC"):
        assert classify_license(name) is LicenseRisk.PERMISSIVE, name


def test_classify_license_recognizes_permissive_free_text() -> None:
    # transformers' actual installed `License` metadata field value.
    assert classify_license("Apache 2.0 License") is LicenseRisk.PERMISSIVE


def test_classify_license_recognizes_mit_variant() -> None:
    # pillow's actual installed `License-Expression` value.
    assert classify_license("MIT-CMU") is LicenseRisk.PERMISSIVE


def test_classify_license_recognizes_copyleft() -> None:
    for name in ("GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-3.0"):
        assert classify_license(name) is LicenseRisk.COPYLEFT, name


def test_classify_license_recognizes_commercial_restricted() -> None:
    for name in ("BUSL-1.1", "SSPL-1.0", "Elastic-2.0"):
        assert classify_license(name) is LicenseRisk.COMMERCIAL_RESTRICTED, name


def test_classify_license_recognizes_classifier_trove_strings() -> None:
    agpl_classifier = (
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
    )
    assert classify_license(agpl_classifier) is LicenseRisk.COPYLEFT
    assert (
        classify_license("License :: OSI Approved :: Apache Software License")
        is LicenseRisk.PERMISSIVE
    )


def test_classify_license_rejects_full_license_text_as_unknown() -> None:
    long_text = "Copyright (c) 2005-2023, NumPy Developers.\n" + ("All rights reserved. " * 20)
    assert len(long_text) > 200
    assert classify_license(long_text) is LicenseRisk.UNKNOWN


def test_classify_license_handles_combined_spdx_expression_all_permissive() -> None:
    assert classify_license("Apache-2.0 AND BSD-3-Clause AND MIT") is LicenseRisk.PERMISSIVE


def test_classify_license_combined_expression_most_restrictive_wins() -> None:
    assert classify_license("MIT AND GPL-3.0") is LicenseRisk.COPYLEFT
    assert classify_license("MIT OR GPL-3.0") is LicenseRisk.COPYLEFT


def test_classify_license_handles_spdx_with_exception_clause() -> None:
    # torch's actual installed License-Expression includes this operand.
    assert classify_license("Apache-2.0 WITH LLVM-exception") is LicenseRisk.PERMISSIVE
    assert (
        classify_license("Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND MIT")
        is LicenseRisk.PERMISSIVE
    )


def test_classify_license_unrecognized_short_string_is_unknown() -> None:
    assert classify_license("Some Made-Up License Name") is LicenseRisk.UNKNOWN


def test_audit_sbom_classifies_every_component_and_counts_risks() -> None:
    sbom = {
        "components": [
            {"name": "a", "version": "1.0", "license": "MIT", "requirementsFile": "r.txt"},
            {
                "name": "ultralytics",
                "version": "8.0",
                "license": "AGPL-3.0",
                "requirementsFile": "r.txt",
            },
            {"name": "b", "version": "2.0", "license": None, "requirementsFile": "r.txt"},
        ],
        "unresolved": [],
    }

    report = audit_sbom(sbom)

    assert len(report.entries) == 3
    assert report.risk_counts == {
        "permissive": 1,
        "copyleft": 1,
        "commercial_restricted": 0,
        "unknown": 1,
    }
    ultralytics_entries = [e for e in report.entries if e.name == "ultralytics"]
    assert len(ultralytics_entries) == 1
    assert ultralytics_entries[0].risk is LicenseRisk.COPYLEFT


def test_audit_sbom_entries_with_risk_filters_correctly() -> None:
    sbom = {
        "components": [
            {"name": "a", "version": "1.0", "license": "MIT", "requirementsFile": "r.txt"},
            {"name": "b", "version": "1.0", "license": "GPL-3.0", "requirementsFile": "r.txt"},
        ],
        "unresolved": [],
    }
    report = audit_sbom(sbom)

    copyleft = report.entries_with_risk(LicenseRisk.COPYLEFT)
    assert [entry.name for entry in copyleft] == ["b"]
