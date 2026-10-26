"""Tests for `sbom.py` -- real `importlib.metadata` resolution, no network."""

from __future__ import annotations

from pathlib import Path

from compliance.sbom import generate_sbom, parse_requirements_file


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_requirements_file_extracts_bare_names(tmp_path: Path) -> None:
    path = _write(tmp_path, "requirements.txt", "numpy\npytest\n")
    assert parse_requirements_file(path) == ["numpy", "pytest"]


def test_parse_requirements_file_strips_version_specifiers(tmp_path: Path) -> None:
    path = _write(tmp_path, "requirements.txt", "numpy>=1.26,<2\npytest==8.0.0\n")
    assert parse_requirements_file(path) == ["numpy", "pytest"]


def test_parse_requirements_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, "requirements.txt", "# a comment\n\nnumpy  # inline comment\n   \n")
    assert parse_requirements_file(path) == ["numpy"]


def test_parse_requirements_file_skips_option_and_url_lines(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "requirements.txt",
        "-r base.txt\n--extra-index-url https://example.invalid\n"
        "git+https://example.invalid/pkg.git\nnumpy\n",
    )
    assert parse_requirements_file(path) == ["numpy"]


def test_parse_requirements_file_dedupes_case_insensitively(tmp_path: Path) -> None:
    path = _write(tmp_path, "requirements.txt", "NumPy\nnumpy>=1.26\n")
    assert parse_requirements_file(path) == ["NumPy"]


def test_generate_sbom_resolves_an_installed_package(tmp_path: Path) -> None:
    path = _write(tmp_path, "requirements.txt", "pytest\n")
    sbom = generate_sbom([path])

    assert sbom["bomFormat"] == "CycloneDX"
    assert len(sbom["components"]) == 1
    component = sbom["components"][0]
    assert component["name"].lower() == "pytest"
    assert component["version"]
    assert component["requirementsFile"] == str(path)
    assert sbom["unresolved"] == []


def test_generate_sbom_records_unresolved_packages_explicitly(tmp_path: Path) -> None:
    path = _write(tmp_path, "requirements.txt", "this-package-does-not-exist-anywhere\n")
    sbom = generate_sbom([path])

    assert sbom["components"] == []
    assert len(sbom["unresolved"]) == 1
    unresolved = sbom["unresolved"][0]
    assert unresolved["name"] == "this-package-does-not-exist-anywhere"
    assert unresolved["reason"] == "not installed in this environment"


def test_generate_sbom_scans_multiple_files_and_keeps_per_file_entries(tmp_path: Path) -> None:
    path_a = _write(tmp_path, "a.txt", "pytest\n")
    path_b = _write(tmp_path, "b.txt", "pytest\n")
    sbom = generate_sbom([path_a, path_b])

    assert len(sbom["components"]) == 2
    assert {c["requirementsFile"] for c in sbom["components"]} == {str(path_a), str(path_b)}


def test_generate_sbom_real_ultralytics_is_agpl_in_this_environment() -> None:
    """Ground-truth check (week 07's real finding, OPERATION-CARDS.md/ДН-03):
    confirms `ultralytics` is actually installed with `AGPL-3.0` in THIS
    environment, so `license_audit.py`'s tests build on a real fact, not
    an assumption."""
    import importlib.metadata as metadata_module

    dist = metadata_module.distribution("ultralytics")
    assert dist.metadata.get("License") == "AGPL-3.0"
