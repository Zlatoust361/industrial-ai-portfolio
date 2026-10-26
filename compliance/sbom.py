"""Software bill of materials (SBOM) generation for this repository (week 21).

CycloneDX-shaped JSON built from this repo's `requirements*.txt` files,
resolved against ACTUALLY INSTALLED package metadata in the current
Python environment via stdlib `importlib.metadata` -- the same
information a `pip-licenses`-style tool reads, reimplemented minimally
rather than adding a new runtime dependency for one report. A
requirement that isn't installed in THIS environment (e.g. a
JetPack-only package from `requirements-jetson.txt` this dev sandbox
never installs) is recorded as `unresolved`, never guessed.
"""

from __future__ import annotations

import re
from pathlib import Path

_REQUIREMENT_NAME_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def parse_requirements_file(path: Path) -> list[str]:
    """Extract package names from one `requirements*.txt` file.

    Args:
        path: Path to a pip requirements file.

    Returns:
        Package names in file order, de-duplicated by first occurrence
        (case-insensitively). Blank lines, comments (`#`), option lines
        (`-r`/`--...`), and URL/VCS requirements are skipped -- nothing
        to resolve a license for from a bare name.
    """
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "://" in line:
            continue
        match = _REQUIREMENT_NAME_PATTERN.match(line)
        if not match:
            continue
        name = match.group(1)
        normalized = name.lower()
        if normalized not in seen:
            seen.add(normalized)
            names.append(name)
    return names


def _resolve_license(metadata: object) -> str | None:
    """Best-known license string for one distribution's metadata.

    Preference order: `License-Expression` (PEP 639 -- a real SPDX
    expression when present, e.g. `torch`'s installed metadata in this
    environment) > `License` field (free text or a short identifier,
    e.g. `transformers`'s `"Apache 2.0 License"`) > the first
    `Classifier: License :: ...` trove entry. `None` if none of these
    are present -- `license_audit.classify_license` turns that into
    `UNKNOWN`, this function never guesses.
    """
    license_expression = metadata.get("License-Expression")
    if license_expression:
        return str(license_expression)
    license_field = metadata.get("License")
    if license_field:
        return str(license_field)
    for classifier in metadata.get_all("Classifier") or []:
        if classifier.startswith("License"):
            return str(classifier)
    return None


def generate_sbom(requirements_files: list[Path]) -> dict[str, object]:
    """Build a CycloneDX-shaped SBOM from this repo's requirements files.

    Args:
        requirements_files: Paths to `requirements*.txt` files to scan.

    Returns:
        A dict matching CycloneDX 1.5's top-level shape:
        `{"bomFormat": "CycloneDX", "specVersion": "1.5", "components": [...],
        "unresolved": [...]}`. One `components` entry per `(requirements
        file, package name)` pair that resolves to an installed
        distribution in the CURRENT environment (deliberately per-file,
        not globally de-duplicated -- which file brought a dependency is
        itself part of the audit trail). `unresolved` lists names that
        don't resolve, each with a `"reason"` -- never silently dropped.
    """
    import importlib.metadata as metadata_module

    components: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []

    for req_path in requirements_files:
        for name in parse_requirements_file(req_path):
            try:
                dist = metadata_module.distribution(name)
            except metadata_module.PackageNotFoundError:
                unresolved.append(
                    {
                        "name": name,
                        "requirementsFile": str(req_path),
                        "reason": "not installed in this environment",
                    }
                )
                continue
            components.append(
                {
                    "type": "library",
                    "name": dist.metadata["Name"] or name,
                    "version": dist.version,
                    "license": _resolve_license(dist.metadata),
                    "requirementsFile": str(req_path),
                }
            )

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": components,
        "unresolved": unresolved,
    }
