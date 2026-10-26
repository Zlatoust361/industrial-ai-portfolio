"""License risk classification for SBOM components (week 21).

`classify_license` never pattern-matches inside a long block of text --
some installed packages (confirmed empirically against this
environment, e.g. `numpy==1.26.4`) put their ENTIRE license text into
the `License` metadata field, not a short identifier; substring-matching
across thousands of characters of legal boilerplate risks a false
positive (a disclaimer mentioning "General Public" in passing) far more
than it risks a missed real classification -- anything longer than
`_MAX_CLASSIFIABLE_LENGTH` is reported `UNKNOWN`, never guessed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

_MAX_CLASSIFIABLE_LENGTH = 200


class LicenseRisk(str, Enum):  # noqa: UP042 -- py3.10 compat, see src/config/environment.py::Environment
    """Coarse license risk category -- see `classify_license`."""

    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    COMMERCIAL_RESTRICTED = "commercial_restricted"
    UNKNOWN = "unknown"


# Hand-curated and deliberately small: SPDX identifiers AND the PyPI
# `License` free-text values this environment's ACTUAL installed
# packages use (confirmed empirically, see tests/test_license_audit.py
# and compliance/README.md) -- not a guess at every license in
# existence. Matched case-insensitively.
_PERMISSIVE = {
    "mit",
    "mit-cmu",
    "mit-0",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "apache-2.0",
    "isc",
    "python-2.0",
    "python-2.0.1",
    "bsl-1.0",
    "0bsd",
    "unlicense",
    "hpnd",
    "zlib",
    "apache 2.0 license",
    "apache 2.0",
    "apache license 2.0",
    "apache software license",
    "mit license",
    "bsd license",
}
_COPYLEFT = {
    "gpl-2.0",
    "gpl-3.0",
    "gpl-2.0-only",
    "gpl-3.0-only",
    "gpl-2.0-or-later",
    "gpl-3.0-or-later",
    "agpl-3.0",
    "agpl-3.0-only",
    "agpl-3.0-or-later",
    "lgpl-2.1",
    "lgpl-3.0",
    "lgpl-2.1-only",
    "lgpl-3.0-only",
    "lgpl-2.1-or-later",
    "lgpl-3.0-or-later",
}
_COMMERCIAL_RESTRICTED = {
    "busl-1.1",
    "sspl-1.0",
    "elastic-2.0",
    "commons-clause",
}

# "Classifier: License :: ..." trove strings aren't SPDX ids -- matched
# by substring against PyPI's own canonical trove-classifier wording.
_CLASSIFIER_COPYLEFT_MARKERS = (
    "affero",
    "gnu general public license",
    "gnu lesser general public license",
)
_CLASSIFIER_PERMISSIVE_MARKERS = (
    "apache software license",
    "mit license",
    "bsd license",
    "python software foundation license",
    "isc license",
)


def classify_license(license_name: str | None) -> LicenseRisk:
    """Classify one license string into a coarse risk category.

    Args:
        license_name: A license identifier/expression/classifier
            string, as `sbom._resolve_license` returns it. Handles a
            PEP 639 combined expression (e.g. torch's installed
            `"Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND ... AND MIT"`)
            by classifying every `AND`/`OR`-separated operand and
            returning the MOST restrictive risk among them -- a package
            under several license terms is only as permissive as its
            most restrictive applicable one. An SPDX `WITH <exception>`
            clause (e.g. `"Apache-2.0 WITH LLVM-exception"`) is reduced
            to its base license before classifying -- an exception only
            GRANTS additional permission, it never adds a restriction,
            so it can't turn a permissive base license into something
            riskier.

    Returns:
        `LicenseRisk.UNKNOWN` for `None`/blank input, or text longer
        than `_MAX_CLASSIFIABLE_LENGTH` (full license text, not a short
        identifier -- see this module's docstring). Otherwise one of
        `PERMISSIVE`/`COPYLEFT`/`COMMERCIAL_RESTRICTED`/`UNKNOWN` -- the
        function NEVER guesses an unrecognized string into a risk
        category.
    """
    if not license_name or not license_name.strip():
        return LicenseRisk.UNKNOWN
    normalized = license_name.strip()
    if len(normalized) > _MAX_CLASSIFIABLE_LENGTH:
        return LicenseRisk.UNKNOWN

    tokens = re.split(r"\s+(?:AND|OR)\s+", normalized)
    token_risks = {_classify_token(token) for token in tokens}

    for risk in (LicenseRisk.COMMERCIAL_RESTRICTED, LicenseRisk.COPYLEFT):
        if risk in token_risks:
            return risk
    if token_risks == {LicenseRisk.PERMISSIVE}:
        return LicenseRisk.PERMISSIVE
    return LicenseRisk.UNKNOWN


def _classify_token(token: str) -> LicenseRisk:
    """Classify one AND/OR-split operand of a license expression."""
    stripped = token.strip().strip("()").strip()
    # An SPDX "WITH <exception>" clause only grants extra permission on
    # top of the base license -- classify the base, ignore the exception.
    stripped = re.split(r"\s+WITH\s+", stripped)[0].strip()
    lowered = stripped.lower()

    if lowered in _COMMERCIAL_RESTRICTED:
        return LicenseRisk.COMMERCIAL_RESTRICTED
    if lowered in _COPYLEFT:
        return LicenseRisk.COPYLEFT
    if lowered in _PERMISSIVE:
        return LicenseRisk.PERMISSIVE

    if lowered.startswith("license"):  # a "Classifier: License :: ..." trove string
        if any(marker in lowered for marker in _CLASSIFIER_COPYLEFT_MARKERS):
            return LicenseRisk.COPYLEFT
        if any(marker in lowered for marker in _CLASSIFIER_PERMISSIVE_MARKERS):
            return LicenseRisk.PERMISSIVE

    return LicenseRisk.UNKNOWN


@dataclass(frozen=True)
class LicenseAuditEntry:
    """One SBOM component's classified license risk.

    Attributes:
        name: Package name.
        version: Installed version.
        license_name: The raw license string `classify_license` was
            given (`None` if unresolvable).
        risk: `classify_license(license_name)`'s result.
        requirements_file: Which `requirements*.txt` this entry came from.
    """

    name: str
    version: str
    license_name: str | None
    risk: LicenseRisk
    requirements_file: str


@dataclass(frozen=True)
class LicenseAuditReport:
    """Every SBOM component's classified risk, plus aggregate counts."""

    entries: tuple[LicenseAuditEntry, ...]
    risk_counts: dict[str, int]

    def entries_with_risk(self, risk: LicenseRisk) -> tuple[LicenseAuditEntry, ...]:
        """Entries classified as `risk`, in SBOM order."""
        return tuple(entry for entry in self.entries if entry.risk == risk)


def audit_sbom(sbom: dict[str, object]) -> LicenseAuditReport:
    """Classify every component in an `sbom.generate_sbom` result.

    Args:
        sbom: An SBOM dict from `sbom.generate_sbom`.

    Returns:
        A `LicenseAuditReport` -- one `LicenseAuditEntry` per
        `sbom["components"]` entry, plus aggregate counts per risk
        category (all four category keys present, `0` if unused).
    """
    entries = []
    for component in sbom.get("components", []):  # type: ignore[union-attr]
        license_name = component.get("license")
        risk = classify_license(license_name if isinstance(license_name, str) else None)
        entries.append(
            LicenseAuditEntry(
                name=str(component["name"]),
                version=str(component["version"]),
                license_name=license_name if isinstance(license_name, str) else None,
                risk=risk,
                requirements_file=str(component["requirementsFile"]),
            )
        )
    counts = {risk.value: 0 for risk in LicenseRisk}
    counts.update(Counter(entry.risk.value for entry in entries))
    return LicenseAuditReport(entries=tuple(entries), risk_counts=counts)
