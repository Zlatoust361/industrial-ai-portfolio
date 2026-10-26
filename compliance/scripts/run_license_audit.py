#!/usr/bin/env python3
"""Week 21's whole non-interactive deliverable, one command.

Generates a real SBOM from this repository's `requirements*.txt` files
(resolved against ACTUALLY INSTALLED package metadata, see
`compliance/sbom.py`), classifies every resolved component's license
risk (`compliance/license_audit.py`) -- confirming that
`ultralytics`/YOLOv8's `AGPL-3.0` (the week 07 finding documented in
`course/OPERATION-CARDS.md`'s `ДН-03`) is DETECTED by this automatic
run, not just known from the prior manual finding -- renders the ИБ
acceptance checklist (`compliance/checklist.py`), and runs the one
adversarial-patch case + defense (`compliance/adversarial_demo.py`).

Usage:
    PYTHONPATH=. python3 compliance/scripts/run_license_audit.py \\
        [--requirements-file requirements.txt ...] [--output-dir outputs/compliance_demo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_COMPLIANCE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_COMPLIANCE_ROOT.parent))

from compliance.adversarial_demo import run_adversarial_patch_demo  # noqa: E402
from compliance.checklist import (  # noqa: E402
    DEFAULT_CHECKLIST,
    checklist_summary,
    render_checklist_markdown,
)
from compliance.license_audit import audit_sbom  # noqa: E402
from compliance.sbom import generate_sbom  # noqa: E402

DEFAULT_REQUIREMENTS_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-jetson.txt",
    "requirements-chat-jetson.txt",
)
DEFAULT_OUTPUT_DIR = "outputs/compliance_demo"


def run_full_audit(requirements_files: list[Path]) -> dict[str, object]:
    """Run SBOM -> license audit -> checklist -> adversarial demo, return one report dict.

    Args:
        requirements_files: Which `requirements*.txt` files to scan.

    Returns:
        A JSON-serializable report with `"sbom"`, `"license_audit"`
        (risk counts + every entry), `"checklist"` (summary + full
        table), and `"adversarial_demo"` keys.
    """
    sbom = generate_sbom(requirements_files)
    audit_report = audit_sbom(sbom)

    license_entries = [
        {
            "name": entry.name,
            "version": entry.version,
            "license": entry.license_name,
            "risk": entry.risk.value,
            "requirementsFile": entry.requirements_file,
        }
        for entry in audit_report.entries
    ]

    adversarial_result = run_adversarial_patch_demo()

    return {
        "sbom": sbom,
        "license_audit": {
            "risk_counts": audit_report.risk_counts,
            "entries": license_entries,
        },
        "checklist": {
            "summary": checklist_summary(),
            "items": [
                {
                    "itemId": item.item_id,
                    "description": item.description,
                    "status": item.status.value,
                    "justification": item.justification,
                    "source": item.source,
                }
                for item in DEFAULT_CHECKLIST
            ],
        },
        "adversarial_demo": {
            "numFrames": adversarial_result.num_frames,
            "rawVerdicts": list(adversarial_result.raw_verdicts),
            "rawFalseNegativeRate": adversarial_result.raw_false_negative_rate,
            "smoothedFinalVerdict": adversarial_result.smoothed_final_verdict,
            "defenseRecoveredCorrectVerdict": adversarial_result.defense_recovered_correct_verdict,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_license_audit.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--requirements-file",
        dest="requirements_files",
        action="append",
        default=None,
        help="Repeatable. Defaults to every requirements*.txt this repo ships.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the full audit, write the report + checklist, print the headline numbers."""
    args = parse_args()
    requirements_files = [Path(p) for p in (args.requirements_files or DEFAULT_REQUIREMENTS_FILES)]
    report = run_full_audit(requirements_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "checklist.md").write_text(render_checklist_markdown(), encoding="utf-8")

    sbom = report["sbom"]
    print(
        f"SBOM: {len(sbom['components'])} resolved component(s), "
        f"{len(sbom['unresolved'])} unresolved."
    )
    print(f"License risk counts: {report['license_audit']['risk_counts']}")

    ultralytics_entries = [
        entry
        for entry in report["license_audit"]["entries"]
        if entry["name"].lower() == "ultralytics"
    ]
    if ultralytics_entries:
        entry = ultralytics_entries[0]
        print(
            f"ultralytics: license={entry['license']!r}, risk={entry['risk']!r} "
            f"(expected 'copyleft' -- the week 07 AGPL-3.0 finding, OPERATION-CARDS.md/ДН-03)"
        )
    else:
        print(
            "ultralytics: NOT FOUND in this environment's resolved components "
            "(not installed, or not listed in the scanned requirements files)."
        )

    print(f"ИБ checklist: {report['checklist']['summary']}")

    demo = report["adversarial_demo"]
    defense_outcome = "RECOVERED" if demo["defenseRecoveredCorrectVerdict"] else "DID NOT recover"
    print(
        f"Adversarial patch demo: raw false-negative rate={demo['rawFalseNegativeRate']:.2f}, "
        f"smoothed final verdict={demo['smoothedFinalVerdict']!r} "
        f"(defense {defense_outcome} the correct verdict)"
    )

    print(f"Report -> {output_dir / 'report.json'}")
    print(f"Checklist -> {output_dir / 'checklist.md'}")


if __name__ == "__main__":
    main()
