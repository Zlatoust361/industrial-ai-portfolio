#!/usr/bin/env python3
"""Stage 8: dataset-integrity validation -- bbox, empty masks, duplicates (week 06, day 3).

Runs after the full chain (`06_split_dataset.py`, `07_export_jetson_val_subset.py`)
so it validates what actually got shipped, not just the manifest. Three
manifest-level checks (`utils.validate.validate_dataset`): bbox validity
(in-bounds, non-degenerate), empty masks (a `"defect"` sample with no
defect boxes, or an `"ok"` sample carrying one -- both a contract
violation, not a legitimate state given `generate_defect_mask`'s
`count_range` always places >=1 shape), and duplicates (`sample_id`,
`image_path`, or a repeated bbox within one sample). Plus a file-level
cross-check against each split's actually-written YOLO directory (catches
conversion-time bugs a manifest-only check can't see -- a missing file,
a coordinate that drifted outside `[0, 1]`), and a self-consistency check
on the Jetson subset if stage 7 ran.

Exits `1` if any `"error"`-severity issue is found -- a real quality gate
for the dataset, not just an advisory report (same "fail loud, don't ship
broken data" stance as `pipeline/scripts/06_quality_gate.py`, week 04,
just scoped to dataset integrity instead of inspection verdicts).
Warnings (e.g. a duplicate bbox within one sample) are reported but don't
fail the gate.

Input: `<data_dir>/manifest_split.json` (stage 6), `<outputs_dir>/yolo/{train,val,test}/`
(stage 6), `<outputs_dir>/jetson_val_subset/` (stage 7, if it ran).
Output: `<outputs_dir>/validation_report.json`.

Usage:
    python3 scripts/08_validate_dataset.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.manifest import load_manifest  # noqa: E402
from utils.split import SPLITS  # noqa: E402
from utils.validate import (  # noqa: E402
    ValidationIssue,
    validate_dataset,
    validate_yolo_directory,
    validate_yolo_files,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `08_validate_dataset.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Validate the final manifest + written YOLO output; write a report; gate on errors."""
    args = parse_args()
    config = load_config(args.config)

    manifest_path = config.paths.data_dir / "manifest_split.json"
    try:
        samples = load_manifest(manifest_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 06_split_dataset.py first).", file=sys.stderr)
        sys.exit(1)

    report = validate_dataset(samples)
    issues: list[ValidationIssue] = list(report.issues)

    yolo_dir = config.paths.outputs_dir / "yolo"
    for split in SPLITS:
        split_dir = yolo_dir / split
        if not split_dir.is_dir():
            print(
                f"Note: '{split_dir}' not found, skipping (run 06_split_dataset.py).",
                file=sys.stderr,
            )
            continue
        split_samples = [sample for sample in samples if sample.split == split]
        issues.extend(validate_yolo_files(split_dir, split_samples))

    jetson_subset_dir = config.paths.outputs_dir / "jetson_val_subset"
    if jetson_subset_dir.is_dir():
        issues.extend(validate_yolo_directory(jetson_subset_dir))
    else:
        print(
            f"Note: '{jetson_subset_dir}' not found, skipping "
            "(run 07_export_jetson_val_subset.py).",
            file=sys.stderr,
        )

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    report_path = config.paths.outputs_dir / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "is_valid": not errors,
                "num_errors": len(errors),
                "num_warnings": len(warnings),
                "issues": [issue.to_dict() for issue in issues],
            },
            indent=2,
        )
    )

    print(
        f"Validated {len(samples)} sample(s): {len(errors)} error(s), {len(warnings)} warning(s)."
    )
    for issue in issues:
        print(f"  [{issue.severity.upper()}] {issue.category} ({issue.sample_id}): {issue.message}")
    print(f"Report written to '{report_path}'.")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
