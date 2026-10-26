"""Dataset-integrity validation (week 06, stage 8).

Three checks, as named in the day-3 prompt: bbox validity (in-bounds,
non-degenerate), empty masks (a `"defect"` sample with no defect boxes,
or an `"ok"` sample carrying one -- both are a contract violation, not a
legitimate dataset state given `generate_defect_mask`'s `count_range`
always places >=1 shape), and duplicates (`sample_id`, `image_path`, or
an exact repeated bbox within one sample's `defect_boxes`). A second
layer (`validate_yolo_files`) cross-checks an actually-written YOLO
directory against the manifest -- catches conversion-time bugs (a missing
label/image file, a normalized coordinate that drifted outside `[0, 1]`)
that a manifest-only check can't see.

Pure logic, no torch -- same "real integrity gate" spirit as
`pipeline/utils/quality_gate.py` (week 04), scoped to dataset integrity
instead of inspection verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.manifest import Bbox, Sample

Severity = str  # "error" | "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """One integrity problem found in the dataset.

    Attributes:
        severity: `"error"` (the dataset is broken/unsafe to train on as
            shipped) or `"warning"` (suspicious, worth a human look, not
            necessarily wrong).
        category: Short machine-readable tag, e.g. `"bbox_out_of_bounds"`,
            `"empty_defect_mask"`, `"duplicate_sample_id"`.
        sample_id: The sample the issue was found on (or the first of a
            duplicate pair/group).
        message: Human-readable description.
    """

    severity: Severity
    category: str
    sample_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """This issue as a plain JSON-serializable dict."""
        return {
            "severity": self.severity,
            "category": self.category,
            "sample_id": self.sample_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    """The full set of issues found across a dataset.

    Attributes:
        issues: Every `ValidationIssue` found, in check order.
    """

    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Only the `"error"`-severity issues."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Only the `"warning"`-severity issues."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """`True` iff there are no `"error"`-severity issues (warnings don't fail the gate)."""
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """This report as a plain JSON-serializable dict."""
        return {
            "is_valid": self.is_valid,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_bbox(box: Bbox, width: int, height: int) -> str | None:
    """Check one box for degeneracy/out-of-bounds.

    Args:
        box: `(x_min, y_min, x_max, y_max)`.
        width: The box's image's width.
        height: The box's image's height.

    Returns:
        A human-readable problem description, or `None` if the box is valid.
    """
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return f"degenerate box {box}: x_max/y_max must be strictly greater than x_min/y_min"
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return f"box {box} out of bounds for a {width}x{height} image"
    return None


def find_bbox_issues(samples: list[Sample]) -> list[ValidationIssue]:
    """Check every sample's `object_bbox` and `defect_boxes` for degeneracy/out-of-bounds.

    Args:
        samples: Samples to check.

    Returns:
        One `"error"` `ValidationIssue` (category `"bad_bbox"`) per invalid box found.
    """
    issues: list[ValidationIssue] = []
    for sample in samples:
        problem = validate_bbox(sample.object_bbox, sample.width, sample.height)
        if problem is not None:
            issues.append(
                ValidationIssue("error", "bad_bbox", sample.sample_id, f"object_bbox: {problem}")
            )
        for defect_box in sample.defect_boxes:
            problem = validate_bbox(defect_box, sample.width, sample.height)
            if problem is not None:
                issues.append(
                    ValidationIssue(
                        "error", "bad_bbox", sample.sample_id, f"defect_boxes: {problem}"
                    )
                )
    return issues


def find_empty_mask_issues(samples: list[Sample]) -> list[ValidationIssue]:
    """Check `label`/`defect_boxes` consistency.

    Args:
        samples: Samples to check.

    Returns:
        One `"error"` `ValidationIssue` per sample where `label == "defect"`
        but `defect_boxes` is empty (category `"empty_defect_mask"`), or
        `label == "ok"` but `defect_boxes` is non-empty (category
        `"unexpected_defect_boxes"`).
    """
    issues: list[ValidationIssue] = []
    for sample in samples:
        if sample.label == "defect" and not sample.defect_boxes:
            issues.append(
                ValidationIssue(
                    "error",
                    "empty_defect_mask",
                    sample.sample_id,
                    "labeled 'defect' but has zero defect_boxes",
                )
            )
        elif sample.label == "ok" and sample.defect_boxes:
            issues.append(
                ValidationIssue(
                    "error",
                    "unexpected_defect_boxes",
                    sample.sample_id,
                    f"labeled 'ok' but has {len(sample.defect_boxes)} defect_boxes",
                )
            )
    return issues


def find_duplicate_issues(samples: list[Sample]) -> list[ValidationIssue]:
    """Check for duplicate `sample_id`s, duplicate `image_path`s, and repeated boxes.

    Args:
        samples: Samples to check.

    Returns:
        `"error"` issues for a duplicate `sample_id` (category
        `"duplicate_sample_id"`) or `image_path` (category
        `"duplicate_image_path"`) -- either would silently collide on
        disk or corrupt id-keyed lookups; `"warning"` issues (category
        `"duplicate_bbox"`) for the exact same box appearing twice in one
        sample's `defect_boxes` -- suspicious (a real generation/
        augmentation bug pattern) but not on its own unsafe to train on.
    """
    issues: list[ValidationIssue] = []

    seen_sample_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_sample_id",
                    sample.sample_id,
                    f"sample_id {sample.sample_id!r} appears more than once",
                )
            )
        seen_sample_ids.add(sample.sample_id)

        if sample.image_path in seen_image_paths:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_image_path",
                    sample.sample_id,
                    f"image_path {sample.image_path!r} is reused by another sample",
                )
            )
        seen_image_paths.add(sample.image_path)

        if len(set(sample.defect_boxes)) != len(sample.defect_boxes):
            issues.append(
                ValidationIssue(
                    "warning",
                    "duplicate_bbox",
                    sample.sample_id,
                    "defect_boxes contains an exact duplicate box",
                )
            )

    return issues


def validate_dataset(samples: list[Sample]) -> ValidationReport:
    """Run every manifest-level check and collect the results.

    Args:
        samples: The dataset to validate (typically the final,
            most-complete manifest -- `manifest_split.json`).

    Returns:
        A `ValidationReport` covering bbox validity, empty-mask
        consistency, and duplicates.
    """
    issues = [
        *find_bbox_issues(samples),
        *find_empty_mask_issues(samples),
        *find_duplicate_issues(samples),
    ]
    return ValidationReport(tuple(issues))


def _validate_label_file(label_path: Path, sample_id: str) -> list[ValidationIssue]:
    """Check every line of a YOLO label file: 5 fields, `cx/cy/w/h` in `[0, 1]`.

    Args:
        label_path: Path to a `labels/*.txt` file.
        sample_id: Identifier to attach to any issue found (the filename
            stem when there's no manifest sample to name).

    Returns:
        `"error"` issues (category `"bad_yolo_label_line"`) for any
        malformed line; empty if every line is well-formed.
    """
    issues: list[ValidationIssue] = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                ValidationIssue(
                    "error",
                    "bad_yolo_label_line",
                    sample_id,
                    f"{label_path}:{line_number}: expected 5 fields, got {len(parts)}",
                )
            )
            continue
        _, cx, cy, w, h = parts
        if not all(0.0 <= float(value) <= 1.0 for value in (cx, cy, w, h)):
            issues.append(
                ValidationIssue(
                    "error",
                    "bad_yolo_label_line",
                    sample_id,
                    f"{label_path}:{line_number}: cx/cy/w/h must be in [0, 1], "
                    f"got {line.strip()!r}",
                )
            )
    return issues


def validate_yolo_files(yolo_dir: str | Path, samples: list[Sample]) -> list[ValidationIssue]:
    """Cross-check an actually-written YOLO directory against `samples`.

    Catches conversion-time bugs a manifest-only check can't see: a
    missing image/label file, or a label line whose normalized
    coordinates drifted outside `[0, 1]`.

    Args:
        yolo_dir: A YOLO root (`images/`, `labels/`, `classes.txt`) --
            either the flat stage-5 output or one of stage 6's per-split
            subdirectories.
        samples: The samples expected to be in `yolo_dir` (matched by filename).

    Returns:
        `"error"` issues (category `"missing_yolo_file"` or
        `"bad_yolo_label_line"`) for anything wrong; empty if `yolo_dir`
        matches `samples` exactly.
    """
    yolo_dir = Path(yolo_dir)
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    issues: list[ValidationIssue] = []

    for sample in samples:
        filename = Path(sample.image_path).name
        stem = Path(filename).stem
        image_path = images_dir / filename
        label_path = labels_dir / f"{stem}.txt"

        if not image_path.is_file():
            issues.append(
                ValidationIssue(
                    "error", "missing_yolo_file", sample.sample_id, f"missing image: {image_path}"
                )
            )
        if not label_path.is_file():
            issues.append(
                ValidationIssue(
                    "error", "missing_yolo_file", sample.sample_id, f"missing label: {label_path}"
                )
            )
            continue

        issues.extend(_validate_label_file(label_path, sample.sample_id))

    return issues


def validate_yolo_directory(yolo_dir: str | Path) -> list[ValidationIssue]:
    """Self-consistency check on a YOLO directory, without a manifest.

    Checks that every image has a matching label file and vice versa,
    and that every label line parses. Useful where there's no sample list
    to check against -- stage 7's `jetson_val_subset/` is a *strict
    subset* of the val split, so `validate_yolo_files` (which expects
    every listed sample to be present) can't be used against it directly
    without re-deriving exactly which val samples were selected.

    Args:
        yolo_dir: A YOLO root (`images/`, `labels/`).

    Returns:
        `"error"` issues (category `"missing_yolo_file"` or
        `"bad_yolo_label_line"`) for any mismatch/malformed line.
    """
    yolo_dir = Path(yolo_dir)
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    issues: list[ValidationIssue] = []

    image_stems = {path.stem for path in images_dir.glob("*") if path.is_file()}
    label_stems = {path.stem for path in labels_dir.glob("*.txt") if path.is_file()}

    for stem in sorted(image_stems - label_stems):
        issues.append(
            ValidationIssue(
                "error", "missing_yolo_file", stem, f"image has no matching label: {stem}"
            )
        )
    for stem in sorted(label_stems - image_stems):
        issues.append(
            ValidationIssue(
                "error", "missing_yolo_file", stem, f"label has no matching image: {stem}"
            )
        )
    for stem in sorted(image_stems & label_stems):
        issues.extend(_validate_label_file(labels_dir / f"{stem}.txt", stem))

    return issues
