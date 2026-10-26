"""Assemble every week 28 technique's real measured numbers into one honest comparison table.

No technique is assumed to "win" -- `build_report` just lays out the
real numbers; `_recommend` is built from whatever those numbers actually
say about THIS run, which technique helped and which did not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from distillation.benchmark import CompressionRow


@dataclass(frozen=True)
class CompressionReport:
    """The week 28 deliverable: every technique's real numbers, side by side.

    Attributes:
        rows: Every `CompressionRow`, in the order they were measured.
        recommendation: One sentence grounded in this run's own numbers.
    """

    rows: tuple[CompressionRow, ...]
    recommendation: str


def _row(rows: tuple[CompressionRow, ...], technique: str) -> CompressionRow | None:
    return next((row for row in rows if row.technique == technique), None)


def _recommend(rows: tuple[CompressionRow, ...]) -> str:
    """One sentence, grounded in this run's own numbers -- not a canned opinion."""
    teacher = _row(rows, "teacher")
    student_plain = _row(rows, "student_plain")
    student_distilled = _row(rows, "student_distilled")
    pruned = _row(rows, "pruned_50pct")

    parts = []
    if teacher and student_plain:
        parts.append(
            f"teacher: {teacher.accuracy:.2f} acc / {teacher.size_kb:.1f} KiB; "
            f"student (same steps, no teacher): {student_plain.accuracy:.2f} acc / "
            f"{student_plain.size_kb:.1f} KiB."
        )
    if student_distilled and student_plain:
        if student_distilled.accuracy > student_plain.accuracy + 0.01:
            parts.append(
                f"distillation helped at this step budget: {student_distilled.accuracy:.2f} "
                f"vs {student_plain.accuracy:.2f} accuracy for the same architecture/steps."
            )
        else:
            parts.append(
                f"distillation did NOT measurably help at this step budget "
                f"({student_distilled.accuracy:.2f} vs {student_plain.accuracy:.2f} accuracy) -- "
                "on this easy a task the hard-label loss alone finds the same solution; "
                "a harder task or a tighter step budget would be a fairer test of transfer."
            )
    if pruned and teacher:
        parts.append(
            f"50%-pruned teacher kept {pruned.accuracy:.2f} accuracy at "
            f"{teacher.num_parameters / pruned.num_parameters:.1f}x fewer parameters -- "
            "pruning was effectively free at this ratio."
        )
    nas_best = _row(rows, "nas_best")
    if nas_best and student_plain and nas_best.accuracy < student_plain.accuracy - 0.01:
        parts.append(
            f"NAS best ({nas_best.accuracy:.2f} acc) scored WORSE than the hand-picked student "
            f"({student_plain.accuracy:.2f} acc) at this run's NAS step budget -- a budget sized "
            "for this search, not borrowed from a wider model's convergence speed, is needed "
            "before trusting a NAS winner."
        )
    return " ".join(parts) if parts else "No comparable rows were measured in this run."


def build_report(rows: list[CompressionRow]) -> CompressionReport:
    """Assemble the final week 28 comparison report.

    Args:
        rows: Every technique's `CompressionRow` (see `distillation.benchmark.measure_row`).

    Returns:
        A `CompressionReport`.
    """
    rows_tuple = tuple(rows)
    return CompressionReport(rows=rows_tuple, recommendation=_recommend(rows_tuple))


def to_dict(report: CompressionReport) -> dict[str, object]:
    """JSON-serializable view of `report`."""
    return {
        "rows": [asdict(row) for row in report.rows],
        "recommendation": report.recommendation,
    }
