"""Combine the classical-CV benchmark and the CLIP baseline into one honest report (week 25).

The whole point of week 25 (`course/ROADMAP-WEEKS-25-30.md`) is a
trade-off table, not a verdict that classical CV "wins" -- `build_report`
reports the CLIP side as `"blocked"` rather than silently omitting it or
guessing a number when the real model can't be reached, same honesty
standard every week since week 15 holds a real-model network boundary to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from classical_cv.benchmark import ClassicalCVBenchReport

CLIP_STATUS_OK = "measured"
CLIP_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class ClipBaselineSummary:
    """CLIP side of the comparison -- either real measured numbers, or an honest `blocked`.

    Attributes:
        status: `"measured"` or `"blocked"`.
        accuracy: Fraction of images classified correctly (`ok` vs
            `defect`). `None` when `status="blocked"`.
        mean_latency_ms: Mean per-image `encode_image` wall-clock time.
            `None` when `status="blocked"`.
        blocked_reason: Human-readable reason, only set when `status="blocked"`.
    """

    status: str
    accuracy: float | None = None
    mean_latency_ms: float | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class ComparisonReport:
    """The week 25 deliverable: classical CV vs CLIP, side by side.

    Attributes:
        classical_cv: Real measured accuracy/latency, see `ClassicalCVBenchReport`.
        clip_baseline: Real measured accuracy/latency, or an honest `blocked`.
        recommendation: One human-readable sentence -- when classical CV is
            a reasonable substitute, not which approach is "better" in general.
    """

    classical_cv: ClassicalCVBenchReport
    clip_baseline: ClipBaselineSummary
    recommendation: str


def _recommend(classical: ClassicalCVBenchReport, clip: ClipBaselineSummary) -> str:
    """One sentence, grounded in this run's own numbers -- not a canned opinion."""
    op = classical.operating_point
    if clip.status == CLIP_STATUS_BLOCKED:
        return (
            f"CLIP baseline blocked ({clip.blocked_reason}); classical CV measured "
            f"precision={op['precision']:.2f} recall={op['recall']:.2f} at "
            f"{classical.mean_latency_ms:.2f} ms/image on CPU, no GPU -- a real number "
            "to weigh against a VLM approach once one is actually reachable, not a "
            "substitute for running that comparison for real."
        )
    speed_ratio = (
        clip.mean_latency_ms / classical.mean_latency_ms
        if clip.mean_latency_ms and classical.mean_latency_ms
        else float("nan")
    )
    return (
        f"Classical CV: precision={op['precision']:.2f} recall={op['recall']:.2f} at "
        f"{classical.mean_latency_ms:.2f} ms/image (CPU). CLIP zero-shot: "
        f"accuracy={clip.accuracy:.2f} at {clip.mean_latency_ms:.2f} ms/image -- "
        f"classical CV ran {speed_ratio:.1f}x faster in this run."
    )


def build_report(
    classical: ClassicalCVBenchReport, clip: ClipBaselineSummary
) -> ComparisonReport:
    """Assemble the final week 25 comparison report.

    Args:
        classical: Output of `classical_cv.benchmark.evaluate_classical_cv`.
        clip: Output of a successful or blocked CLIP baseline attempt.

    Returns:
        A `ComparisonReport` ready for `to_dict`/JSON serialization.
    """
    return ComparisonReport(
        classical_cv=classical, clip_baseline=clip, recommendation=_recommend(classical, clip)
    )


def to_dict(report: ComparisonReport) -> dict[str, object]:
    """JSON-serializable view of `report` (drops `per_image.candidates`' nested dataclasses
    down to plain dicts via `dataclasses.asdict`)."""
    return {
        "classical_cv": asdict(report.classical_cv),
        "clip_baseline": asdict(report.clip_baseline),
        "recommendation": report.recommendation,
    }
