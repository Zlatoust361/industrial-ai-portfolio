"""GREEN/YELLOW/RED quality gate over the interpret stage's `defect_rate`.

Pure logic, no I/O -- `06_quality_gate.py` reads `outputs/interpretations.json`
and writes the `QualityGateResult` below to `outputs/metrics/quality_gate.json`;
the CI gate wired up day 3 reads that file (or this script's exit code, see
`06_quality_gate.py`'s docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

STATUSES: tuple[str, ...] = ("GREEN", "YELLOW", "RED")


@dataclass(frozen=True)
class QualityGateResult:
    """The quality-gate verdict for one pipeline run.

    Attributes:
        status: One of `STATUSES`.
        defect_rate: The `defect_rate` this verdict was computed from.
        green_max_defect_rate: The threshold below which `status` is `"GREEN"`.
        yellow_max_defect_rate: The threshold below which (and not GREEN)
            `status` is `"YELLOW"`; at or above it, `"RED"`.
    """

    status: str
    defect_rate: float
    green_max_defect_rate: float
    yellow_max_defect_rate: float

    def to_dict(self) -> dict[str, object]:
        """This result as a plain JSON-serializable dict."""
        return {
            "status": self.status,
            "defect_rate": self.defect_rate,
            "thresholds": {
                "green_max_defect_rate": self.green_max_defect_rate,
                "yellow_max_defect_rate": self.yellow_max_defect_rate,
            },
        }


def evaluate_quality_gate(
    defect_rate: float, green_max_defect_rate: float, yellow_max_defect_rate: float
) -> QualityGateResult:
    """Classify a `defect_rate` into GREEN/YELLOW/RED against two thresholds.

    Args:
        defect_rate: Fraction of images flagged as `"defect"`, in `[0, 1]`.
        green_max_defect_rate: `defect_rate` strictly below this -> GREEN.
        yellow_max_defect_rate: `defect_rate` strictly below this (and not
            GREEN) -> YELLOW; at or above it -> RED.

    Returns:
        A `QualityGateResult` with the computed `status`.

    Raises:
        ValueError: If `defect_rate` isn't in `[0, 1]`, or
            `green_max_defect_rate >= yellow_max_defect_rate`.
    """
    if not 0.0 <= defect_rate <= 1.0:
        raise ValueError(f"defect_rate must be in [0, 1], got {defect_rate}.")
    if not green_max_defect_rate < yellow_max_defect_rate:
        raise ValueError(
            f"green_max_defect_rate ({green_max_defect_rate}) must be < "
            f"yellow_max_defect_rate ({yellow_max_defect_rate})."
        )

    if defect_rate < green_max_defect_rate:
        status = "GREEN"
    elif defect_rate < yellow_max_defect_rate:
        status = "YELLOW"
    else:
        status = "RED"

    return QualityGateResult(
        status=status,
        defect_rate=defect_rate,
        green_max_defect_rate=green_max_defect_rate,
        yellow_max_defect_rate=yellow_max_defect_rate,
    )
