"""Turns raw per-image detection scores into ok/defect verdicts + a summary.

Pure logic over the `detections` list `04_detect.py` writes to
`outputs/detections.json` -- no torch/model dependency, so `05_interpret.py`'s
actual decision logic (as opposed to `04_detect.py`'s model-running glue) is
fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageVerdict:
    """One image's ok/defect call.

    Attributes:
        image: Path of the image this verdict is about (as recorded in `detections.json`).
        defect_score: The raw score `04_detect.py` produced for this image.
        verdict: `"ok"` or `"defect"`.
    """

    image: str
    defect_score: float
    verdict: str

    def to_dict(self) -> dict[str, object]:
        """This verdict as a plain JSON-serializable dict."""
        return {"image": self.image, "defect_score": self.defect_score, "verdict": self.verdict}


def interpret_detections(
    detections: list[dict[str, object]], defect_score_threshold: float
) -> tuple[list[ImageVerdict], dict[str, object]]:
    """Classify every detection entry as ok/defect and summarize the batch.

    Args:
        detections: The `detections` list from `outputs/detections.json`;
            each entry needs an `"image"` (str) and a `"defect_score"`
            (float in `[0, 1]`) key. Backend-specific extra keys (e.g.
            CLIP's `"similarities"`, YOLO's `"boxes"`) are ignored here.
        defect_score_threshold: A `defect_score` at or above this makes the
            verdict `"defect"`.

    Returns:
        `(verdicts, summary)`, one `ImageVerdict` per detection entry (same
        order) and `summary = {"total", "defect_count", "ok_count",
        "defect_rate"}`. `defect_rate` is `0.0` for an empty `detections`
        list, not a division-by-zero error.

    Raises:
        ValueError: If `defect_score_threshold` isn't in `[0, 1]`.
    """
    if not 0.0 <= defect_score_threshold <= 1.0:
        raise ValueError(
            f"defect_score_threshold must be in [0, 1], got {defect_score_threshold}."
        )

    verdicts: list[ImageVerdict] = []
    for entry in detections:
        score = float(entry["defect_score"])  # type: ignore[arg-type]
        verdict = "defect" if score >= defect_score_threshold else "ok"
        verdicts.append(
            ImageVerdict(image=str(entry["image"]), defect_score=score, verdict=verdict)
        )

    total = len(verdicts)
    defect_count = sum(1 for v in verdicts if v.verdict == "defect")
    summary = {
        "total": total,
        "defect_count": defect_count,
        "ok_count": total - defect_count,
        "defect_rate": (defect_count / total) if total > 0 else 0.0,
    }
    return verdicts, summary
