"""One adversarial-patch case + one defense, measured before/after (week 21).

Not a research-level attack -- the level `ОБ-04`/приёмка actually checks
for: can a simple PHYSICAL occlusion (a sticker covering part of a
visible defect) fool the real detector on individual frames, and does a
cheap defense that's already in this repository
(`tracking.smoothing.VerdictSmoother`, week 13) measurably help. This
module documents ONE such case with a measured effect, not an attempt to
close every possible attack (see this module's and
`compliance/README.md`'s Honesty note for the defense's real limit).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.utils.interpret import interpret_detections
from tracking.smoothing import VerdictSmoother

_PIPELINE_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "scripts"

_BACKGROUND_RGB = (50, 50, 50)
_IMAGE_SIZE = 64
_MARKER_BOX = (10, 10, 30, 30)  # (row0, col0, row1, col1), exclusive
DEFECT_SCORE_THRESHOLD = 0.5

# Deterministic per-frame patch coverage -- NOT random: simulates a
# physical sticker whose alignment wobbles frame to frame (vibration,
# parallax as the part moves under the camera), not a sticker that
# perfectly, consistently covers the defect every single frame. 2 of
# these 5 coverage values (0.95, 1.00) hide ENOUGH of the real marker to
# evade `StubDetector` on that frame alone -- a real, measured per-frame
# vulnerability. An attacker who reliably covered 3+ of 5 frames would
# also beat the window-5 majority-vote defense below (see this module's
# docstring and `compliance/README.md`'s Honesty note) -- that limit is
# the point, not something this module hides.
DEFAULT_PATCH_COVERAGES: tuple[float, ...] = (0.70, 0.95, 1.00, 0.80, 0.70)

_stub_detector_class: type | None = None


def _get_stub_detector_class() -> type:
    """Dynamically load `pipeline/scripts/04_detect.py`'s `StubDetector`, once.

    `04_detect.py` can't be imported with a normal `import` statement (a
    leading digit isn't a valid Python identifier) -- this loads the
    REAL file by path, the same helper every week's `tests/conftest.py::
    load_script_module` already uses, so this demo runs against week
    04's ACTUAL detection algorithm, not a reimplementation of it.
    """
    global _stub_detector_class
    if _stub_detector_class is None:
        spec = importlib.util.spec_from_file_location(
            "compliance_pipeline_04_detect", _PIPELINE_SCRIPTS_DIR / "04_detect.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(_PIPELINE_SCRIPTS_DIR.parent))
        spec.loader.exec_module(module)
        _stub_detector_class = module.StubDetector
    return _stub_detector_class


def _make_patched_frame(coverage: float) -> Image.Image:
    """Render one frame: the real `StubDetector` marker, patched by `coverage`.

    Args:
        coverage: Fraction (`[0, 1]`) of the marker rectangle's rows
            covered by a background-colored "sticker", from the top.
            `0.0` = marker fully visible (no attack this frame); `1.0` =
            marker fully hidden.

    Returns:
        A `64x64` RGB image.
    """
    stub_detector_class = _get_stub_detector_class()
    array = np.full((_IMAGE_SIZE, _IMAGE_SIZE, 3), _BACKGROUND_RGB, dtype=np.uint8)
    row0, col0, row1, col1 = _MARKER_BOX
    # Reads the real detector's own marker color (not a duplicated
    # constant that could silently drift from it).
    array[row0:row1, col0:col1] = stub_detector_class._MARKER_RGB
    covered_rows = round((row1 - row0) * coverage)
    array[row0 : row0 + covered_rows, col0:col1] = _BACKGROUND_RGB
    return Image.fromarray(array, mode="RGB")


@dataclass(frozen=True)
class AdversarialDemoResult:
    """One run's measured before/after effect of the temporal-smoothing defense.

    Attributes:
        num_frames: How many frames of the same tracked (genuinely
            defective) object were simulated.
        raw_verdicts: Per-frame verdict `StubDetector` + week 04's
            threshold rule actually produced, in frame order.
        raw_false_negative_rate: Fraction of `raw_verdicts` that are
            `"ok"` despite the object being genuinely defective -- the
            attack's measured per-frame success rate, BEFORE any defense.
        smoothed_final_verdict: `VerdictSmoother`'s majority-vote
            verdict after all `num_frames` frames, AFTER the defense.
        defense_recovered_correct_verdict: Whether
            `smoothed_final_verdict == "defect"` (the true label) -- the
            actual measured payoff, not just an assertion that smoothing
            "should" help.
    """

    num_frames: int
    raw_verdicts: tuple[str, ...]
    raw_false_negative_rate: float
    smoothed_final_verdict: str
    defense_recovered_correct_verdict: bool


def run_adversarial_patch_demo(
    coverages: tuple[float, ...] = DEFAULT_PATCH_COVERAGES,
    window_size: int | None = None,
) -> AdversarialDemoResult:
    """Run the full before/after adversarial-patch demonstration.

    Args:
        coverages: Per-frame patch coverage fractions (see
            `DEFAULT_PATCH_COVERAGES`'s docstring for the scenario the
            default simulates). Must be non-empty.
        window_size: `VerdictSmoother`'s rolling window. `None` (default)
            uses `len(coverages)`, so the final smoothed verdict reflects
            the whole simulated track.

    Returns:
        The measured `AdversarialDemoResult`.

    Raises:
        ValueError: If `coverages` is empty.
    """
    if not coverages:
        raise ValueError("coverages must be non-empty.")
    if window_size is None:
        window_size = len(coverages)

    detector = _get_stub_detector_class()()
    raw_verdicts: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, coverage in enumerate(coverages):
            frame_path = Path(tmp_dir) / f"frame_{i:03d}.png"
            _make_patched_frame(coverage).save(frame_path)
            detection = detector.detect_batch([frame_path])[0]
            verdicts, _ = interpret_detections(
                [{"image": str(frame_path), "defect_score": detection["defect_score"]}],
                DEFECT_SCORE_THRESHOLD,
            )
            raw_verdicts.append(verdicts[0].verdict)

    false_negatives = sum(1 for verdict in raw_verdicts if verdict == "ok")
    raw_false_negative_rate = false_negatives / len(raw_verdicts)

    smoother = VerdictSmoother(window_size=window_size)
    smoothed = None
    for raw_verdict in raw_verdicts:
        smoothed = smoother.update(track_id=0, raw_label=raw_verdict)

    return AdversarialDemoResult(
        num_frames=len(coverages),
        raw_verdicts=tuple(raw_verdicts),
        raw_false_negative_rate=raw_false_negative_rate,
        smoothed_final_verdict=smoothed.label,  # type: ignore[union-attr]
        defense_recovered_correct_verdict=smoothed.label == "defect",  # type: ignore[union-attr]
    )
