"""DefectDetectionScenario -- classify defect type, reject defective items (week 12, day 3).

One cycle: grab a frame from `image_source`, zero-shot-classify it
against `"ok"` plus every `src.nlp.DEFECT_TYPES` label (week 11's
`scratch`/`pit`/`crack` registry, reused as-is -- not a new label set)
via `context.vision.classify_by_prompts`, then command
`context.manipulator` to reject anything classified as a defect. An
`"ok"` result takes no manipulator action and isn't logged -- the shared
`Database`/`DetectionRecord` (week 11 day 4) is a log of DETECTIONS, not
a record of every inspected-and-passed item.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from src.database import DetectionRecord
from src.nlp import DEFECT_TYPES
from src.robot import ManipulatorError
from src.scenarios.base import Scenario, ScenarioContext, ScenarioResult
from src.scenarios.config import DefectDetectionConfig
from src.vision import VisionBridgeError

OK_LABEL = "ok"


def _defect_prompts(defect_types: tuple[str, ...] = DEFECT_TYPES) -> dict[str, list[str]]:
    """`{"ok": [...], "scratch": [...], ...}` -- one reference phrase per class."""
    phrases = {OK_LABEL: ["a photo of an ok, defect-free surface"]}
    phrases.update(
        {defect_type: [f"a photo of a {defect_type} defect"] for defect_type in defect_types}
    )
    return phrases


@dataclass
class DefectDetectionScenario(Scenario):
    """Classifies one item per `run()` call, rejects it with the manipulator if defective.

    Attributes:
        config: See `src.scenarios.config.DefectDetectionConfig`.
        image_source: Zero-argument callable returning one frame's PNG
            bytes, or `None` if none is currently available (see
            `src.scenarios.image_sources.build_defect_detection_image_source`).
        defect_types: Class labels considered defects, alongside the
            fixed `"ok"` label. Defaults to `src.nlp.DEFECT_TYPES`
            (week 11's registry) -- overridable for a line inspecting a
            different defect taxonomy.
    """

    config: DefectDetectionConfig
    image_source: Callable[[], bytes | None]
    defect_types: tuple[str, ...] = DEFECT_TYPES
    name = "defect_detection"

    def run(self, context: ScenarioContext) -> ScenarioResult:
        """Classify one frame and reject it if defective, or report why it couldn't.

        Returns:
            `success=False` (never raises) for: no frame available, the
            vision service unreachable/erroring, an uncertain
            classification, or a manipulator failure partway through the
            reject sequence. `success=True` for both an `"ok"` result
            (no action taken) and a completed reject (logged to
            `context.database`) -- `data["predicted_class"]` and
            `data["action"]` (`"pass"` or `"reject"`) distinguish them.
        """
        image_bytes = self.image_source()
        if image_bytes is None:
            return ScenarioResult(
                self.name, success=False, message="No frame available from the image source."
            )

        try:
            result = context.vision.classify_by_prompts(
                image_bytes, _defect_prompts(self.defect_types)
            )
        except VisionBridgeError as exc:
            return ScenarioResult(
                self.name, success=False, message=f"Defect classification failed: {exc}"
            )

        if result.margin < self.config.margin_threshold:
            return ScenarioResult(
                self.name,
                success=False,
                message=(
                    f"Uncertain defect classification (margin={result.margin:.4f} < "
                    f"threshold={self.config.margin_threshold})."
                ),
                data={"similarities": result.similarities, "margin": result.margin},
            )

        if result.predicted_class == OK_LABEL:
            return ScenarioResult(
                self.name,
                success=True,
                message="OK -- no defect found, no action taken.",
                data={"predicted_class": OK_LABEL, "action": "pass"},
            )

        try:
            context.manipulator.move_to(self.config.pick_position)
            context.manipulator.close_gripper()
            context.manipulator.move_to(self.config.reject_position)
            context.manipulator.open_gripper()
            context.manipulator.home()
        except ManipulatorError as exc:
            return ScenarioResult(
                self.name, success=False, message=f"Manipulator move failed: {exc}"
            )

        context.database.record(
            DetectionRecord(
                defect_type=result.predicted_class,
                confidence=result.similarities[result.predicted_class],
                detected_at=time.time(),
                source=self.name,
            )
        )
        return ScenarioResult(
            self.name,
            success=True,
            message=f"Rejected: classified as {result.predicted_class!r}.",
            data={
                "predicted_class": result.predicted_class,
                "action": "reject",
                "similarities": result.similarities,
            },
        )
