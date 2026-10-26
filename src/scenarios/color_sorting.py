"""ColorSortingScenario -- classify an item's color, physically sort it (week 12, day 3).

One cycle: grab a frame from `image_source`, zero-shot-classify its color
via `context.vision.classify_by_prompts` (week 11 day 3's HTTP-calling
bridge, week 02's cosine-similarity math -- nothing reimplemented here),
then command `context.manipulator` (week 12 day 3's SO-ARM 101 interface)
through pick -> drop-into-bin -> home. `Scenario.run(context)`'s uniform
signature (`src/scenarios/base.py`) is kept exactly -- the per-cycle input
(which frame) is injected via `image_source` at construction, not an
extra `run()` parameter, so a driver script calls `scenario.run(context)`
the same way for any `Scenario` subclass.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from src.database import DetectionRecord
from src.robot import ManipulatorError
from src.scenarios.base import Scenario, ScenarioContext, ScenarioResult
from src.scenarios.config import ColorSortingConfig
from src.vision import VisionBridgeError


def _color_prompts(colors: tuple[str, ...]) -> dict[str, list[str]]:
    """One reference phrase per color -- what `classify_by_prompts` classifies against."""
    return {color: [f"a photo of a {color} colored object"] for color in colors}


@dataclass
class ColorSortingScenario(Scenario):
    """Sorts one item per `run()` call into the bin matching its classified color.

    Attributes:
        config: See `src.scenarios.config.ColorSortingConfig`.
        image_source: Zero-argument callable returning one frame's PNG
            bytes, or `None` if none is currently available (see
            `src.scenarios.image_sources.build_color_sorting_image_source`).
    """

    config: ColorSortingConfig
    image_source: Callable[[], bytes | None]
    name = "color_sorting"

    def run(self, context: ScenarioContext) -> ScenarioResult:
        """Classify one frame's color and sort it, or report why it couldn't.

        Returns:
            `success=False` (never raises) for: no frame available, the
            vision service unreachable/erroring, an uncertain
            classification (`margin` below `config.margin_threshold`), no
            bin configured for the predicted color, or a manipulator
            failure partway through the pick/place sequence -- in every
            case `message` says which. `success=True` only after the
            full pick -> bin -> home sequence completes and the sort is
            logged to `context.database`.
        """
        image_bytes = self.image_source()
        if image_bytes is None:
            return ScenarioResult(
                self.name, success=False, message="No frame available from the image source."
            )

        try:
            result = context.vision.classify_by_prompts(
                image_bytes, _color_prompts(self.config.colors)
            )
        except VisionBridgeError as exc:
            return ScenarioResult(
                self.name, success=False, message=f"Color classification failed: {exc}"
            )

        if result.margin < self.config.margin_threshold:
            return ScenarioResult(
                self.name,
                success=False,
                message=(
                    f"Uncertain color classification (margin={result.margin:.4f} < "
                    f"threshold={self.config.margin_threshold})."
                ),
                data={"similarities": result.similarities, "margin": result.margin},
            )

        bin_position = self.config.bin_positions.get(result.predicted_class)
        if bin_position is None:
            return ScenarioResult(
                self.name,
                success=False,
                message=f"No bin configured for color {result.predicted_class!r}.",
            )

        try:
            context.manipulator.move_to(self.config.pick_position)
            context.manipulator.close_gripper()
            context.manipulator.move_to(bin_position)
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
            message=f"Sorted into the {result.predicted_class!r} bin.",
            data={"color": result.predicted_class, "similarities": result.similarities},
        )
