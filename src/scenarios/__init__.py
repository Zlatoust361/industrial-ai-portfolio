"""Scenarios layer -- capstone orchestration built on top of vision/nlp/robot/database (week 12).

This package is genuinely new (not a re-export of an earlier week, unlike
`vision`/`nlp`/`robot`/`database`/`logging`): no prior week had a
standalone "watch, decide, act" loop driving a physical manipulator. Day
2 laid the shared scaffolding (`base.py`: `Scenario`, `ScenarioContext`,
`ScenarioResult`, `build_default_context`); day 3 adds the two concrete
scenarios this week's deliverable names (`color_sorting.py`,
`defect_detection.py`, each a `Scenario` subclass), `config.py`'s
env-aware (`dev`/`jetson`) loaders, and `image_sources.py`'s synthetic
(dev)/real-camera (jetson) frame sources.
"""

from __future__ import annotations

from src.scenarios.base import Scenario, ScenarioContext, ScenarioResult, build_default_context
from src.scenarios.color_sorting import ColorSortingScenario
from src.scenarios.defect_detection import DefectDetectionScenario

__all__ = [
    "Scenario",
    "ScenarioContext",
    "ScenarioResult",
    "build_default_context",
    "ColorSortingScenario",
    "DefectDetectionScenario",
]
