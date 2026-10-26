"""Tests for `src.scenarios.color_sorting.ColorSortingScenario` (week 12, day 3).

Real end-to-end against a real `VisionBridge` (`httpx.MockTransport`,
same pattern `test_vision_bridge.py` established) and a real
`Manipulator`/`Database` (simulate mode) -- only the HTTP layer is
mocked, everything else in the call chain is the actual code.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from assistant.utils.config import load_bridges_config
from assistant.vision_bridge import VisionBridge
from src.config.environment import Environment
from src.database import Database
from src.robot import JointPosition, Manipulator, OPCUABridge, ROSInterface
from src.scenarios.base import ScenarioContext
from src.scenarios.color_sorting import ColorSortingScenario, _color_prompts
from src.scenarios.config import CameraSource, ColorSortingConfig


def _config(colors=("red", "green")) -> ColorSortingConfig:
    return ColorSortingConfig(
        colors=colors,
        margin_threshold=0.1,
        pick_position=JointPosition(2048, 2800, 1200, 2048, 2048, 4095),
        bin_positions={
            "red": JointPosition(1000, 2400, 1600, 2048, 2048, 0),
            "green": JointPosition(2048, 2400, 1600, 2048, 2048, 0),
        },
        # camera isn't used by these tests -- image_source is injected directly
        camera=CameraSource("synthetic"),
    )


def _context(tmp_path: Path, vision: VisionBridge) -> ScenarioContext:
    manipulator = Manipulator()  # simulate=True by default (config/manipulator.yaml)
    manipulator.connect()
    return ScenarioContext(
        vision=vision,
        robot_ros=ROSInterface(),
        robot_opc_ua=OPCUABridge(),
        manipulator=manipulator,
        database=Database(tmp_path / "detections.db"),
        nlp=None,
        environment=Environment.DEV,
    )


def _vision_bridge_biased_toward(winning_color: str) -> VisionBridge:
    winning_phrase = _color_prompts((winning_color,))[winning_color][0]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embed/image":
            return httpx.Response(200, json={"vectors": [[1.0, 0.0]], "model_name": "x", "dim": 2})
        body = request.content.decode()
        # Match the exact reference phrase for winning_color, not just the
        # color name -- "green colored"/"blue colored"/etc. all contain
        # the substring "red" via "coloRED", so a bare
        # `winning_color in body` check would false-match every class's
        # phrase, not just the winner's.
        vector = [1.0, 0.0] if winning_phrase in body else [0.0, 1.0]
        return httpx.Response(200, json={"vectors": [vector], "model_name": "x", "dim": 2})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VisionBridge(config=load_bridges_config().vision_bridge, client=client)


def test_no_frame_available_is_a_clean_failure(tmp_path: Path) -> None:
    vision = _vision_bridge_biased_toward("red")
    scenario = ColorSortingScenario(config=_config(), image_source=lambda: None)
    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)

    assert result.success is False
    assert "No frame available" in result.message


def test_confident_classification_sorts_and_logs(tmp_path: Path) -> None:
    vision = _vision_bridge_biased_toward("red")
    scenario = ColorSortingScenario(config=_config(), image_source=lambda: b"fake-jpeg-bytes")
    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)
        db_len = len(ctx.database)

    assert result.success is True
    assert result.data["color"] == "red"
    assert db_len == 1


def test_uncertain_classification_does_not_move_or_log(tmp_path: Path) -> None:
    def flat_handler(request: httpx.Request) -> httpx.Response:
        # Every class gets an identical vector -- margin is always 0.
        return httpx.Response(200, json={"vectors": [[1.0, 0.0]], "model_name": "x", "dim": 2})

    client = httpx.Client(transport=httpx.MockTransport(flat_handler))
    vision = VisionBridge(config=load_bridges_config().vision_bridge, client=client)
    scenario = ColorSortingScenario(config=_config(), image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)
        db_len = len(ctx.database)

    assert result.success is False
    assert "Uncertain" in result.message
    assert db_len == 0


def test_vision_service_unreachable_is_a_clean_failure(tmp_path: Path) -> None:
    # real network, nothing listening
    vision = VisionBridge(config=load_bridges_config().vision_bridge)
    scenario = ColorSortingScenario(config=_config(), image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)

    assert result.success is False
    assert "classification failed" in result.message.lower()


def test_no_bin_configured_for_predicted_color_is_a_clean_failure(tmp_path: Path) -> None:
    config = _config(colors=("red", "purple"))  # "purple" has no bin_positions entry
    vision = _vision_bridge_biased_toward("purple")
    scenario = ColorSortingScenario(config=config, image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)

    assert result.success is False
    assert "No bin configured" in result.message


def test_scenario_name_is_color_sorting() -> None:
    scenario = ColorSortingScenario(config=_config(), image_source=lambda: None)
    assert scenario.name == "color_sorting"
