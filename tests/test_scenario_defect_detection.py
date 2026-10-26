"""Tests for `src.scenarios.defect_detection.DefectDetectionScenario` (week 12, day 3).

Same real-end-to-end shape as `test_scenario_color_sorting.py`: a real
`VisionBridge` against `httpx.MockTransport`, a real `Manipulator`/
`Database` in simulate mode.
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
from src.scenarios.config import CameraSource, DefectDetectionConfig
from src.scenarios.defect_detection import OK_LABEL, DefectDetectionScenario, _defect_prompts


def _config() -> DefectDetectionConfig:
    return DefectDetectionConfig(
        margin_threshold=0.1,
        pick_position=JointPosition(2048, 2800, 1200, 2048, 2048, 4095),
        reject_position=JointPosition(500, 2400, 1600, 2048, 2048, 0),
        camera=CameraSource("synthetic"),
    )


def _context(tmp_path: Path, vision: VisionBridge) -> ScenarioContext:
    manipulator = Manipulator()
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


def _vision_bridge_biased_toward(winning_class: str) -> VisionBridge:
    winning_phrase = _defect_prompts()[winning_class][0]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embed/image":
            return httpx.Response(200, json={"vectors": [[1.0, 0.0]], "model_name": "x", "dim": 2})
        body = request.content.decode()
        # Match the exact reference phrase for winning_class, not just its
        # label -- a bare substring check on the label risks the same
        # false-positive-via-another-word bug a shorter check would hit
        # (see test_scenario_color_sorting.py's own note on this).
        vector = [1.0, 0.0] if winning_phrase in body else [0.0, 1.0]
        return httpx.Response(200, json={"vectors": [vector], "model_name": "x", "dim": 2})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VisionBridge(config=load_bridges_config().vision_bridge, client=client)


def test_no_frame_available_is_a_clean_failure(tmp_path: Path) -> None:
    vision = _vision_bridge_biased_toward("crack")
    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: None)
    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)

    assert result.success is False
    assert "No frame available" in result.message


def test_ok_classification_takes_no_action_and_is_not_logged(tmp_path: Path) -> None:
    vision = _vision_bridge_biased_toward(OK_LABEL)
    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)
        db_len = len(ctx.database)

    assert result.success is True
    assert result.data == {"predicted_class": "ok", "action": "pass"}
    assert db_len == 0


def test_defect_classification_rejects_and_logs(tmp_path: Path) -> None:
    vision = _vision_bridge_biased_toward("crack")
    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)
        db_len = len(ctx.database)
        (record,) = ctx.database.query(limit=1)

    assert result.success is True
    assert result.data["predicted_class"] == "crack"
    assert result.data["action"] == "reject"
    assert db_len == 1
    assert record.defect_type == "crack"
    assert record.source == "defect_detection"


def test_uncertain_classification_does_not_move_or_log(tmp_path: Path) -> None:
    def flat_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vectors": [[1.0, 0.0]], "model_name": "x", "dim": 2})

    client = httpx.Client(transport=httpx.MockTransport(flat_handler))
    vision = VisionBridge(config=load_bridges_config().vision_bridge, client=client)
    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)
        db_len = len(ctx.database)

    assert result.success is False
    assert "Uncertain" in result.message
    assert db_len == 0


def test_vision_service_unreachable_is_a_clean_failure(tmp_path: Path) -> None:
    # real network, nothing listening
    vision = VisionBridge(config=load_bridges_config().vision_bridge)
    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: b"x")

    with _context(tmp_path, vision) as ctx:
        result = scenario.run(ctx)

    assert result.success is False
    assert "classification failed" in result.message.lower()


def test_default_defect_types_come_from_src_nlp() -> None:
    from src.nlp import DEFECT_TYPES

    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: None)
    assert scenario.defect_types == DEFECT_TYPES


def test_scenario_name_is_defect_detection() -> None:
    scenario = DefectDetectionScenario(config=_config(), image_source=lambda: None)
    assert scenario.name == "defect_detection"
