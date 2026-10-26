"""Tests for `src.scenarios.base` (week 12, capstone) -- genuinely new code, not a re-export."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.environment import Environment
from src.scenarios import Scenario, ScenarioContext, ScenarioResult, build_default_context


def test_build_default_context_resolves_dev_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLM_ENVIRONMENT", raising=False)
    with build_default_context(db_path=str(tmp_path / "d.db")) as ctx:
        assert ctx.environment is Environment.DEV


def test_build_default_context_accepts_an_environment_override(tmp_path: Path) -> None:
    with build_default_context("jetson", db_path=str(tmp_path / "d.db")) as ctx:
        assert ctx.environment is Environment.JETSON


def test_build_default_context_wires_real_working_layers(tmp_path: Path) -> None:
    with build_default_context("dev", db_path=str(tmp_path / "d.db")) as ctx:
        assert ctx.robot_ros.get_status()["mode"] == "simulate"
        assert ctx.robot_opc_ua.get_status()["mode"] == "simulate"
        assert len(ctx.database) == 0
        assert ctx.nlp is not None


def test_build_default_context_manipulator_is_already_connected(tmp_path: Path) -> None:
    with build_default_context("dev", db_path=str(tmp_path / "d.db")) as ctx:
        assert ctx.manipulator.get_status()["connected"] is True
        # No connect() call needed here -- build_default_context already did it.
        positions = ctx.manipulator.get_positions()
        assert set(positions) == {
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        }


def test_build_default_context_with_nlp_false_skips_the_llm_parser(tmp_path: Path) -> None:
    with build_default_context("dev", db_path=str(tmp_path / "d.db"), with_nlp=False) as ctx:
        assert ctx.nlp is None


def test_scenario_context_shutdown_is_idempotent_via_context_manager(tmp_path: Path) -> None:
    ctx = build_default_context("dev", db_path=str(tmp_path / "d.db"))
    with ctx:
        pass
    # __exit__ already called shutdown(); calling it again must not raise.
    ctx.shutdown()


def test_scenario_is_an_abstract_base_class() -> None:
    with pytest.raises(TypeError):
        Scenario()  # type: ignore[abstract]


def test_scenario_subclass_must_implement_run() -> None:
    class Incomplete(Scenario):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_scenario_subclass_run_is_called_with_the_context(tmp_path: Path) -> None:
    class Echo(Scenario):
        name = "echo"

        def run(self, context: ScenarioContext) -> ScenarioResult:
            return ScenarioResult(
                scenario=self.name, success=True, message="ok", data={"env": context.environment}
            )

    with build_default_context("dev", db_path=str(tmp_path / "d.db")) as ctx:
        result = Echo().run(ctx)

    assert result.success is True
    assert result.scenario == "echo"
    assert result.data["env"] is Environment.DEV
