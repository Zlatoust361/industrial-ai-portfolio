"""Shared scenario scaffolding -- what `color_sorting.py`/`defect_detection.py` (day 3) build on.

A scenario is a standalone, long-running loop (watch a camera feed,
decide, act) -- unlike `assistant/action_executor.py`'s `ActionExecutor`,
which dispatches ONE already-parsed chat command per call and returns.
The two are complementary, not overlapping: an operator's chat command
can still reach `ActionExecutor` while a scenario is running (both can
share the same `robot`/`database` layer instances, see `ScenarioContext`
below), but a scenario itself isn't triggered by chat text at all --
day 3's manipulator-driven scenarios run on their own.

`ScenarioContext`/`build_default_context` exist so every concrete
scenario gets one real, tested way to construct the shared layers it
needs, instead of wiring `VisionBridge()`/`LLMParser()`/`ROSInterface()`/
`OPCUABridge()`/`Manipulator()`/`Database()` by hand and risking several
slightly different constructions. `manipulator` (week 12 day 3) is the
newest addition -- day 2 didn't have `src.robot.manipulator` yet.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from src.config.environment import Environment, resolve_environment
from src.database import Database
from src.nlp import LLMParser
from src.robot import Manipulator, OPCUABridge, ROSInterface
from src.vision import VisionBridge


@dataclass(frozen=True)
class ScenarioResult:
    """The outcome of one `Scenario.run()` call.

    Attributes:
        scenario: The scenario's own `name`, echoed back for logging.
        success: `False` for anything from a setup failure through an
            unhandled exception inside `run()` -- `message` explains why.
        message: Human-readable summary.
        data: Structured payload (e.g. sorted-item counts, defect
            records) -- `{}` when there's nothing beyond `message`.
    """

    scenario: str
    success: bool
    message: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class ScenarioContext:
    """The four shared layers a scenario runs against, plus which environment it's in.

    Attributes:
        vision: See `src.vision.VisionBridge`.
        robot_ros: See `src.robot.ROSInterface`.
        robot_opc_ua: See `src.robot.OPCUABridge`.
        manipulator: See `src.robot.Manipulator` (SO-ARM 101 control, week 12 day 3).
        database: See `src.database.Database`.
        nlp: See `src.nlp.LLMParser` -- optional, most scenarios act on
            vision/sensor input rather than operator text, but a scenario
            that also takes chat overrides (e.g. "стоп") needs it.
        environment: `Environment.DEV` or `Environment.JETSON` --
            resolved once for the whole context, not re-derived per layer.
    """

    vision: VisionBridge
    robot_ros: ROSInterface
    robot_opc_ua: OPCUABridge
    manipulator: Manipulator
    database: Database
    nlp: LLMParser | None
    environment: Environment

    def shutdown(self) -> None:
        """Tear down every layer that owns a real connection/thread."""
        if self.nlp is not None:
            self.nlp.shutdown()
        self.robot_ros.shutdown()
        self.robot_opc_ua.disconnect()
        self.manipulator.disconnect()
        self.database.close()

    def __enter__(self) -> ScenarioContext:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()


def build_default_context(
    environment: str | Environment | None = None,
    *,
    db_path: str = "outputs/scenarios_demo/detections.db",
    with_nlp: bool = True,
) -> ScenarioContext:
    """Construct a `ScenarioContext` from each layer's own real default config.

    Args:
        environment: `Environment`, a raw `"dev"`/`"jetson"` string, or
            `None` to resolve from `$VLM_ENVIRONMENT` (see
            `src.config.environment.resolve_environment`).
        db_path: Where the scenario's `Database` persists detections.
        with_nlp: Build a real `LLMParser` too. `False` skips it (and its
            background rate-limiter thread) for a scenario that never
            reads operator text.

    Returns:
        A `ScenarioContext` wired from real, already-tested constructors
        (`VisionBridge()`, `ROSInterface()`, `OPCUABridge()`, `Database()`,
        optionally `LLMParser()`) -- the same defaults
        `assistant.action_executor.ActionExecutor()` and
        `assistant.chat_app.py`'s `lifespan` already use, plus a
        `Manipulator()` that's already `connect()`ed (unlike the other
        bridges, which connect lazily on first real use, `Manipulator`
        needs an explicit `connect()` before any position read/write --
        done here so the returned context is immediately ready to move,
        not a half-initialized object the caller has to remember to
        finish setting up).
    """
    resolved_env = (
        environment
        if isinstance(environment, Environment)
        else resolve_environment(environment)
    )
    manipulator = Manipulator()
    manipulator.connect()
    return ScenarioContext(
        vision=VisionBridge(),
        robot_ros=ROSInterface(),
        robot_opc_ua=OPCUABridge(),
        manipulator=manipulator,
        database=Database(db_path),
        nlp=LLMParser() if with_nlp else None,
        environment=resolved_env,
    )


class Scenario(abc.ABC):
    """Base class every concrete scenario (day 3) implements.

    Subclasses provide `name` and `run()`; `ActionExecutor`'s "never
    propagate a handler exception" contract is deliberately mirrored by
    NOT catching exceptions here -- a scenario is a long-running loop a
    human operator is meant to see fail loudly (logs, exit code), unlike
    one chat request that must always get *some* answer back.
    """

    name: str

    @abc.abstractmethod
    def run(self, context: ScenarioContext) -> ScenarioResult:
        """Execute the scenario once against `context`. See subclass docstrings."""
        raise NotImplementedError
