"""One dev/jetson environment switch, shared by every capstone layer (week 12).

Every subsystem built across weeks 1-11 already has its OWN environment
concept, independently: `src.config.TARGETS`/`apply_target_overrides`
(week 01, CLIP precision), `CONFIG_PATH` (week 08,
`src/api/dependencies.py::get_config`, which of two full YAML files to
load), `docker-compose.jetson.yml`'s `mem_limit` (week 08), and
`assistant/config/*.yaml`'s per-file settings (week 11) that don't
distinguish dev/jetson at all yet. None of that is wrong -- each was the
right scope for its own week. `scenarios/` (day 3) is the first thing in
this repo that needs to make ONE dev-vs-jetson decision and have every
layer it touches (vision/nlp/robot/database) agree on it, so this module
exists to be that single source of truth, not to replace or duplicate
any of the above.
"""

from __future__ import annotations

import os
from enum import Enum

_ENV_VAR = "VLM_ENVIRONMENT"


class Environment(str, Enum):  # noqa: UP042
    """A scenario runs in exactly one of these.

    Matches `src.config.TARGETS` (`"dev"`/`"jetson"`) exactly -- same two
    names, so `Environment.JETSON.value` can be passed straight into
    `apply_target_overrides` without a translation table.

    `str, Enum` on purpose, not `enum.StrEnum` (ruff's UP042 wants the
    latter) -- `StrEnum` needs Python 3.11, `pyproject.toml` declares
    `>=3.11`, but a real dev machine running Python 3.10 hit this via
    `ImportError: cannot import name 'StrEnum' from 'enum'` in week 21's
    `compliance/checklist.py`/week 24's `capstone2/report.py` (same
    pattern, same real traceback). `str, Enum` behaves identically for
    every way this codebase actually uses these enums (serialization,
    `==` against a plain string, `.value`) and costs nothing on 3.11+ --
    confirmed by this sandbox's own test suite, still green on 3.11
    after the change.
    """

    DEV = "dev"
    JETSON = "jetson"


def resolve_environment(override: str | None = None) -> Environment:
    """Resolve which environment a scenario/layer is running in.

    Resolution order: `override` if given, else `$VLM_ENVIRONMENT`, else
    `Environment.DEV` -- the same "explicit arg > env var > safe default"
    order `src/api/dependencies.py::get_config` already established for
    `CONFIG_PATH`, applied here to the coarser dev-vs-jetson question
    instead of a single file path.

    Args:
        override: Explicit value, typically a CLI `--env` flag; skips the
            environment variable entirely when given.

    Returns:
        The resolved `Environment`.

    Raises:
        ValueError: If the resolved string isn't a valid `Environment`.
    """
    raw = override if override is not None else os.environ.get(_ENV_VAR, Environment.DEV.value)
    try:
        return Environment(raw)
    except ValueError as exc:
        valid = ", ".join(e.value for e in Environment)
        raise ValueError(
            f"Unknown environment {raw!r} (from "
            f"{'override' if override is not None else f'${_ENV_VAR}'}); "
            f"expected one of: {valid}."
        ) from exc


def is_jetson(env: Environment) -> bool:
    """`True` for `Environment.JETSON` -- a small readability helper over the enum compare."""
    return env is Environment.JETSON
