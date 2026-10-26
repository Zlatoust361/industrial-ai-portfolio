"""Intent/parameter schemas for `llm_parser.py` (week 11, day 2).

One `pydantic` model per known intent's parameters -- both the local
rule-based matcher and the external Claude API tier validate their
output against the SAME registry (`INTENT_PARAM_SCHEMAS`) before a
`ParsedCommand` is considered valid, so "the LLM made something up" and
"the regex extracted something malformed" fail the exact same way:
`pydantic.ValidationError`, caught by the caller, never silently executed.
`extra="forbid"` on every model -- an unrecognized parameter name (typo,
hallucinated field) is a validation failure, not a silently-ignored key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFECT_TYPES: tuple[str, ...] = ("scratch", "pit", "crack")
"""Canonical defect-type names -- matches `config/defects.yaml`'s shape
names (week 05), the vocabulary the rest of this repo already uses."""


class NoParams(BaseModel):
    """Parameter schema for intents that take no arguments (e.g. `get_status`)."""

    model_config = ConfigDict(extra="forbid")


class ListDefectsParams(BaseModel):
    """Parameters for `"list_defects"` -- e.g. "покажи последние 50 дефектов crack с conf>0.9"."""

    model_config = ConfigDict(extra="forbid")

    defect_type: Literal["scratch", "pit", "crack"] | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=500)


class AskRunbookParams(BaseModel):
    """Parameters for `"ask_runbook"` (week 19) -- e.g. "что делать при обрыве камеры?".

    Attributes:
        question: The operator's free-text question, passed through
            verbatim to `rag.integration.answer_question(_with_default_index)`
            -- this schema's only job is to make sure SOME non-empty
            question text was actually extracted, not to parse it further.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


INTENT_PARAM_SCHEMAS: dict[str, type[BaseModel]] = {
    "list_defects": ListDefectsParams,
    "get_status": NoParams,
    "restart_training": NoParams,
    "help": NoParams,
    "ask_runbook": AskRunbookParams,
}
"""Every intent `llm_parser.py` knows how to execute, and the parameter
shape each one requires. Day 3's `action_executor.py` is expected to have
one handler per key here -- this registry is the contract between the two."""

KNOWN_INTENTS: tuple[str, ...] = tuple(INTENT_PARAM_SCHEMAS)


class ParsedCommand(BaseModel):
    """One parsed operator command, ready to hand to day 3's `action_executor.py`.

    Attributes:
        intent: One of `KNOWN_INTENTS`, or a sentinel
            (`"pending"`/`"unavailable"`/`"error"`/`"unknown"`) when
            `llm_parser.py` couldn't produce a real, executable intent --
            callers MUST check `intent in KNOWN_INTENTS` (or `error is
            None`) before dispatching to an executor, never assume every
            `ParsedCommand` is actionable.
        parameters: Already validated against `INTENT_PARAM_SCHEMAS[intent]`
            when `intent` is a real, known intent; `{}` for sentinel intents.
        source: Which tier produced this -- `"local"` (rule-based
            matcher) or `"api"` (external Claude API, or a sentinel
            result from that tier's queue/rate-limit/error handling).
        raw_text: The original operator text, for logging/audit.
        confidence: `1.0` for a clean local rule match; `None` for the
            API tier (Claude's tool-use response carries no confidence
            score) or a sentinel result.
        error: `None` for a real, executable result; a short
            human-readable reason otherwise (queue full, still pending,
            API call failed, etc.).
    """

    intent: str
    parameters: dict[str, object]
    source: Literal["local", "api"]
    raw_text: str
    confidence: float | None = None
    error: str | None = None

    @property
    def is_actionable(self) -> bool:
        """`True` if this is a real, schema-valid, executable command."""
        return self.error is None and self.intent in KNOWN_INTENTS


def validate_parameters(intent: str, parameters: dict[str, object]) -> BaseModel:
    """Validate `parameters` against `intent`'s registered schema.

    Args:
        intent: Must be a key of `INTENT_PARAM_SCHEMAS`.
        parameters: Raw parameter dict, from either the local matcher or
            the external API tier.

    Returns:
        The validated, typed parameter model.

    Raises:
        KeyError: If `intent` isn't in `INTENT_PARAM_SCHEMAS` (unknown
            intent -- from a hallucinated Claude tool-use call, or a
            local-matcher bug).
        pydantic.ValidationError: If `parameters` don't match that
            intent's schema (wrong type, out-of-range value, unknown key).
    """
    schema = INTENT_PARAM_SCHEMAS[intent]
    return schema.model_validate(parameters)
