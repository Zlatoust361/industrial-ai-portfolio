"""Unit tests for `assistant.schemas`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from assistant.schemas import (
    INTENT_PARAM_SCHEMAS,
    KNOWN_INTENTS,
    AskRunbookParams,
    ListDefectsParams,
    NoParams,
    ParsedCommand,
    validate_parameters,
)


def test_known_intents_matches_schema_registry_keys() -> None:
    assert set(KNOWN_INTENTS) == set(INTENT_PARAM_SCHEMAS)


# -- NoParams ------------------------------------------------------------------


def test_no_params_accepts_empty_dict() -> None:
    NoParams.model_validate({})


def test_no_params_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        NoParams.model_validate({"unexpected": 1})


# -- ListDefectsParams -----------------------------------------------------------


def test_list_defects_params_defaults() -> None:
    params = ListDefectsParams.model_validate({})
    assert params.defect_type is None
    assert params.min_confidence is None
    assert params.limit == 20


def test_list_defects_params_accepts_full_valid_input() -> None:
    params = ListDefectsParams.model_validate(
        {"defect_type": "crack", "min_confidence": 0.9, "limit": 50}
    )
    assert params.defect_type == "crack"
    assert params.min_confidence == 0.9
    assert params.limit == 50


def test_list_defects_params_rejects_unknown_defect_type() -> None:
    with pytest.raises(ValidationError):
        ListDefectsParams.model_validate({"defect_type": "dent"})


@pytest.mark.parametrize("min_confidence", [-0.1, 1.1])
def test_list_defects_params_rejects_out_of_range_confidence(min_confidence: float) -> None:
    with pytest.raises(ValidationError):
        ListDefectsParams.model_validate({"min_confidence": min_confidence})


@pytest.mark.parametrize("limit", [0, -1, 501])
def test_list_defects_params_rejects_out_of_range_limit(limit: int) -> None:
    with pytest.raises(ValidationError):
        ListDefectsParams.model_validate({"limit": limit})


def test_list_defects_params_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        ListDefectsParams.model_validate({"sort_order": "asc"})


# -- validate_parameters --------------------------------------------------------


def test_validate_parameters_unknown_intent_raises_key_error() -> None:
    with pytest.raises(KeyError):
        validate_parameters("delete_everything", {})


def test_validate_parameters_invalid_params_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        validate_parameters("list_defects", {"limit": -5})


def test_validate_parameters_success_returns_validated_model() -> None:
    result = validate_parameters("list_defects", {"limit": 5})
    assert isinstance(result, ListDefectsParams)
    assert result.limit == 5


# -- ParsedCommand.is_actionable -------------------------------------------------


def test_parsed_command_is_actionable_for_a_real_result() -> None:
    command = ParsedCommand(
        intent="get_status", parameters={}, source="local", raw_text="статус", confidence=1.0
    )
    assert command.is_actionable


@pytest.mark.parametrize("intent", ["pending", "unavailable", "error", "unknown"])
def test_parsed_command_not_actionable_for_sentinel_intents(intent: str) -> None:
    command = ParsedCommand(
        intent=intent, parameters={}, source="api", raw_text="...", error="something"
    )
    assert not command.is_actionable


def test_ask_runbook_params_accepts_a_question() -> None:
    params = AskRunbookParams.model_validate({"question": "что делать при обрыве камеры?"})
    assert params.question == "что делать при обрыве камеры?"


def test_ask_runbook_params_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        AskRunbookParams.model_validate({"question": ""})


def test_ask_runbook_params_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        AskRunbookParams.model_validate({"question": "x", "unexpected": 1})


def test_parsed_command_not_actionable_when_error_set_even_with_known_intent() -> None:
    # Defensive: a caller that (incorrectly) set intent="get_status" AND an
    # error should still be told not to execute it.
    command = ParsedCommand(
        intent="get_status", parameters={}, source="api", raw_text="...", error="stale"
    )
    assert not command.is_actionable
