"""llm_parser.py -- text -> intent + parameters, two-tier (week 11, day 2).

Tier 1, local rule-based matching (`match_locally`): plain `re` regexes
for the handful of command shapes an operator on the line actually types
over and over -- list defects with filters, status, restart training,
help. Cheap enough to run on the Jetson Orin Nano itself with no extra
model, no network, sub-millisecond.

Tier 2, external Claude API (`call_claude_api`): only reached when tier 1
finds nothing. Uses Claude's tool-use (function calling) to force a
structured `{intent, parameters}` response, then validates it against the
exact same `assistant.schemas.INTENT_PARAM_SCHEMAS` registry tier 1 uses
-- a hallucinated intent name or malformed parameter fails the same way
regardless of which tier produced it, and is never executed.

`LLMParser` ties the two tiers together behind one `parse()` call, routes
tier-2 requests through `rate_limiter.ExternalCallQueue` (never blocks on
the network beyond `max_wait_seconds`), and tracks how much of the
traffic each tier actually handles (`ParseStats`) -- day 2's own "залогируй
долю локально vs через API" ask.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from assistant.rate_limiter import ExternalCallQueue, QueueFullError, TokenBucketRateLimiter
from assistant.schemas import KNOWN_INTENTS, ParsedCommand, validate_parameters
from assistant.utils.config import ParserConfig, load_parser_config

logger = logging.getLogger(__name__)

# -- tier 1: local rule-based matcher ----------------------------------------

_DEFECT_TYPE_ALIASES: dict[str, str] = {
    "scratch": "scratch",
    "царапина": "scratch",
    "царапину": "scratch",
    "скретч": "scratch",
    "pit": "pit",
    "скол": "pit",
    "вмятина": "pit",
    "crack": "crack",
    "трещина": "crack",
    "трещину": "crack",
}
"""Russian/English surface synonyms -> canonical `assistant.schemas.DEFECT_TYPES` name."""

_LIST_TRIGGER = re.compile(
    r"(?:покажи|показать|список|list|show)\w*.{0,20}?(?:дефект\w*|defects?)", re.IGNORECASE
)
_LIMIT_PATTERN = re.compile(
    r"(?:последн\w*|last|top)\s+(\d+)|(\d+)\s+(?:дефект\w*|defects?)", re.IGNORECASE
)
"""Deliberately NOT a bare `\\d+` search -- that would also grab a stray
digit out of e.g. `"conf>0.9"`'s `0`. Only counts a number immediately
next to a limit-shaped context: after "последние"/"last"/"top", or right
before "дефект(ов)"/"defect(s)"."""
_DEFECT_TYPE_PATTERN = re.compile(
    "|".join(re.escape(word) for word in _DEFECT_TYPE_ALIASES), re.IGNORECASE
)
_CONFIDENCE_EN_PATTERN = re.compile(r"conf(?:idence)?\s*(?:[><]=?|=)\s*(\d*\.?\d+)")
_CONFIDENCE_RU_PATTERN = re.compile(
    r"(?:уверенност\w*|conf)\s*(?:выше|больше|>=?)\s*(\d*\.?\d+)", re.IGNORECASE
)

_STATUS_TRIGGER = re.compile(
    r"\b(статус|как дела|состояние|status|health)\b", re.IGNORECASE
)
_RESTART_TRAINING_TRIGGER = re.compile(
    r"(перезапусти|перезапуск|restart)\w*.{0,15}?(обучени\w*|training)", re.IGNORECASE
)
_HELP_TRIGGER = re.compile(r"\b(помощь|что ты умеешь|команды|help|\?)\b", re.IGNORECASE)

_ASK_RUNBOOK_TRIGGER = re.compile(
    r"камер\w*|camera|качеств\w*|quality.?gate|диск\w*|\bdisk\b|tensorrt|движок|engine|"
    r"температур\w*|throttl\w*|перегрев\w*|инцидент\w*|incident\w*|runbook|регламент\w*",
    re.IGNORECASE,
)
"""Narrow, topic-specific (not "any question mark") -- matches vocabulary
that actually maps to `docs/runbook.md`'s own 5 incident sections (camera/
video loss, quality-gate degradation, disk overflow, TensorRT engine
failure after a JetPack upgrade, thermal throttling), week 19's RAG
intent. Deliberately checked LAST in `_LOCAL_MATCHERS` (see there) --
this is the loosest trigger of the set, and must never shadow a more
specific existing intent."""


def _match_list_defects(text: str) -> dict[str, object] | None:
    if not _LIST_TRIGGER.search(text):
        return None
    params: dict[str, object] = {}
    if limit_match := _LIMIT_PATTERN.search(text):
        limit_str = limit_match.group(1) or limit_match.group(2)
        params["limit"] = int(limit_str)
    if type_match := _DEFECT_TYPE_PATTERN.search(text):
        params["defect_type"] = _DEFECT_TYPE_ALIASES[type_match.group(0).lower()]
    conf_match = _CONFIDENCE_EN_PATTERN.search(text) or _CONFIDENCE_RU_PATTERN.search(text)
    if conf_match:
        params["min_confidence"] = float(conf_match.group(1))
    return params


def _match_get_status(text: str) -> dict[str, object] | None:
    return {} if _STATUS_TRIGGER.search(text) else None


def _match_restart_training(text: str) -> dict[str, object] | None:
    return {} if _RESTART_TRAINING_TRIGGER.search(text) else None


def _match_help(text: str) -> dict[str, object] | None:
    return {} if _HELP_TRIGGER.search(text) else None


def _match_ask_runbook(text: str) -> dict[str, object] | None:
    return {"question": text} if _ASK_RUNBOOK_TRIGGER.search(text) else None


_LOCAL_MATCHERS: tuple[tuple[str, Callable[[str], dict[str, object] | None]], ...] = (
    # Order matters: more specific/less overlap-prone triggers first, so
    # e.g. "restart training" can't accidentally also satisfy a broader
    # pattern checked earlier. list_defects is checked near the end --
    # its trigger is loose (any "show ... defects"-shaped phrase).
    # ask_runbook is checked LAST -- it's the loosest of all (any runbook-
    # topic keyword, see _ASK_RUNBOOK_TRIGGER), so every more specific
    # intent above gets first refusal.
    ("restart_training", _match_restart_training),
    ("get_status", _match_get_status),
    ("help", _match_help),
    ("list_defects", _match_list_defects),
    ("ask_runbook", _match_ask_runbook),
)


def match_locally(text: str) -> ParsedCommand | None:
    """Try every registered local rule matcher, in priority order.

    A matcher "hitting" (returning a dict, even `{}`) isn't enough on its
    own -- the extracted parameters must also pass `INTENT_PARAM_SCHEMAS`
    validation (e.g. a `--limit` of `9999` from a garbled number). A
    trigger match with invalid parameters is treated as a non-match for
    THAT matcher and the search continues, rather than either executing
    something invalid or giving up on the whole text.

    Args:
        text: Raw operator command text.

    Returns:
        A validated `ParsedCommand` (`source="local"`, `confidence=1.0`)
        from the first matcher that both triggers and validates, or
        `None` if nothing in the registry matched -- the caller should
        fall through to `call_claude_api`/the external tier.
    """
    for intent, matcher in _LOCAL_MATCHERS:
        raw_params = matcher(text)
        if raw_params is None:
            continue
        try:
            validated = validate_parameters(intent, raw_params)
        except ValidationError:
            continue
        return ParsedCommand(
            intent=intent,
            parameters=validated.model_dump(),
            source="local",
            raw_text=text,
            confidence=1.0,
        )
    return None


# -- tier 2: external Claude API ---------------------------------------------

CLAUDE_API_ENDPOINT = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"
CLAUDE_TOOL_NAME = "emit_parsed_command"

_SYSTEM_PROMPT = (
    "You parse short operator commands for an industrial defect-inspection line "
    "into a structured intent and parameters. Commands may be in Russian or "
    "English. Always call the emit_parsed_command tool with your best "
    "interpretation -- never reply in plain text."
)


def _build_tool_schema() -> dict[str, object]:
    """The tool-use schema forcing Claude's response into `{intent, parameters}` shape.

    `parameters` is left as a generic `object` here -- per-intent
    validation happens afterward via `assistant.schemas.validate_parameters`,
    the same function tier 1 uses, so both tiers are held to the exact
    same contract rather than trusting the tool schema alone.
    """
    return {
        "name": CLAUDE_TOOL_NAME,
        "description": (
            "Emit the parsed intent and parameters for the operator's command."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": list(KNOWN_INTENTS),
                    "description": "The single best-matching known intent.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for that intent -- {} if it takes none.",
                },
            },
            "required": ["intent", "parameters"],
        },
    }


def call_claude_api(
    text: str, config: ParserConfig, client: object | None = None
) -> ParsedCommand:
    """Call the Claude API (tool use) to parse `text` that tier 1 couldn't.

    Args:
        text: Raw operator command text.
        config: `assistant.utils.config.ParserConfig` (model/timeout/API-key-env).
        client: Optional `httpx.Client`-like object (anything with a
            `.post(url, headers=, json=, timeout=)` method returning an
            `httpx.Response`-like object). Defaults to the real network
            via `httpx.post` -- tests inject an
            `httpx.Client(transport=httpx.MockTransport(...))` instead,
            same pattern as `src/synthetic/diffusion_pipeline.py`'s
            `StabilityAPIBackend` (week 05).

    Returns:
        A validated `ParsedCommand` with `source="api"`.

    Raises:
        RuntimeError: If `config.claude_api.api_key_env` isn't set in the
            environment.
        httpx.HTTPStatusError: On a non-2xx response.
        KeyError: If Claude's tool-use response doesn't include a
            `tool_use` content block, or names an intent outside
            `KNOWN_INTENTS`.
        pydantic.ValidationError: If Claude's `parameters` don't match
            that intent's schema.
    """
    import httpx

    api_key = os.environ.get(config.claude_api.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"${config.claude_api.api_key_env} is not set -- required to call the Claude API "
            "(assistant/config/parser_config.yaml's claude_api.api_key_env)."
        )

    poster = client.post if client is not None else httpx.post
    response = poster(
        CLAUDE_API_ENDPOINT,
        headers={
            "x-api-key": api_key,
            "anthropic-version": CLAUDE_API_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": config.claude_api.model,
            "max_tokens": 512,
            "system": _SYSTEM_PROMPT,
            "tools": [_build_tool_schema()],
            "tool_choice": {"type": "tool", "name": CLAUDE_TOOL_NAME},
            "messages": [{"role": "user", "content": text}],
        },
        timeout=config.claude_api.timeout_s,
    )
    response.raise_for_status()
    body = response.json()

    tool_use = next(
        (block for block in body["content"] if block.get("type") == "tool_use"), None
    )
    if tool_use is None:
        raise KeyError("Claude's response had no tool_use content block.")

    intent = tool_use["input"]["intent"]
    raw_params = tool_use["input"]["parameters"]
    validated = validate_parameters(intent, raw_params)  # KeyError/ValidationError on bad output

    return ParsedCommand(
        intent=intent,
        parameters=validated.model_dump(),
        source="api",
        raw_text=text,
        confidence=None,
    )


# -- stats: local vs API split ------------------------------------------------


@dataclass
class ParseStats:
    """Running counts of how `LLMParser.parse()` resolved each request.

    Day 2's own "залогируй долю команд локально vs через API" ask.
    """

    local_count: int = 0
    api_count: int = 0
    pending_count: int = 0
    rejected_count: int = 0
    error_count: int = 0

    def record(self, outcome: str) -> None:
        """Increment the counter for `outcome`.

        Args:
            outcome: One of `"local"`/`"api"`/`"pending"`/`"rejected"`/`"error"`.
        """
        attr = f"{outcome}_count"
        setattr(self, attr, getattr(self, attr) + 1)

    @property
    def total(self) -> int:
        return (
            self.local_count
            + self.api_count
            + self.pending_count
            + self.rejected_count
            + self.error_count
        )

    def summary(self) -> dict[str, object]:
        """A JSON-serializable snapshot: raw counts plus each tier's fraction of `total`."""
        total = self.total

        def fraction(count: int) -> float:
            return (count / total) if total else 0.0

        return {
            "total": total,
            "local_count": self.local_count,
            "local_fraction": fraction(self.local_count),
            "api_count": self.api_count,
            "api_fraction": fraction(self.api_count),
            "pending_count": self.pending_count,
            "rejected_count": self.rejected_count,
            "error_count": self.error_count,
        }


# -- orchestration -------------------------------------------------------------

_SENTINEL_UNAVAILABLE = "unavailable"
_SENTINEL_PENDING = "pending"
_SENTINEL_ERROR = "error"


class LLMParser:
    """Two-tier command parser: local rules first, rate-limited external API fallback."""

    def __init__(
        self, config: ParserConfig | None = None, api_client: object | None = None
    ) -> None:
        """
        Args:
            config: Defaults to `load_parser_config()` (real
                `assistant/config/parser_config.yaml`).
            api_client: Passed through to `call_claude_api` -- see its
                own docstring. `None` uses the real network.
        """
        self._config = config if config is not None else load_parser_config()
        self._api_client = api_client
        rate_limiter = TokenBucketRateLimiter(
            max_calls=self._config.rate_limit.max_calls_per_window,
            window_seconds=self._config.rate_limit.window_seconds,
        )
        self._queue = ExternalCallQueue(
            worker_fn=self._call_api,
            rate_limiter=rate_limiter,
            max_queue_size=self._config.rate_limit.max_queue_size,
        )
        self.stats = ParseStats()

    def _call_api(self, text: str) -> ParsedCommand:
        return call_claude_api(text, self._config, client=self._api_client)

    def parse(self, text: str, max_wait_s: float | None = None) -> ParsedCommand:
        """Parse one operator command: local tier first, external tier on a miss.

        Args:
            text: Raw operator command text.
            max_wait_s: Overrides `config.rate_limit.max_wait_seconds`
                for this one call.

        Returns:
            A `ParsedCommand`. Check `.is_actionable` before dispatching
            to an executor -- a `"pending"`/`"unavailable"`/`"error"`
            intent means the external tier didn't produce a usable result
            within the wait budget (still processing, queue was full, or
            the call itself failed), not that it's safe to execute.
        """
        local = match_locally(text)
        if local is not None:
            self.stats.record("local")
            logger.info(
                "parsed locally", extra={"intent": local.intent, "source": "local"}
            )
            return local

        try:
            future = self._queue.submit(text)
        except QueueFullError as exc:
            self.stats.record("rejected")
            logger.warning("external-call queue full, rejecting", extra={"reason": str(exc)})
            return ParsedCommand(
                intent=_SENTINEL_UNAVAILABLE,
                parameters={},
                source="api",
                raw_text=text,
                error=str(exc),
            )

        wait_s = max_wait_s if max_wait_s is not None else self._config.rate_limit.max_wait_seconds
        try:
            result = future.result(timeout=wait_s)
        except FutureTimeoutError:
            self.stats.record("pending")
            logger.info(
                "external call still pending after max_wait_s", extra={"max_wait_s": wait_s}
            )
            return ParsedCommand(
                intent=_SENTINEL_PENDING,
                parameters={},
                source="api",
                raw_text=text,
                error=f"still processing after {wait_s}s -- try again shortly",
            )
        except Exception as exc:  # noqa: BLE001 -- any tier-2 failure must produce a
            self.stats.record("error")  # ParsedCommand, never propagate to the caller
            logger.warning("external call failed", extra={"reason": str(exc)})
            return ParsedCommand(
                intent=_SENTINEL_ERROR, parameters={}, source="api", raw_text=text, error=str(exc)
            )

        self.stats.record("api")
        logger.info("parsed via api", extra={"intent": result.intent, "source": "api"})
        return result

    def log_summary(self) -> dict[str, object]:
        """Emit one INFO log line with the current local-vs-API split, and return it."""
        summary = self.stats.summary()
        logger.info("parse stats summary", extra=summary)
        return summary

    def save_stats(self, path: str | Path) -> None:
        """Write `self.stats.summary()` to `path` as JSON."""
        Path(path).write_text(json.dumps(self.stats.summary(), indent=2))

    def shutdown(self) -> None:
        """Stop the background external-call worker thread."""
        self._queue.shutdown()

    def __enter__(self) -> LLMParser:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
