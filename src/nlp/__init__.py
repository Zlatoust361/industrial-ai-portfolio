"""NLP layer -- the capstone's single entry point for "what did the operator ask for" (week 12).

Composition, not new code: a re-export of week 11's already-built,
already-tested two-tier command parser, so `scenarios/` (day 3) can turn
an operator's text into a validated `ParsedCommand` without reaching into
`assistant.llm_parser`/`assistant.schemas`/`assistant.rate_limiter`
directly.

- `LLMParser`/`match_locally`/`call_claude_api` (week 11 day 2): local
  `re`-based matching first, external Claude API only on a miss, both
  validated against the same schema registry.
- `ParsedCommand`/`validate_parameters`/`KNOWN_INTENTS`/`DEFECT_TYPES`
  (week 11 day 2): the shared schema both tiers -- and any caller
  constructing a command by hand -- validate against.
- `TokenBucketRateLimiter`/`ExternalCallQueue`/`QueueFullError` (week 11
  day 2): the rate-limiting/queue machinery guarding the external tier,
  re-exported for a caller that wants to inspect/tune it directly rather
  than only through `LLMParser`'s own config.
"""

from __future__ import annotations

from assistant.llm_parser import LLMParser, ParseStats, call_claude_api, match_locally
from assistant.rate_limiter import ExternalCallQueue, QueueFullError, TokenBucketRateLimiter
from assistant.schemas import (
    DEFECT_TYPES,
    KNOWN_INTENTS,
    ParsedCommand,
    validate_parameters,
)

__all__ = [
    "LLMParser",
    "ParseStats",
    "match_locally",
    "call_claude_api",
    "ParsedCommand",
    "validate_parameters",
    "KNOWN_INTENTS",
    "DEFECT_TYPES",
    "TokenBucketRateLimiter",
    "ExternalCallQueue",
    "QueueFullError",
]
