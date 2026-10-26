"""Logging layer -- one structured-JSON logging setup, shared by every process (week 12).

Composition, not new code: a re-export of week 08's `JSONFormatter`/
`configure_logging` (`src/api/logging_config.py`), which `assistant/`'s
scripts and `chat_app.py` (week 11) already reuse directly. This package
exists so `vision`/`nlp`/`robot`/`database`/`scenarios` can all import
logging from one path (`src.logging`) at the same level they live at,
rather than every non-API layer reaching sideways into `src.api`'s own
module for something that isn't API-specific.

Named `src/logging/`, not `src/log.py` or similar -- deliberately, to
confirm it doesn't collide with the stdlib `logging` module: Python 3's
imports are absolute by default, so a bare `import logging` anywhere in
this codebase (including inside this very package) still resolves to
the standard library, never to `src.logging`, which is only reachable
via that fully-qualified name. Verified for real, not just reasoned
about -- see `tests/test_logging_layer.py`.
"""

from __future__ import annotations

from src.api.logging_config import JSONFormatter, configure_logging

__all__ = ["configure_logging", "JSONFormatter"]
