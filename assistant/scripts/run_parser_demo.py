"""CLI: run `llm_parser.py` against a batch of example operator commands (week 11, day 2).

Real, not staged: every command below is actually run through
`LLMParser.parse()`. This sandbox has no `ANTHROPIC_API_KEY`, so
commands that miss the local rule-based tier fall through to the queue,
the (nonexistent) worker call fails cleanly with a `RuntimeError` naming
the missing env var, and `parse()` reports that honestly as
`intent="error"` -- not a stub standing in for a real API response, the
actual designed behavior of the API tier when it can't reach Claude.

Usage:
    PYTHONPATH=. python assistant/scripts/run_parser_demo.py \\
        [--commands-file path/to/commands.txt] [--output-dir outputs/assistant_demo]

`--commands-file` is one command per line (blank lines/`#`-comments
ignored); without it, a built-in example set is used (see `DEFAULT_COMMANDS`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from assistant.llm_parser import LLMParser
from src.api.logging_config import configure_logging

DEFAULT_COMMANDS: tuple[str, ...] = (
    "покажи последние 50 дефектов crack с conf>0.9",
    "статус",
    "перезапусти обучение",
    "помощь",
    "list defects with confidence > 0.8",
    "покажи все дефекты",
    "покажи 10 дефектов pit",
    "почему стало больше брака на третьей смене?",
    "что делать если станок сломался",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_parser_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--commands-file", default=None)
    parser.add_argument("--output-dir", default="outputs/assistant_demo")
    parser.add_argument(
        "--max-wait-s",
        type=float,
        default=None,
        help="Override config's rate_limit.max_wait_seconds.",
    )
    return parser.parse_args()


def _load_commands(commands_file: str | None) -> tuple[str, ...]:
    if commands_file is None:
        return DEFAULT_COMMANDS
    lines = Path(commands_file).read_text(encoding="utf-8").splitlines()
    stripped = (line.strip() for line in lines)
    return tuple(line for line in stripped if line and not line.startswith("#"))


def main() -> None:
    """Parse every example command, print each result, save a stats summary."""
    args = parse_args()
    configure_logging(logging.INFO)

    commands = _load_commands(args.commands_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with LLMParser() as parser:
        for text in commands:
            result = parser.parse(text, max_wait_s=args.max_wait_s)
            results.append(
                {
                    "text": text,
                    "intent": result.intent,
                    "parameters": result.parameters,
                    "source": result.source,
                    "is_actionable": result.is_actionable,
                    "error": result.error,
                }
            )
            marker = "OK" if result.is_actionable else "!!"
            error_suffix = f" error={result.error}" if result.error else ""
            print(
                f"[{marker}] {text!r} -> intent={result.intent} source={result.source} "
                f"params={result.parameters}{error_suffix}"
            )

        summary = parser.log_summary()
        parser.save_stats(output_dir / "stats.json")

    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print()
    print(
        f"Local: {summary['local_count']}/{summary['total']} "
        f"({summary['local_fraction']:.0%}); "
        f"API: {summary['api_count']}/{summary['total']} ({summary['api_fraction']:.0%}); "
        f"pending: {summary['pending_count']}; rejected: {summary['rejected_count']}; "
        f"error: {summary['error_count']}"
    )
    print(f"Results written to '{output_dir / 'results.json'}'.")
    print(f"Stats written to '{output_dir / 'stats.json'}'.")


if __name__ == "__main__":
    main()
