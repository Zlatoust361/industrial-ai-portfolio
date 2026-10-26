"""CLI: run the full parse -> execute pipeline against example operator commands (week 11, day 3).

Seeds a few demo detections (synthetic -- no client data, same
convention as every other week), then runs the two commands week 11's
own deliverable description names verbatim -- "покажи последние 50
дефектов crack с conf>0.9" and "перезапусти обучение" -- plus "статус"
and "помощь", each through the REAL `LLMParser` -> `ActionExecutor`
pipeline (day 2 + day 3, wired together).

Usage:
    PYTHONPATH=. python assistant/scripts/run_action_demo.py \\
        [--output-dir outputs/assistant_demo]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from assistant.action_executor import ActionExecutor
from assistant.llm_parser import LLMParser
from src.api.logging_config import configure_logging

DEMO_COMMANDS: tuple[str, ...] = (
    "покажи последние 50 дефектов crack с conf>0.9",
    "статус",
    "перезапусти обучение",
    "помощь",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_action_demo.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/assistant_demo")
    return parser.parse_args()


def _seed_demo_detections(executor: ActionExecutor) -> None:
    """Synthetic detections -- no client data, matching this repo's session-wide rule."""
    for defect_type, confidence in (
        ("crack", 0.97),
        ("crack", 0.93),
        ("crack", 0.85),  # below the 0.9 filter used below -- should NOT show up
        ("scratch", 0.99),
        ("pit", 0.80),
    ):
        executor.record_detection(defect_type, confidence, source="demo")


def main() -> None:
    """Seed demo detections, run every DEMO_COMMANDS through parse -> execute, save results."""
    args = parse_args()
    configure_logging(logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with LLMParser() as parser, ActionExecutor() as executor:
        _seed_demo_detections(executor)

        for text in DEMO_COMMANDS:
            parsed = parser.parse(text)
            outcome = executor.execute(parsed)
            results.append(
                {
                    "text": text,
                    "parsed_intent": parsed.intent,
                    "parsed_source": parsed.source,
                    "success": outcome.success,
                    "message": outcome.message,
                    "data": outcome.data,
                }
            )
            print(f"> {text}")
            print(f"  parsed: intent={parsed.intent} source={parsed.source}")
            print(f"  executed: success={outcome.success} message={outcome.message}")
            print()

        stats = parser.log_summary()

    (output_dir / "action_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    (output_dir / "action_parse_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Results written to '{output_dir / 'action_results.json'}'.")
    print(f"Parse stats written to '{output_dir / 'action_parse_stats.json'}'.")


if __name__ == "__main__":
    main()
