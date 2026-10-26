#!/usr/bin/env python3
"""Week 19's whole non-interactive deliverable, one command.

Ingests this repository's REAL `docs/runbook.md`/`docs/architecture.md`
(not synthetic text -- these are real documents written for week 10/08),
builds a real CLIP-text-encoder-backed index, and asks it the exact
questions `course/ROADMAP-WEEKS-13-24.md`'s week 19 entry names
("почему сработал quality gate", "что делать при обрыве камеры"), plus
one deliberately unrelated question -- to prove the grounded/ungrounded
distinction actually works, not just the grounded happy path.

Usage:
    PYTHONPATH=. python3 scripts/run_rag_demo.py [--output outputs/rag_demo/report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RAG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAG_ROOT.parent))

from rag.index import RagIndex, TextEncoderFn  # noqa: E402
from rag.ingestion import ingest_markdown_documents  # noqa: E402
from rag.integration import DEFAULT_DOCUMENT_PATHS, answer_question  # noqa: E402

DEFAULT_OUTPUT = "outputs/rag_demo/report.json"

DEMO_QUESTIONS: tuple[str, ...] = (
    "Почему сработал quality gate и что делать?",
    "Что делать при обрыве камеры?",
    "Какая погода сегодня в Новосибирске?",  # deliberately unrelated -- must NOT be "answered"
)


def run_rag_demo(
    document_paths: tuple[str, ...],
    questions: tuple[str, ...],
    encode_fn_factory: object | None = None,
) -> dict[str, object]:
    """Ingest `document_paths`, build the index, answer every `questions` entry.

    Args:
        document_paths: Passed to `rag.ingestion.ingest_markdown_documents`.
        questions: Questions to ask, in order.
        encode_fn_factory: Zero-argument callable returning a
            `TextEncoderFn`. `None` builds the REAL CLIP text encoder
            (`rag.integration._get_default_encode_fn`) -- this is the
            one real network-reaching call path; tests inject a fake
            factory instead, same injectable-call pattern every other
            week's demo script uses.

    Returns:
        A JSON-serializable report: one entry per question, with its
        answer text, sources, and `is_grounded`.
    """
    if encode_fn_factory is None:
        from rag.integration import _get_default_encode_fn

        encode_fn_factory = _get_default_encode_fn

    encode_fn: TextEncoderFn = encode_fn_factory()
    chunks = ingest_markdown_documents(list(document_paths))
    index = RagIndex.build(chunks, encode_fn)

    results = []
    for question in questions:
        answer = answer_question(question, index, encode_fn)
        results.append(
            {
                "question": question,
                "answer": answer.text,
                "sources": answer.sources,
                "is_grounded": answer.is_grounded,
            }
        )

    return {
        "document_paths": list(document_paths),
        "num_chunks_indexed": len(chunks),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_rag_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the demo, print each question's grounded/ungrounded status, write the full report."""
    args = parse_args()
    report = run_rag_demo(DEFAULT_DOCUMENT_PATHS, DEMO_QUESTIONS)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Indexed {report['num_chunks_indexed']} chunk(s) from {report['document_paths']}.")
    for result in report["results"]:
        status = "GROUNDED" if result["is_grounded"] else "NOT FOUND"
        print(f"  [{status}] {result['question']!r} -> sources={result['sources']}")
    print(f"Report -> {output_path}")


if __name__ == "__main__":
    main()
