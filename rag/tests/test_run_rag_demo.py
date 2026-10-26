"""Tests for `scripts/run_rag_demo.py` -- the week 19 one-command deliverable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import assert_reaches_network_boundary, load_script_module

run_rag_demo = load_script_module("run_rag_demo.py")


def _fake_encode(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        is_camera = "camera" in text.lower() or "камер" in text.lower()
        vectors.append([1.0, 0.0] if is_camera else [0.0, 1.0])
    return np.array(vectors)


def test_run_rag_demo_with_fake_encoder(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("## Camera\n\nCamera troubleshooting steps.\n", encoding="utf-8")

    report = run_rag_demo.run_rag_demo(
        document_paths=(str(doc_path),),
        questions=("What about the camera?", "unrelated billing question"),
        encode_fn_factory=lambda: _fake_encode,
    )

    assert report["num_chunks_indexed"] == 1
    assert report["results"][0]["is_grounded"] is True
    assert report["results"][1]["is_grounded"] is False


def test_main_writes_report(tmp_path: Path, monkeypatch) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("## Camera\n\nCamera troubleshooting steps.\n", encoding="utf-8")
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(
        run_rag_demo,
        "parse_args",
        lambda: run_rag_demo.argparse.Namespace(output=str(output_path)),
    )
    monkeypatch.setattr(run_rag_demo, "DEFAULT_DOCUMENT_PATHS", (str(doc_path),))
    monkeypatch.setattr(
        run_rag_demo,
        "run_rag_demo",
        lambda document_paths, questions: {
            "document_paths": list(document_paths),
            "num_chunks_indexed": 1,
            "results": [
                {"question": q, "answer": "x", "sources": [], "is_grounded": False}
                for q in questions
            ],
        },
    )

    run_rag_demo.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["num_chunks_indexed"] == 1
    assert len(report["results"]) == len(run_rag_demo.DEMO_QUESTIONS)


def test_run_rag_demo_real_default_encoder_reaches_network_boundary() -> None:
    def real_default_factory():
        from rag.integration import _get_default_encode_fn

        return _get_default_encode_fn()

    def call_it():
        run_rag_demo.run_rag_demo(
            document_paths=("docs/runbook.md",),
            questions=("test question",),
            encode_fn_factory=real_default_factory,
        )

    assert_reaches_network_boundary(call_it)
