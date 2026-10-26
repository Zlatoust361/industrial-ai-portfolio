"""Tests for `rag.answer.build_answer`."""

from __future__ import annotations

from rag.answer import NOT_FOUND_TEXT, build_answer
from rag.ingestion import DocumentChunk
from rag.retriever import RetrievedChunk


def test_build_answer_with_no_retrieved_chunks_is_not_grounded() -> None:
    answer = build_answer([])
    assert answer.is_grounded is False
    assert answer.text == NOT_FOUND_TEXT
    assert answer.sources == []


def test_build_answer_quotes_the_top_chunk_verbatim() -> None:
    chunk = DocumentChunk(
        text="This is the regulation's actual text.", source_path="doc.md", section="S1"
    )
    answer = build_answer([RetrievedChunk(chunk=chunk, similarity=0.9)])

    assert answer.is_grounded is True
    assert answer.text == chunk.text
    assert answer.sources == [("doc.md", "S1")]


def test_build_answer_respects_max_chunks() -> None:
    chunks = [
        DocumentChunk(text="first", source_path="doc.md", section="S1"),
        DocumentChunk(text="second", source_path="doc.md", section="S2"),
    ]
    retrieved = [RetrievedChunk(chunk=c, similarity=0.9 - 0.1 * i) for i, c in enumerate(chunks)]

    one_chunk = build_answer(retrieved, max_chunks=1)
    two_chunks = build_answer(retrieved, max_chunks=2)

    assert "second" not in one_chunk.text
    assert "second" in two_chunks.text
    assert len(two_chunks.sources) == 2
