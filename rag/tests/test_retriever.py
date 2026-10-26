"""Tests for `rag.retriever.retrieve` -- pure numpy, no CLIP."""

from __future__ import annotations

import numpy as np

from rag.index import RagIndex
from rag.ingestion import DocumentChunk
from rag.retriever import retrieve


def _normalized(vector: list[float]) -> np.ndarray:
    array = np.array(vector)
    return array / np.linalg.norm(array)


def _index_with(vectors: list[list[float]]) -> RagIndex:
    chunks = [
        DocumentChunk(text=f"chunk {i}", source_path="doc.md", section=f"S{i}")
        for i in range(len(vectors))
    ]
    embeddings = np.array([_normalized(v) for v in vectors])
    return RagIndex(chunks=chunks, embeddings=embeddings)


def test_retrieve_returns_most_similar_first() -> None:
    index = _index_with([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = _normalized([1.0, 0.0])

    results = retrieve(query, index, top_k=3, min_similarity=0.0)

    assert [r.chunk.section for r in results] == ["S0", "S1", "S2"]
    assert results[0].similarity >= results[1].similarity >= results[2].similarity


def test_retrieve_drops_results_below_min_similarity() -> None:
    index = _index_with([[1.0, 0.0], [0.0, 1.0]])
    query = _normalized([1.0, 0.0])

    results = retrieve(query, index, top_k=3, min_similarity=0.9)

    assert len(results) == 1
    assert results[0].chunk.section == "S0"


def test_retrieve_returns_empty_when_nothing_clears_threshold() -> None:
    index = _index_with([[0.0, 1.0], [0.0, 1.0]])
    query = _normalized([1.0, 0.0])  # orthogonal to everything in the index

    results = retrieve(query, index, top_k=3, min_similarity=0.2)
    assert results == []


def test_retrieve_respects_top_k() -> None:
    index = _index_with([[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]])
    query = _normalized([1.0, 0.0])

    results = retrieve(query, index, top_k=1, min_similarity=0.0)
    assert len(results) == 1
    assert results[0].chunk.section == "S0"
