"""Tests for `rag.index.RagIndex` -- fake encoder, no real CLIP."""

from __future__ import annotations

import numpy as np
import pytest

from rag.index import RagIndex
from rag.ingestion import DocumentChunk


def _fake_encode(texts: list[str]) -> np.ndarray:
    # Deterministic: embedding = [len(text), 0, 0, ...] normalized -- just needs to be
    # a real, distinguishable vector per distinct text for these tests.
    return np.array([[float(len(text)), 1.0, 0.0] for text in texts])


def test_build_embeds_every_chunk() -> None:
    chunks = [
        DocumentChunk(text="short", source_path="a.md", section="A"),
        DocumentChunk(text="a much longer piece of text", source_path="a.md", section="B"),
    ]
    index = RagIndex.build(chunks, _fake_encode)

    assert len(index) == 2
    assert index.embeddings.shape == (2, 3)
    np.testing.assert_array_equal(index.chunks, chunks)


def test_build_rejects_empty_chunk_list() -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        RagIndex.build([], _fake_encode)
