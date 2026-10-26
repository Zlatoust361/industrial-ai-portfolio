"""Tests for `rag.integration` -- fake encoder (injectable path), real files (default path)."""

from __future__ import annotations

import numpy as np
import pytest

import rag.integration as integration
from rag.index import RagIndex
from rag.ingestion import DocumentChunk


def _fake_encode(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        is_camera = "camera" in text.lower() or "камер" in text.lower()
        vectors.append([1.0, 0.0] if is_camera else [0.0, 1.0])
    return np.array(vectors)


@pytest.fixture(autouse=True)
def _reset_default_index():
    integration.reset_default_index()
    yield
    integration.reset_default_index()


def test_answer_question_with_injected_encoder() -> None:
    chunks = [
        DocumentChunk(text="Camera troubleshooting steps.", source_path="doc.md", section="Camera"),
        DocumentChunk(text="Unrelated billing policy.", source_path="doc.md", section="Billing"),
    ]
    index = RagIndex.build(chunks, _fake_encode)

    answer = integration.answer_question("What about the camera?", index, _fake_encode)

    assert answer.is_grounded is True
    assert answer.sources == [("doc.md", "Camera")]


def test_answer_question_ungrounded_for_unrelated_query() -> None:
    chunks = [DocumentChunk(text="Camera troubleshooting.", source_path="doc.md", section="Camera")]
    index = RagIndex.build(chunks, _fake_encode)

    # min_similarity default (0.2) combined with this fake encoder's orthogonal
    # vectors means an unrelated query gets similarity 0.0 -- below threshold.
    answer = integration.answer_question("billing question", index, _fake_encode)
    assert answer.is_grounded is False


def test_get_default_runbook_index_is_cached(monkeypatch) -> None:
    build_calls = []
    original_build = RagIndex.build

    def _counting_build(chunks, encode_fn):
        build_calls.append(1)
        return original_build(chunks, encode_fn)

    monkeypatch.setattr(RagIndex, "build", staticmethod(_counting_build))
    monkeypatch.setattr(integration, "_get_default_encode_fn", lambda: _fake_encode)

    integration.get_default_runbook_index()
    integration.get_default_runbook_index()

    assert len(build_calls) == 1  # built once, cached on the second call
