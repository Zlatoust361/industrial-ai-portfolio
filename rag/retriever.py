"""Cosine-similarity retrieval over a `RagIndex`, with an explicit "nothing relevant" outcome.

Pure NumPy -- works on ALREADY-COMPUTED embeddings, no encoder/network
call inside this module (the query is embedded once by the caller, see
`rag.integration.answer_question`). This is what makes `retrieve` itself
fully unit-testable without a real CLIP model, same separation
`active_learning.incremental_train` keeps from CLIP encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.index import RagIndex
from rag.ingestion import DocumentChunk

DEFAULT_MIN_SIMILARITY = 0.2
"""Below this cosine similarity, a chunk is treated as NOT relevant --
see `retrieve`'s docstring for why this matters more than it looks like
it should."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk retrieved for a query, with its similarity score."""

    chunk: DocumentChunk
    similarity: float


def retrieve(
    query_embedding: np.ndarray,
    index: RagIndex,
    top_k: int = 3,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[RetrievedChunk]:
    """Find the most similar chunks to `query_embedding`, or none at all.

    Args:
        query_embedding: Shape `(embedding_dim,)`, L2-normalized (same
            encoder as `index.embeddings` -- a query encoded by a
            DIFFERENT model than the index would make the dot product
            meaningless, not just less accurate).
        index: A built `RagIndex`.
        top_k: Maximum chunks to return.
        min_similarity: Chunks below this score are dropped, not
            returned as low-confidence matches. This is the one thing
            that lets `rag.answer.build_answer` tell "the regulation
            doesn't cover this" apart from "the regulation's answer just
            happens to be the least-bad of an irrelevant corpus" --
            without this threshold, `retrieve` would ALWAYS return
            `top_k` chunks for ANY query, however unrelated, and the
            caller could never distinguish a genuine miss from a weak hit.

    Returns:
        Up to `top_k` `RetrievedChunk`s with `similarity >= min_similarity`,
        sorted by similarity descending. Empty if nothing clears the
        threshold -- a real possible outcome, not an error.
    """
    similarities = index.embeddings @ query_embedding
    order = np.argsort(-similarities, kind="stable")

    results: list[RetrievedChunk] = []
    for i in order[: min(top_k, len(order))]:
        if similarities[i] < min_similarity:
            break  # order is descending -- every remaining one is lower still
        results.append(RetrievedChunk(chunk=index.chunks[i], similarity=float(similarities[i])))
    return results
