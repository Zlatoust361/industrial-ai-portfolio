"""In-memory vector index over `DocumentChunk`s -- no vector-DB dependency.

A few dozen chunks (the realistic scale for one factory's regulations,
per `course/ROADMAP-WEEKS-13-24.md`'s week 19 entry: "корпус в десятки
документов") fit entirely in memory as a plain NumPy array -- a real
vector database would be solving a problem this scale doesn't have yet.
Embedding happens through an INJECTED encoder function (same pattern as
`captioning.prompting`'s `CaptionFn`), never built inside this module --
`rag/README.md`'s Honesty note explains which encoder the real call site
uses (CLIP's text branch, weeks 01/02's text encoder, not a new model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.ingestion import DocumentChunk


class TextEncoderFn(Protocol):
    """Callable shape `RagIndex.build`/`embed_query` expect for text -> embedding."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class RagIndex:
    """Chunks + their embeddings, parallel arrays in the same order.

    Attributes:
        chunks: One `DocumentChunk` per row of `embeddings`.
        embeddings: Shape `(len(chunks), embedding_dim)`, L2-normalized
            (so a plain dot product with a query embedding IS cosine
            similarity -- see `rag.retriever.retrieve`).
    """

    chunks: list[DocumentChunk]
    embeddings: np.ndarray

    @classmethod
    def build(cls, chunks: list[DocumentChunk], encode_fn: TextEncoderFn) -> RagIndex:
        """Embed every chunk's text and build the index.

        Args:
            chunks: From `rag.ingestion.ingest_markdown_document(s)`.
            encode_fn: Turns a batch of texts into L2-normalized
                embeddings, shape `(len(texts), embedding_dim)`. The real
                call site passes `src.models.clip.CLIPInference.encode_text`
                (already L2-normalizing, see that module); tests inject
                a deterministic fake instead.

        Returns:
            A new `RagIndex`.

        Raises:
            ValueError: If `chunks` is empty -- an index over zero
                documents can't usefully answer anything, and silently
                building one would turn a real ingestion bug (wrong path,
                empty corpus) into "every question returns no match",
                indistinguishable from the corpus genuinely lacking an answer.
        """
        if not chunks:
            raise ValueError(
                "Cannot build a RagIndex over zero chunks -- check that ingestion actually "
                "found documents/sections (empty corpus looks identical to 'no answer found' "
                "otherwise)."
            )
        embeddings = encode_fn([chunk.text for chunk in chunks])
        return cls(chunks=chunks, embeddings=embeddings)

    def __len__(self) -> int:
        return len(self.chunks)
