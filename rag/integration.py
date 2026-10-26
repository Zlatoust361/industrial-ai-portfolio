"""Ties ingestion + index + retrieval + answer together, plus the real default path.

`answer_question` is the fully-injectable, fully-testable core (no
network, no real CLIP) -- `answer_question_with_default_index` is the ONE
real call site this week reaches a real model, lazily building a
module-level cached index/encoder the first time it's actually called
(same lazy-load convention `src.models.clip.CLIPLoader`/every other
week's loader already follows: constructing the caller must never force
a network call before it's actually needed).
"""

from __future__ import annotations

from rag.answer import RAGAnswer, build_answer
from rag.index import RagIndex, TextEncoderFn
from rag.ingestion import ingest_markdown_documents
from rag.retriever import DEFAULT_MIN_SIMILARITY, retrieve

DEFAULT_DOCUMENT_PATHS: tuple[str, ...] = ("docs/runbook.md", "docs/architecture.md")
"""The real documents this repository already has -- see `rag/README.md`'s
Honesty note for why a real factory's actual regulations aren't
available in this sandbox, and this is the closest real stand-in."""


def answer_question(
    question: str,
    index: RagIndex,
    encode_fn: TextEncoderFn,
    top_k: int = 3,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_chunks: int = 1,
) -> RAGAnswer:
    """Embed `question`, retrieve, build the grounded (or honestly ungrounded) answer.

    Args:
        question: The operator's free-text question.
        index: A built `RagIndex`.
        encode_fn: Same encoder `index` was built with -- embedding the
            query with a DIFFERENT model than the index would make the
            similarity scores meaningless (see `rag.retriever.retrieve`'s
            docstring).
        top_k: Passed to `rag.retriever.retrieve`.
        min_similarity: Passed to `rag.retriever.retrieve`.
        max_chunks: Passed to `rag.answer.build_answer`.

    Returns:
        A `RAGAnswer`.
    """
    query_embedding = encode_fn([question])[0]
    retrieved = retrieve(query_embedding, index, top_k, min_similarity)
    return build_answer(retrieved, max_chunks)


_default_index: RagIndex | None = None
_default_encode_fn: TextEncoderFn | None = None


def _get_default_encode_fn() -> TextEncoderFn:
    """Lazily build (once) the real CLIP text encoder this week's default path uses.

    Reuses `src.models.clip.CLIPLoader`/`CLIPInference` directly (weeks
    01/02's text branch) -- not a new embedding model. This is where a
    real run reaches `huggingface.co` (see `rag/README.md`'s Honesty note).
    """
    global _default_encode_fn
    if _default_encode_fn is None:
        from src.config import load_config
        from src.models.clip import CLIPInference, CLIPLoader

        config = load_config()
        inference = CLIPInference(CLIPLoader(config.model, config.memory))

        def encode(texts: list[str]):
            return inference.encode_text(texts).vectors

        _default_encode_fn = encode
    return _default_encode_fn


def get_default_runbook_index() -> RagIndex:
    """Lazily build (once) the real index over `DEFAULT_DOCUMENT_PATHS`.

    Cached at module level -- rebuilt only once per process, not once per
    question. `reset_default_index` below exists so tests don't leak
    this cache across test cases.
    """
    global _default_index
    if _default_index is None:
        encode_fn = _get_default_encode_fn()
        chunks = ingest_markdown_documents(list(DEFAULT_DOCUMENT_PATHS))
        _default_index = RagIndex.build(chunks, encode_fn)
    return _default_index


def reset_default_index() -> None:
    """Clear the module-level cache -- call this between tests that touch the default path."""
    global _default_index, _default_encode_fn
    _default_index = None
    _default_encode_fn = None


def answer_question_with_default_index(
    question: str,
    top_k: int = 3,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_chunks: int = 1,
) -> RAGAnswer:
    """`answer_question`, against the real default index/encoder -- the one real network path.

    This is what `assistant.action_executor.ActionExecutor`'s `ask_runbook`
    handler calls when no `rag_answer_fn` was injected (see
    `rag/README.md`'s integration section).
    """
    index = get_default_runbook_index()
    encode_fn = _get_default_encode_fn()
    return answer_question(question, index, encode_fn, top_k, min_similarity, max_chunks)
