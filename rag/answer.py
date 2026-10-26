"""Builds a grounded answer from retrieved chunks -- extractive, not generative.

Deliberately quotes the regulation's own text verbatim (plus its
citation) instead of asking an LLM to paraphrase/compose across chunks.
For "what does the instruction say" (this week's actual framing, per
`course/ROADMAP-WEEKS-13-24.md`'s week 19 entry), a literal quote is
MORE trustworthy than a generative rewrite, not less -- and it keeps this
week's new logic deterministic and fully testable, instead of adding a
second generative network call next to week 15's captioning VLM. See
`rag/README.md`'s Honesty note for the generative alternative this
deliberately leaves for later (routing through `assistant/llm_parser.py`'s
existing local/cloud split to compose a multi-chunk answer).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.retriever import RetrievedChunk

NOT_FOUND_TEXT = (
    "В доступных документах не найден раздел, релевантный этому вопросу -- "
    "это не предположение модели, а честный результат поиска по регламенту."
)


@dataclass(frozen=True)
class RAGAnswer:
    """One answer to one question, with its grounding explicit.

    Attributes:
        text: The answer text -- a verbatim quote of the top retrieved
            chunk(s) when `is_grounded`, `NOT_FOUND_TEXT` otherwise.
        sources: `(source_path, section)` pairs the answer was built
            from -- empty when `is_grounded` is `False`.
        is_grounded: `True` only when at least one chunk cleared
            `rag.retriever.DEFAULT_MIN_SIMILARITY`. This is the field
            that actually separates "from the document" from "the model
            made it up" -- `text` alone can't carry that distinction
            (fluent, confident-sounding text looks the same either way).
    """

    text: str
    sources: list[tuple[str, str | None]]
    is_grounded: bool


def build_answer(retrieved: list[RetrievedChunk], max_chunks: int = 1) -> RAGAnswer:
    """Build a `RAGAnswer` from `rag.retriever.retrieve`'s output.

    Args:
        retrieved: Already sorted by similarity descending (as
            `retrieve` returns it); empty means nothing cleared the
            relevance threshold.
        max_chunks: How many of the top retrieved chunks to quote. `1`
            (the default) keeps the answer to a single, clearly-sourced
            section -- raise it to compose from more, at the cost of a
            longer, multi-source quote.

    Returns:
        A `RAGAnswer`. `is_grounded=False` with `NOT_FOUND_TEXT` when
        `retrieved` is empty; otherwise the top `max_chunks` chunks'
        text, concatenated, with their `(source_path, section)` as `sources`.
    """
    if not retrieved:
        return RAGAnswer(text=NOT_FOUND_TEXT, sources=[], is_grounded=False)

    top = retrieved[:max_chunks]
    text = "\n\n---\n\n".join(item.chunk.text for item in top)
    sources = [(item.chunk.source_path, item.chunk.section) for item in top]
    return RAGAnswer(text=text, sources=sources, is_grounded=True)
