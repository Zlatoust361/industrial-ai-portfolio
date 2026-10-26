"""Markdown document ingestion: split a real doc into retrievable, cited chunks.

Chunks by HEADER STRUCTURE (`##`/`###` sections), not by a fixed
character/token count -- a fixed-size window would routinely cut a
runbook's "Symptom" from its own "Fix" mid-sentence, and more importantly
would make `section` (the thing a citation actually names) meaningless.
Deliberately Markdown-only in this first pass -- `docs/runbook.md`/
`docs/architecture.md` are real Markdown documents already in this
repository, not PDF/DOCX (see `rag/README.md`'s Honesty note for why a
PDF/DOCX parser isn't added before there's a real PDF/DOCX regulation to
parse).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class DocumentChunk:
    """One retrievable section of a document.

    Attributes:
        text: The section's full text, header line included (so the
            retrieved chunk is self-describing even without `section`
            attached separately).
        source_path: Path of the document this chunk came from, as given
            to `ingest_markdown_document` -- NOT resolved to an absolute
            path, so a citation matches what a human would actually type
            to find the file.
        section: The header text this chunk falls under (without the
            leading `#`s), e.g. `"2. Model quality degradation (embeddings
            drift, wrong predictions)"`. `None` for text before the first
            header (typically the document's own title/intro).
        page: Reserved for a future PDF/DOCX ingester (`None` here --
            Markdown has no pages).
    """

    text: str
    source_path: str
    section: str | None
    page: int | None = None


def ingest_markdown_document(path: str | Path, min_header_level: int = 2) -> list[DocumentChunk]:
    """Split one Markdown document into one chunk per header at or above `min_header_level`.

    Args:
        path: Path to a `.md` file.
        min_header_level: The header depth (`1` = `#`, `2` = `##`, ...)
            that starts a new chunk. `2` (the default) matches
            `docs/runbook.md`'s own structure (`##` = incident type,
            `###` = sub-case) -- a `###` subsection stays attached to its
            parent `##` chunk rather than fragmenting into its own,
            smaller citation. Raise this value for a document whose
            sub-sections are specific enough to cite independently. A
            header MORE general than `min_header_level` (e.g. the
            document's own `#` title) ALSO always starts its own chunk --
            it's a boundary at every level at or above the one requested,
            not only exactly at it.

    Returns:
        One `DocumentChunk` per section, in document order. Any text
        before the first qualifying header becomes one chunk with
        `section=None` (typically the document's title/intro) -- dropped
        only if it's empty after stripping.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Document not found: {document_path}")

    lines = document_path.read_text(encoding="utf-8").splitlines()
    source_path = str(path)

    chunks: list[DocumentChunk] = []
    current_section: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            chunk = DocumentChunk(text=text, source_path=source_path, section=current_section)
            chunks.append(chunk)

    for line in lines:
        match = _HEADER_PATTERN.match(line)
        if match and len(match.group(1)) <= min_header_level:
            _flush()
            current_section = match.group(2)
            current_lines = [line]
        else:
            current_lines.append(line)
    _flush()

    return chunks


def ingest_markdown_documents(
    paths: list[str | Path], min_header_level: int = 2
) -> list[DocumentChunk]:
    """`ingest_markdown_document` over several files, concatenated in order given."""
    chunks: list[DocumentChunk] = []
    for path in paths:
        chunks.extend(ingest_markdown_document(path, min_header_level))
    return chunks
