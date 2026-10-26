"""Tests for `rag.ingestion` -- real files on disk, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion import DocumentChunk, ingest_markdown_document, ingest_markdown_documents

_SAMPLE = """# Document Title

Intro text right under the title.

## 1. First Section

Body of the first section.

### 1a. A subsection

Subsection body -- should stay attached to "1. First Section" at min_header_level=2.

## 2. Second Section

Body of the second section.
"""


def test_ingest_splits_by_header_level(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(_SAMPLE, encoding="utf-8")

    chunks = ingest_markdown_document(doc_path)

    # A level-1 "#" header starts its own chunk too (it's even more general
    # than "##", so it must also be a boundary regardless of min_header_level)
    # -- "Document Title", "1. First Section" (+ subsection), "2. Second Section".
    assert len(chunks) == 3
    assert chunks[0].section == "Document Title"
    assert "Intro text" in chunks[0].text
    assert chunks[1].section == "1. First Section"
    assert "subsection body" in chunks[1].text.lower()  # ### stayed attached
    assert chunks[2].section == "2. Second Section"


def test_ingest_leading_text_with_no_header_at_all_has_section_none(tmp_path: Path) -> None:
    doc_path = tmp_path / "headerless_intro.md"
    content = "Just some text, no header anywhere above it.\n\n## 1. A Section\n\nbody\n"
    doc_path.write_text(content, encoding="utf-8")

    chunks = ingest_markdown_document(doc_path)
    assert chunks[0].section is None
    assert "Just some text" in chunks[0].text


def test_ingest_records_source_path(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(_SAMPLE, encoding="utf-8")

    chunks = ingest_markdown_document(doc_path)
    assert all(chunk.source_path == str(doc_path) for chunk in chunks)
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.page is None for chunk in chunks)


def test_ingest_higher_min_header_level_splits_subsections_too(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(_SAMPLE, encoding="utf-8")

    chunks = ingest_markdown_document(doc_path, min_header_level=3)
    sections = [chunk.section for chunk in chunks]
    assert "1a. A subsection" in sections


def test_ingest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Document not found"):
        ingest_markdown_document(tmp_path / "does_not_exist.md")


def test_ingest_documents_concatenates_in_order(tmp_path: Path) -> None:
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("## A1\n\ntext a\n", encoding="utf-8")
    second.write_text("## B1\n\ntext b\n", encoding="utf-8")

    chunks = ingest_markdown_documents([first, second])
    assert [chunk.source_path for chunk in chunks] == [str(first), str(second)]


def test_ingest_real_runbook_document() -> None:
    """The real deliverable touches a REAL document this repository already has."""
    chunks = ingest_markdown_document("docs/runbook.md")
    assert len(chunks) >= 5  # the runbook's own 5 numbered incident sections
    sections = [chunk.section for chunk in chunks if chunk.section]
    assert any("Camera" in s or "Model quality" in s for s in sections)
