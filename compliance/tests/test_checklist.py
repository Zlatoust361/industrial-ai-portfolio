"""Tests for `checklist.py`."""

from __future__ import annotations

from compliance.checklist import (
    DEFAULT_CHECKLIST,
    ChecklistStatus,
    checklist_summary,
    render_checklist_markdown,
)


def test_default_checklist_items_have_unique_ids() -> None:
    ids = [item.item_id for item in DEFAULT_CHECKLIST]
    assert len(ids) == len(set(ids))


def test_default_checklist_items_all_have_nonempty_justification_and_source() -> None:
    for item in DEFAULT_CHECKLIST:
        assert item.justification.strip()
        assert item.source.strip()


def test_checklist_summary_counts_match_default_checklist() -> None:
    summary = checklist_summary()
    assert sum(summary.values()) == len(DEFAULT_CHECKLIST)
    assert set(summary.keys()) == {status.value for status in ChecklistStatus}


def test_checklist_summary_counts_custom_items() -> None:
    from compliance.checklist import ChecklistItem

    items = (
        ChecklistItem("X-1", "desc", ChecklistStatus.OPEN, "why", "src"),
        ChecklistItem("X-2", "desc", ChecklistStatus.OPEN, "why", "src"),
        ChecklistItem("X-3", "desc", ChecklistStatus.CLOSED, "why", "src"),
    )
    summary = checklist_summary(items)
    assert summary == {"closed": 1, "open": 2, "not_applicable": 0}


def test_render_checklist_markdown_lists_every_item_id() -> None:
    rendered = render_checklist_markdown()
    for item in DEFAULT_CHECKLIST:
        assert item.item_id in rendered


def test_render_checklist_markdown_is_a_table() -> None:
    rendered = render_checklist_markdown()
    lines = rendered.splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").strip()) <= {"-"}
