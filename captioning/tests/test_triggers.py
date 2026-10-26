"""Tests for `captioning.triggers`."""

from __future__ import annotations

import pytest

from captioning.triggers import (
    TRIGGER_ON_DEFECT_CONFIRMED,
    TRIGGER_ON_EVERY_FRAME,
    should_caption,
)


def test_on_defect_confirmed_fires_only_for_defect() -> None:
    assert should_caption("defect", TRIGGER_ON_DEFECT_CONFIRMED) is True
    assert should_caption("ok", TRIGGER_ON_DEFECT_CONFIRMED) is False


def test_on_every_frame_always_fires() -> None:
    assert should_caption("defect", TRIGGER_ON_EVERY_FRAME) is True
    assert should_caption("ok", TRIGGER_ON_EVERY_FRAME) is True


def test_unknown_trigger_raises() -> None:
    with pytest.raises(ValueError, match="Unknown trigger"):
        should_caption("defect", "on_demand")
