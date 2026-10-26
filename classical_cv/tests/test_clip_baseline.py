"""Unit tests for `classical_cv.clip_baseline` -- real CLIP load attempts, no mocking.

`test_build_clip_inference_reaches_real_network_boundary` and
`test_run_clip_baseline_or_blocked_reports_blocked_not_a_crash` both make
a REAL attempt to reach `huggingface.co` and are expected to hit this
sandbox's network boundary (see the module's own docstring) -- same
pattern `captioning/tests/test_model.py` already uses for the exact same
checkpoint family.
"""

from __future__ import annotations

from conftest import assert_reaches_network_boundary

from classical_cv.clip_baseline import build_clip_inference, run_clip_baseline_or_blocked
from classical_cv.defect_images import build_demo_dataset
from classical_cv.report import CLIP_STATUS_BLOCKED
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_build_clip_inference_reaches_real_network_boundary() -> None:
    assert_reaches_network_boundary(build_clip_inference)


def test_run_clip_baseline_or_blocked_reports_blocked_not_a_crash() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=2, seed=42)
    summary = run_clip_baseline_or_blocked(dataset)
    assert summary.status == CLIP_STATUS_BLOCKED
    assert summary.accuracy is None
    assert summary.mean_latency_ms is None
    assert summary.blocked_reason
