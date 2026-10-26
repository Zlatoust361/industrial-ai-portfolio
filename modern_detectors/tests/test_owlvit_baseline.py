"""Unit tests for `modern_detectors.owlvit_baseline` -- real OWL-ViT load attempts, no mocking.

Both tests make a REAL attempt to reach `huggingface.co` and are expected
to hit this sandbox's network boundary (see the module's own docstring)
-- same pattern `classical_cv/tests/test_clip_baseline.py` already uses
for the exact same reason.
"""

from __future__ import annotations

from conftest import assert_reaches_network_boundary

from classical_cv.defect_images import build_demo_dataset
from modern_detectors.dataset import resize_dataset
from modern_detectors.owlvit_baseline import build_owlvit_model, run_owlvit_or_blocked
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_build_owlvit_model_reaches_real_network_boundary() -> None:
    assert_reaches_network_boundary(build_owlvit_model)


def test_run_owlvit_or_blocked_reports_blocked_not_a_crash() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=2, seed=42)
    images = resize_dataset(dataset, (512, 512))

    summary = run_owlvit_or_blocked(images)

    assert summary.status == "blocked"
    assert summary.average_precision is None
    assert summary.mean_latency_ms is None
    assert summary.blocked_reason
