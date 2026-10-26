"""Tests for `scripts/generate_defect_comparison_report.py` (week 05, day 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.generate_defect_comparison_report import (
    aggregate_metric_stats,
    main,
    method_b_metrics,
    pick_examples,
    render_report,
)


def _entry(identifier: str, status: str, score: float, **metric_overrides: float) -> dict:
    metrics = {
        "psnr_outside_mask": 100.0,
        "ssim_outside_mask": 0.999,
        "blur_inside_mask": 5.0,
        "contrast_inside_mask": 3.0,
        **metric_overrides,
    }
    return {
        "identifier": identifier,
        "status": status,
        "composite_score": score,
        "metrics": metrics,
    }


# -- pick_examples -------------------------------------------------------------


def test_pick_examples_selects_best_median_worst_by_score() -> None:
    results = [
        _entry("a", "rejected", 0.1),
        _entry("b", "selected", 0.9),
        _entry("c", "selected", 0.5),
    ]
    picked = pick_examples(results)
    labels = [label for label, _ in picked]
    identifiers = [entry["identifier"] for _, entry in picked]

    assert labels == ["best", "median", "worst"]
    assert identifiers == ["b", "c", "a"]


def test_pick_examples_handles_a_single_result() -> None:
    results = [_entry("only", "selected", 0.5)]
    picked = pick_examples(results)
    assert [entry["identifier"] for _, entry in picked] == ["only", "only", "only"]


# -- aggregate_metric_stats ----------------------------------------------------


def test_aggregate_metric_stats_counts_and_metric_ranges() -> None:
    results = [
        _entry("a", "selected", 0.9, blur_inside_mask=10.0),
        _entry("b", "selected", 0.8, blur_inside_mask=20.0),
        _entry("c", "rejected", 0.1, blur_inside_mask=0.0),
    ]
    stats = aggregate_metric_stats(results)

    assert stats["count"] == {"selected": 2, "rejected": 1}
    assert stats["metrics"]["blur_inside_mask"]["mean"] == pytest.approx(10.0)
    assert stats["metrics"]["blur_inside_mask"]["min"] == 0.0
    assert stats["metrics"]["blur_inside_mask"]["max"] == 20.0


# -- method_b_metrics -----------------------------------------------------------


def test_method_b_metrics_missing_files_returns_none(tmp_path: Path) -> None:
    result = method_b_metrics(tmp_path / "nope", tmp_path / "nope.json")
    assert result is None


def test_method_b_metrics_no_matching_sample_returns_none(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_images"
    sample_dir.mkdir()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(sample_dir / "defect_sample_000000.jpg")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [{"sample_id": "sample_000099"}]}))

    assert method_b_metrics(sample_dir, manifest_path) is None


def test_method_b_metrics_computes_real_blur_contrast_from_defect_boxes(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_images"
    sample_dir.mkdir()
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(image).save(sample_dir / "defect_sample_000000.jpg")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample_000000",
                        "defect_boxes": [[10, 10, 30, 30]],
                    }
                ]
            }
        )
    )

    result = method_b_metrics(sample_dir, manifest_path)
    assert result is not None
    assert result["source_sample_id"] == "sample_000000"
    assert result["defect_boxes"] == [[10, 10, 30, 30]]
    assert result["blur_inside_mask"] >= 0.0
    assert result["contrast_inside_mask"] >= 0.0


def test_method_b_metrics_against_the_real_committed_week06_artifacts() -> None:
    # Real repo data, not a fixture -- proves this actually works against
    # what week 06 committed, not just a synthetic stand-in.
    sample_dir = Path("outputs/week06/sample_images")
    manifest_path = Path("outputs/week06/manifest_split_demo.json")
    if not (sample_dir.exists() and manifest_path.exists()):
        pytest.skip("outputs/week06/ demo artifacts not present in this checkout")

    result = method_b_metrics(sample_dir, manifest_path)
    assert result is not None
    assert result["source_sample_id"] == "sample_000000"
    assert result["blur_inside_mask"] > 0.0
    assert result["contrast_inside_mask"] > 0.0


# -- render_report ---------------------------------------------------------------


def test_render_report_includes_all_expected_sections() -> None:
    results = [_entry("a", "selected", 0.9), _entry("b", "rejected", 0.1)]
    stats = aggregate_metric_stats(results)
    examples = pick_examples(results)
    for _, entry in examples:
        entry["triptych_relative"] = f"images/{entry['identifier']}.png"

    text = render_report(stats, examples, method_b=None, inpainted_manifest_path=Path("m.json"))

    assert "# Real vs. synthetic defects" in text
    assert "## Method A: mask + inpainting" in text
    assert "## Method B: DR scene + blended parametric shape" in text
    assert "**Unavailable**" in text  # method_b=None
    assert "## The path to the edge" in text
    assert "dataset_pipeline/scripts/01_generate_synthetic.py" in text
    assert "train_dr_yolo.py" in text
    assert "docker-compose.jetson.yml" in text


def test_render_report_includes_method_b_section_when_available() -> None:
    results = [_entry("a", "selected", 0.9)]
    stats = aggregate_metric_stats(results)
    examples = pick_examples(results)
    for _, entry in examples:
        entry["triptych_relative"] = "images/a.png"

    method_b = {
        "blur_inside_mask": 42.0,
        "contrast_inside_mask": 7.0,
        "defect_boxes": [[1, 2, 3, 4]],
        "source_sample_id": "sample_000000",
    }
    text = render_report(stats, examples, method_b=method_b, inpainted_manifest_path=Path("m.json"))

    assert "**Unavailable**" not in text
    assert "42.0000" in text
    assert "sample_000000" in text


# -- main() CLI integration ------------------------------------------------------


def _run_cli(args: list[str]) -> None:
    argv = ["generate_defect_comparison_report.py", *args]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv


def test_main_missing_manifest_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(["--inpainted-manifest", str(tmp_path / "nope.json")])
    assert exc_info.value.code != 0


def test_main_empty_results_exits_nonzero(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"results": []}))
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(["--inpainted-manifest", str(manifest_path)])
    assert exc_info.value.code != 0


def _write_fake_inpainted_manifest(tmp_path: Path, count: int) -> Path:
    """A real, self-consistent day-3-style manifest: real PNGs on disk for
    original/mask/inpainted, matching `run_diffusion_pipeline.py`'s schema."""
    images_dir = tmp_path / "day3_images"
    images_dir.mkdir()
    rng = np.random.default_rng(1)

    results = []
    for i in range(count):
        original = Image.fromarray(rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8))
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        inpainted = original.copy()

        original_path = images_dir / f"orig_{i}.png"
        mask_path = images_dir / f"mask_{i}.png"
        inpainted_path = images_dir / f"inpainted_{i}.png"
        original.save(original_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        inpainted.save(inpainted_path)

        results.append(
            {
                "identifier": f"sample_{i}",
                "status": "selected" if i % 4 != 0 else "rejected",
                "composite_score": 1.0 - i * 0.1,
                "metrics": {
                    "psnr_outside_mask": 100.0,
                    "ssim_outside_mask": 0.999,
                    "blur_inside_mask": float(i + 1),
                    "contrast_inside_mask": float(i + 2),
                },
                "image": str(inpainted_path),
                "original_image": str(original_path),
                "mask": str(mask_path),
            }
        )

    manifest_path = tmp_path / "inpainted_manifest.json"
    manifest_path.write_text(json.dumps({"backend": "cv2", "results": results}, indent=2))
    return manifest_path


def test_main_full_run_writes_report_and_images(tmp_path: Path) -> None:
    manifest_path = _write_fake_inpainted_manifest(tmp_path, count=6)
    output_dir = tmp_path / "report"

    _run_cli(
        [
            "--inpainted-manifest",
            str(manifest_path),
            "--week06-sample-dir",
            str(tmp_path / "does_not_exist"),
            "--week06-manifest",
            str(tmp_path / "does_not_exist.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    report_text = (output_dir / "report.md").read_text()
    assert "# Real vs. synthetic defects" in report_text
    assert "**Unavailable**" in report_text  # no week06 fixtures given

    summary = json.loads((output_dir / "metrics_summary.json").read_text())
    assert summary["method_a"]["count"] == {"selected": 4, "rejected": 2}
    assert summary["method_b"] is None

    triptychs = list((output_dir / "images").glob("*_triptych.png"))
    assert len(triptychs) == 3  # best, median, worst
    for path in triptychs:
        with Image.open(path) as img:
            assert img.width > 0 and img.height > 0
