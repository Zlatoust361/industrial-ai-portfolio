"""Integration test for `scripts/run_diffusion_pipeline.py` (week 05, day 3).

Real end to end with the `cv2` backend -- no torch/GPU/network needed, so
unlike most of this repo's CLI scripts, this one is fully exercisable
here: real masks, a real (deterministic synthetic) base image per mask,
real `cv2.inpaint` calls, real quality-metric computation, a real
top-75%-selection split, real PNG/manifest.json files written to disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.run_diffusion_pipeline import main
from src.synthetic.config import DefectsConfig, ShapeSpec
from src.synthetic.diffusion_pipeline import synthesize_placeholder_surface
from src.synthetic.mask_generation import generate_defect_mask

SHAPE = ShapeSpec(name="pit", type="ellipse", weight=1.0, size_range=(10, 16))


def _write_mask_manifest(
    mask_dir: Path, count: int, image_size: tuple[int, int] = (64, 64)
) -> Path:
    """Writes `count` real mask PNGs + a manifest.json matching
    `scripts/generate_defect_masks.py`'s schema (week 05 day 2)."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    config = DefectsConfig(seed=0, margin_fraction=0.1, count_range=(1, 1), shapes=(SHAPE,))
    rng = np.random.default_rng(0)

    entries = []
    for i in range(count):
        mask, _instances = generate_defect_mask(image_size, config, rng=rng)
        mask_path = mask_dir / f"image_{i:03d}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        entries.append(
            {"mask": str(mask_path), "image_size": list(image_size), "coverage_fraction": 0.1}
        )

    manifest_path = mask_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"config": "test", "seed": 0, "masks": entries}, indent=2))
    return manifest_path


def _run_cli(args: list[str]) -> None:
    argv = ["run_diffusion_pipeline.py", *args]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv


def test_full_pipeline_synthetic_base_images_end_to_end(tmp_path: Path) -> None:
    manifest_path = _write_mask_manifest(tmp_path / "masks", count=8)
    output_dir = tmp_path / "inpainted"

    _run_cli(
        [
            "--mask-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    result_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result_manifest["backend"] == "cv2"
    assert len(result_manifest["results"]) == 8
    selected = [r for r in result_manifest["results"] if r["status"] == "selected"]
    rejected = [r for r in result_manifest["results"] if r["status"] == "rejected"]
    assert len(selected) == 6  # floor(8 * 0.75)
    assert len(rejected) == 2

    for entry in result_manifest["results"]:
        image_path = Path(entry["image"])
        original_path = Path(entry["original_image"])
        mask_path = Path(entry["mask"])
        assert image_path.exists()
        assert original_path.exists()
        assert mask_path.exists()
        with Image.open(image_path) as img:
            assert img.size == (64, 64)
        with Image.open(original_path) as original:
            assert original.size == (64, 64)
        assert set(entry["metrics"]) == {
            "psnr_outside_mask",
            "ssim_outside_mask",
            "blur_inside_mask",
            "contrast_inside_mask",
        }

    # Selected results are written in descending composite-score order.
    scores = [r["composite_score"] for r in selected]
    assert scores == sorted(scores, reverse=True)
    # Every selected result scores at least as well as every rejected one.
    assert min(scores) >= max(r["composite_score"] for r in rejected)


def test_saved_original_image_matches_the_deterministic_placeholder_surface(tmp_path: Path) -> None:
    manifest_path = _write_mask_manifest(tmp_path / "masks", count=3)
    output_dir = tmp_path / "inpainted"

    _run_cli(
        ["--mask-manifest", str(manifest_path), "--output-dir", str(output_dir), "--seed", "5"]
    )

    result_manifest = json.loads((output_dir / "manifest.json").read_text())
    by_identifier = {r["identifier"]: r for r in result_manifest["results"]}

    for i in range(3):
        identifier = f"image_{i:03d}"
        saved_original = np.array(Image.open(by_identifier[identifier]["original_image"]))
        expected = np.array(synthesize_placeholder_surface((64, 64), seed=5 + i))
        assert np.array_equal(saved_original, expected)


def test_full_pipeline_with_real_base_images(tmp_path: Path) -> None:
    manifest_path = _write_mask_manifest(tmp_path / "masks", count=2)
    image_dir = tmp_path / "base_images"
    image_dir.mkdir()
    synthesize_placeholder_surface((64, 64), seed=100).save(image_dir / "image_000.png")
    # image_001's base image intentionally missing -- exercise the skip path.

    output_dir = tmp_path / "inpainted"
    _run_cli(
        [
            "--mask-manifest",
            str(manifest_path),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    result_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert len(result_manifest["results"]) == 1  # only image_000 had a base image
    assert result_manifest["results"][0]["identifier"] == "image_000"


def test_missing_mask_manifest_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(["--mask-manifest", str(tmp_path / "does_not_exist.json")])
    assert exc_info.value.code != 0


def test_empty_mask_manifest_exits_nonzero(tmp_path: Path) -> None:
    manifest_path = tmp_path / "empty_manifest.json"
    manifest_path.write_text(json.dumps({"config": "test", "seed": 0, "masks": []}))

    with pytest.raises(SystemExit) as exc_info:
        _run_cli(["--mask-manifest", str(manifest_path)])
    assert exc_info.value.code != 0


def test_diffusers_backend_override_exits_nonzero_in_this_sandbox(tmp_path: Path) -> None:
    # Real check: this sandbox has no torch/diffusers, so overriding to
    # --backend diffusers must fail cleanly with a nonzero exit, not a
    # traceback -- the same "smoke test to the exact boundary" pattern
    # used throughout this repo for hardware/GPU-dependent code.
    manifest_path = _write_mask_manifest(tmp_path / "masks", count=1)
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(
            [
                "--mask-manifest",
                str(manifest_path),
                "--backend",
                "diffusers",
                "--output-dir",
                str(tmp_path / "inpainted"),
            ]
        )
    assert exc_info.value.code != 0


def test_keep_fraction_zero_selects_nothing(tmp_path: Path) -> None:
    manifest_path = _write_mask_manifest(tmp_path / "masks", count=4)
    diffusion_config_path = tmp_path / "diffusion.yaml"
    real_config = Path("config/diffusion.yaml").read_text()
    patched = real_config.replace("keep_fraction: 0.75", "keep_fraction: 0.0")
    diffusion_config_path.write_text(patched)

    output_dir = tmp_path / "inpainted"
    _run_cli(
        [
            "--mask-manifest",
            str(manifest_path),
            "--diffusion-config",
            str(diffusion_config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    result_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result_manifest["keep_fraction"] == 0.0
    assert all(r["status"] == "rejected" for r in result_manifest["results"])
