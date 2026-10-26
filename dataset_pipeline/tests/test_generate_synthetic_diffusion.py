"""Unit + script-level tests for `01_generate_synthetic.py --source diffusion` (C.1).

Mirrors `test_script_pipeline.py`'s style (real files/images throughout,
no mocking -- the default `cv2` backend needs neither GPU nor network, so
there's nothing to mock). Covers:

- `--source dr` (default/omitted) is completely unaffected by this
  module's existence -- see `test_script_pipeline.py` for the full
  byte-for-byte-style regression coverage of that path; this file only
  adds coverage for the new `--source diffusion` code path itself.
- `approximate_diffusion_dr_params`/`generate_sample_diffusion`/
  `select_diffusion_candidates`'s pure logic in isolation.
- The full `--source diffusion` output feeding cleanly into stage 2
  (stratification) and stage 8 (validation) -- the actual "does the
  schema reconcile" question C.1 asks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import load_script_module, write_dataset_config
from PIL import Image

generate_synthetic = load_script_module("01_generate_synthetic.py")
stratify_dataset = load_script_module("02_stratify_dataset.py")
validate_dataset_script = load_script_module("08_validate_dataset.py")

from src.synthetic.config import load_defects_config  # noqa: E402
from src.synthetic.diffusion_config import load_diffusion_config  # noqa: E402
from src.synthetic.diffusion_pipeline import build_backend  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFECTS_CONFIG_PATH = REPO_ROOT / "config" / "defects.yaml"
DIFFUSION_CONFIG_PATH = REPO_ROOT / "config" / "diffusion.yaml"


# ---------------------------------------------------------------------------
# Pure-logic unit tests
# ---------------------------------------------------------------------------


def test_approximate_diffusion_dr_params_has_the_shape_stratify_needs() -> None:
    image = Image.new("RGB", (64, 64), color=(180, 100, 50))

    params = generate_synthetic.approximate_diffusion_dr_params(image)

    # Exactly the keys utils.stratify.assign_stratum reads.
    assert "brightness" in params["light"]
    assert "rotation_deg" in params["camera"]
    assert len(params["background_color"]) == 3
    # Honest traceability marker -- not a real DR renderer's output.
    assert params["source"] == "diffusion"
    # No camera/pose concept for a flat inpainted surface -- always "center".
    assert params["camera"]["rotation_deg"] == 0.0


def test_approximate_diffusion_dr_params_background_color_matches_border_mean() -> None:
    # A uniform image's border mean is trivially that same color.
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))

    params = generate_synthetic.approximate_diffusion_dr_params(image)

    assert params["background_color"] == [10, 20, 30]


def _load_configs():
    defects_config = load_defects_config(DEFECTS_CONFIG_PATH)
    diffusion_config = load_diffusion_config(DIFFUSION_CONFIG_PATH)
    backend = build_backend(diffusion_config)
    return defects_config, diffusion_config, backend


def test_generate_sample_diffusion_ok_sample_has_no_mask_and_full_frame_bbox() -> None:
    defects_config, diffusion_config, backend = _load_configs()
    generation_config = _fake_generation_config(defect_fraction=0.0)
    rng = np.random.default_rng(1)

    candidate = generate_synthetic.generate_sample_diffusion(
        0, (128, 96), defects_config, generation_config, diffusion_config, backend, rng
    )

    assert candidate.sample.label == "ok"
    assert candidate.sample.defect_boxes == ()
    assert candidate.sample.object_bbox == (0, 0, 128, 96)
    assert candidate.mask is None
    assert candidate.base_image is None


def test_generate_sample_diffusion_defect_sample_has_mask_and_matching_defect_boxes() -> None:
    defects_config, diffusion_config, backend = _load_configs()
    generation_config = _fake_generation_config(defect_fraction=1.0)
    rng = np.random.default_rng(2)

    candidate = generate_synthetic.generate_sample_diffusion(
        0, (512, 512), defects_config, generation_config, diffusion_config, backend, rng
    )

    assert candidate.sample.label == "defect"
    assert len(candidate.sample.defect_boxes) > 0
    assert candidate.mask is not None
    assert candidate.base_image is not None
    # Inpainted result is the same size as the base placeholder surface.
    assert candidate.image.size == candidate.base_image.size == (512, 512)


def test_select_diffusion_candidates_never_gates_ok_samples() -> None:
    defects_config, diffusion_config, backend = _load_configs()
    generation_config = _fake_generation_config(defect_fraction=0.0)
    rng = np.random.default_rng(3)

    candidates = [
        generate_synthetic.generate_sample_diffusion(
            i, (64, 64), defects_config, generation_config, diffusion_config, backend, rng
        )
        for i in range(5)
    ]
    assert all(c.mask is None for c in candidates)  # sanity: all "ok"

    kept, num_rejected = generate_synthetic.select_diffusion_candidates(
        candidates, diffusion_config
    )

    assert len(kept) == 5
    assert num_rejected == 0


def test_select_diffusion_candidates_keeps_top_fraction_of_defect_candidates() -> None:
    defects_config, diffusion_config, backend = _load_configs()
    generation_config = _fake_generation_config(defect_fraction=1.0)
    rng = np.random.default_rng(4)

    candidates = [
        generate_synthetic.generate_sample_diffusion(
            i, (512, 512), defects_config, generation_config, diffusion_config, backend, rng
        )
        for i in range(8)
    ]
    assert all(c.mask is not None for c in candidates)  # sanity: all "defect"

    kept, num_rejected = generate_synthetic.select_diffusion_candidates(
        candidates, diffusion_config
    )

    expected_keep = int(len(candidates) * diffusion_config.quality_gate.keep_fraction)
    assert len(kept) == expected_keep
    assert num_rejected == len(candidates) - expected_keep
    # Kept candidates stay in original generation order (not re-sorted by score).
    kept_indices = [candidates.index(c) for c in kept]
    assert kept_indices == sorted(kept_indices)


def _fake_generation_config(*, defect_fraction: float):
    from utils.config import GenerationConfig

    return GenerationConfig(
        count=1,
        defect_fraction=defect_fraction,
        object_class_name="part",
        defect_class_name="defect",
        defect_overlay_color=(200, 40, 40),
        defect_overlay_alpha=0.55,
        object_pool_size=1,
        seed=7,
        dr_parameters_path=REPO_ROOT / "config" / "parameters.yaml",
        defects_config_path=DEFECTS_CONFIG_PATH,
        diffusion_config_path=DIFFUSION_CONFIG_PATH,
    )


# ---------------------------------------------------------------------------
# Script-level (CLI/main()) tests
# ---------------------------------------------------------------------------


def test_source_diffusion_generates_a_valid_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_dataset_config(tmp_path, count=20)
    monkeypatch.setattr(
        sys,
        "argv",
        ["01_generate_synthetic.py", "--config", str(config_path), "--source", "diffusion"],
    )

    generate_synthetic.main()

    manifest_path = tmp_path / "data" / "manifest_generated.json"
    manifest = json.loads(manifest_path.read_text())
    samples = manifest["samples"]

    assert 0 < len(samples) <= 20  # quality gate may drop some "defect" candidates
    for sample in samples:
        # Schema honesty: whole-frame object_bbox for every diffusion-sourced sample.
        assert tuple(sample["object_bbox"]) == (0, 0, sample["width"], sample["height"])
        assert sample["dr_params"]["source"] == "diffusion"
        if sample["label"] == "defect":
            assert len(sample["defect_boxes"]) > 0
        else:
            assert sample["defect_boxes"] == []
        image_path = tmp_path / "data" / sample["image_path"]
        assert image_path.is_file()


def test_source_diffusion_output_feeds_stage_02_stratify_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_dataset_config(tmp_path, count=12)
    monkeypatch.setattr(
        sys,
        "argv",
        ["01_generate_synthetic.py", "--config", str(config_path), "--source", "diffusion"],
    )
    generate_synthetic.main()

    monkeypatch.setattr(sys, "argv", ["02_stratify_dataset.py", "--config", str(config_path)])
    stratify_dataset.main()

    stratified = json.loads((tmp_path / "data" / "manifest_stratified.json").read_text())
    assert all(sample["stratum"] is not None for sample in stratified["samples"])
    # Documented, real consequence of the approximated dr_params: angle
    # collapses to "center" for every diffusion-sourced sample.
    assert all(sample["stratum"].split("_")[2] == "center" for sample in stratified["samples"])


def test_source_diffusion_output_passes_stage_08_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_dataset_config(tmp_path, count=16)

    for module, script_name, extra_args in (
        (generate_synthetic, "01_generate_synthetic.py", ["--source", "diffusion"]),
        (stratify_dataset, "02_stratify_dataset.py", []),
        (load_script_module("03_augment_dataset.py"), "03_augment_dataset.py", []),
        (load_script_module("04_convert_coco.py"), "04_convert_coco.py", []),
        (load_script_module("05_convert_yolo.py"), "05_convert_yolo.py", []),
        (load_script_module("06_split_dataset.py"), "06_split_dataset.py", []),
    ):
        monkeypatch.setattr(
            sys, "argv", [script_name, "--config", str(config_path), *extra_args]
        )
        module.main()

    monkeypatch.setattr(sys, "argv", ["08_validate_dataset.py", "--config", str(config_path)])
    validate_dataset_script.main()  # raises via sys.exit(1) if the gate fails

    report = json.loads((tmp_path / "outputs" / "validation_report.json").read_text())
    assert report["is_valid"] is True
    assert report["num_errors"] == 0


def test_source_dr_default_is_unaffected_by_source_flag_plumbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--source` defaults to `"dr"` -- omitting it entirely must behave identically."""
    config_path = write_dataset_config(tmp_path, count=8)

    monkeypatch.setattr(sys, "argv", ["01_generate_synthetic.py", "--config", str(config_path)])
    generate_synthetic.main()
    implicit_manifest = (tmp_path / "data" / "manifest_generated.json").read_text()

    tmp_path_2 = tmp_path / "explicit"
    tmp_path_2.mkdir()
    config_path_2 = write_dataset_config(tmp_path_2, count=8)
    monkeypatch.setattr(
        sys, "argv", ["01_generate_synthetic.py", "--config", str(config_path_2), "--source", "dr"]
    )
    generate_synthetic.main()
    explicit_manifest = (tmp_path_2 / "data" / "manifest_generated.json").read_text()

    # manifest_generated.json only holds relative image paths/sample-index-derived
    # ids -- no absolute tmp_path leaks in, so these should be identical outright.
    assert implicit_manifest == explicit_manifest
