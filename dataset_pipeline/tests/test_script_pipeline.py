"""Script-level (CLI/`main()`) tests for the full 01-08 stage chain (week 06).

Runs the real chain end to end on a tiny synthetic config (12 generated
samples) -- everything here is real generation/stratification/augmentation/
conversion/split/export/validation, no mocking, since none of it needs
`torch`. Each stage's own `test_*.py` file covers its pure logic in
isolation; this file covers the actual scripts' `main()`/CLI wiring and
the "stage N fails clearly if stage N-1 hasn't run yet" contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_dataset_config

generate_synthetic = load_script_module("01_generate_synthetic.py")
stratify_dataset = load_script_module("02_stratify_dataset.py")
augment_dataset = load_script_module("03_augment_dataset.py")
convert_coco = load_script_module("04_convert_coco.py")
convert_yolo = load_script_module("05_convert_yolo.py")
split_dataset = load_script_module("06_split_dataset.py")
export_jetson_val_subset = load_script_module("07_export_jetson_val_subset.py")
validate_dataset_script = load_script_module("08_validate_dataset.py")


def test_full_chain_runs_and_produces_a_jetson_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # count=120 (not the smaller counts other tests use) so that at least a
    # few of the ~36 possible strata end up with >1 sample and the split
    # actually allocates something to val -- a single-sample stratum always
    # goes to train (see test_split.py's largest-remainder test), so a
    # too-small count here would make every val_samples empty and this
    # "happy path" test would never reach stage 7's success path at all.
    # jetson_subset thresholds are still sized down to match this dataset --
    # a real run uses generation.count in the thousands (see dataset_pipeline/README.md).
    config_path = write_dataset_config(
        tmp_path, count=120, jetson_min_count=1, jetson_target_count=2, jetson_max_count=5
    )

    for module, script_name in (
        (generate_synthetic, "01_generate_synthetic.py"),
        (stratify_dataset, "02_stratify_dataset.py"),
        (augment_dataset, "03_augment_dataset.py"),
        (convert_coco, "04_convert_coco.py"),
        (convert_yolo, "05_convert_yolo.py"),
        (split_dataset, "06_split_dataset.py"),
        (export_jetson_val_subset, "07_export_jetson_val_subset.py"),
        (validate_dataset_script, "08_validate_dataset.py"),
    ):
        monkeypatch.setattr(sys, "argv", [script_name, "--config", str(config_path)])
        module.main()

    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"

    assert (data_dir / "manifest_generated.json").is_file()
    assert (data_dir / "manifest_stratified.json").is_file()
    assert (data_dir / "manifest_augmented.json").is_file()
    assert (data_dir / "manifest_split.json").is_file()

    distribution_report = json.loads(
        (outputs_dir / "metrics" / "stratify_distribution.json").read_text()
    )
    assert distribution_report["total_samples"] == 120
    assert sum(distribution_report["label_distribution"].values()) == 120
    assert set(distribution_report["brightness_distribution"]) == {"low", "mid", "high"}
    assert "brightness" in distribution_report["balance_ratios"]

    manifest = json.loads((data_dir / "manifest_split.json").read_text())
    assert len(manifest["samples"]) > 120  # 120 base + stage 3's augmented variants
    assert any(sample["source_sample_id"] is not None for sample in manifest["samples"])
    assert all(sample["split"] in ("train", "val", "test") for sample in manifest["samples"])

    assert (outputs_dir / "coco" / "annotations.json").is_file()
    assert (outputs_dir / "coco" / "annotations_train.json").is_file()
    for split in ("train", "val", "test"):
        assert (outputs_dir / "yolo" / split / "classes.txt").is_file()

    subset_dir = outputs_dir / "jetson_val_subset"
    assert (subset_dir / "classes.txt").is_file()
    assert list((subset_dir / "images").iterdir())
    assert list((subset_dir / "labels").iterdir())

    validation_report = json.loads((outputs_dir / "validation_report.json").read_text())
    assert validation_report["is_valid"] is True
    assert validation_report["num_errors"] == 0


def test_stage_02_warns_on_stderr_for_an_imbalanced_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A handful of samples spread across up to 36 strata reliably leaves at
    # least one marginal axis skewed below the default balance_warn_ratio (0.3).
    config_path = write_dataset_config(tmp_path, count=6)
    monkeypatch.setattr(sys, "argv", ["01_generate_synthetic.py", "--config", str(config_path)])
    generate_synthetic.main()

    monkeypatch.setattr(sys, "argv", ["02_stratify_dataset.py", "--config", str(config_path)])
    stratify_dataset.main()

    stderr = capsys.readouterr().err
    assert "Warning:" in stderr
    assert "imbalanced" in stderr


def test_stage_08_exits_nonzero_on_corrupted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_dataset_config(tmp_path, count=6)
    for module, script_name in (
        (generate_synthetic, "01_generate_synthetic.py"),
        (stratify_dataset, "02_stratify_dataset.py"),
        (augment_dataset, "03_augment_dataset.py"),
        (convert_coco, "04_convert_coco.py"),
        (convert_yolo, "05_convert_yolo.py"),
        (split_dataset, "06_split_dataset.py"),
    ):
        monkeypatch.setattr(sys, "argv", [script_name, "--config", str(config_path)])
        module.main()

    # Corrupt the real stage-6 output: force a "defect" sample to carry an
    # empty mask, the exact contract violation 08_validate_dataset.py exists to catch.
    manifest_path = tmp_path / "data" / "manifest_split.json"
    manifest = json.loads(manifest_path.read_text())
    corrupted = False
    for sample in manifest["samples"]:
        if sample["label"] == "defect" and sample["defect_boxes"]:
            sample["defect_boxes"] = []
            corrupted = True
            break
    assert corrupted, "test setup: expected at least one defect sample to corrupt"
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(sys, "argv", ["08_validate_dataset.py", "--config", str(config_path)])
    with pytest.raises(SystemExit) as exc_info:
        validate_dataset_script.main()

    assert exc_info.value.code == 1
    assert "empty_defect_mask" in capsys.readouterr().out


def test_stage_02_missing_stage_01_output_exits_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_dataset_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["02_stratify_dataset.py", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc_info:
        stratify_dataset.main()

    assert exc_info.value.code == 1
    assert "01_generate_synthetic.py" in capsys.readouterr().err


def test_stage_06_missing_stage_05_output_exits_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_dataset_config(tmp_path, count=6)
    for module, script_name in (
        (generate_synthetic, "01_generate_synthetic.py"),
        (stratify_dataset, "02_stratify_dataset.py"),
        (augment_dataset, "03_augment_dataset.py"),
        (convert_coco, "04_convert_coco.py"),
        # deliberately skip 05_convert_yolo.py
    ):
        monkeypatch.setattr(sys, "argv", [script_name, "--config", str(config_path)])
        module.main()

    monkeypatch.setattr(sys, "argv", ["06_split_dataset.py", "--config", str(config_path)])
    with pytest.raises(SystemExit) as exc_info:
        split_dataset.main()

    assert exc_info.value.code == 1
    assert "05_convert_yolo.py" in capsys.readouterr().err


def test_stage_07_too_few_val_samples_exits_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # default jetson_subset.min_count (200) vastly exceeds this tiny run's val split.
    config_path = write_dataset_config(tmp_path, count=6, jetson_min_count=200)
    for module, script_name in (
        (generate_synthetic, "01_generate_synthetic.py"),
        (stratify_dataset, "02_stratify_dataset.py"),
        (augment_dataset, "03_augment_dataset.py"),
        (convert_coco, "04_convert_coco.py"),
        (convert_yolo, "05_convert_yolo.py"),
        (split_dataset, "06_split_dataset.py"),
    ):
        monkeypatch.setattr(sys, "argv", [script_name, "--config", str(config_path)])
        module.main()

    monkeypatch.setattr(
        sys, "argv", ["07_export_jetson_val_subset.py", "--config", str(config_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        export_jetson_val_subset.main()

    assert exc_info.value.code == 1
    assert "generation.count" in capsys.readouterr().err
