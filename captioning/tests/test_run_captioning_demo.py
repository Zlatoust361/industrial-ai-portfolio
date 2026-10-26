"""Tests for `scripts/run_captioning_demo.py` -- the week 15 one-command deliverable."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module
from PIL import Image

from captioning.triggers import TRIGGER_ON_DEFECT_CONFIRMED
from captioning.utils.config import (
    CaptioningConfig,
    MemorySection,
    ModelSection,
    PathsSection,
    PromptingSection,
    TriggerSection,
)

run_captioning_demo = load_script_module("run_captioning_demo.py")


def _fake_config(tmp_path: Path, trigger: str = TRIGGER_ON_DEFECT_CONFIRMED) -> CaptioningConfig:
    return CaptioningConfig(
        model=ModelSection(
            name="fake-model",
            type="vlm-captioning",
            precision="fp32",
            device="cpu",
            seed=42,
            cache_dir=tmp_path / ".cache",
        ),
        memory=MemorySection(memory_budget_mb=2048, device_total_mb=8192),
        prompting=PromptingSection(allowed_phrases=("scratch",), max_new_tokens=40),
        trigger=TriggerSection(mode=trigger),
        paths=PathsSection(
            outputs_dir=tmp_path,
            augmented_report_path=tmp_path / "augmented_verdicts.json",
            evaluation_report_path=tmp_path / "evaluation_report.json",
        ),
    )


def _fake_caption_fn(image: Image.Image, prompt: str) -> str:
    return "scratch along the edge"


class _FakeClipInference:
    def encode_image(self, images: list[Image.Image]) -> object:
        import numpy as np

        class _Result:
            vectors = np.stack([[1.0, 0.0]] * len(images))

        return _Result()

    def encode_text(self, texts: list[str]) -> object:
        import numpy as np

        class _Result:
            vectors = np.stack([[1.0, 0.0]] * len(texts))

        return _Result()


def test_generate_synthetic_verdicts_is_deterministic_for_a_fixed_seed() -> None:
    verdicts_a, images_a = run_captioning_demo.generate_synthetic_verdicts(
        num_images=10, defect_rate=0.4, seed=7
    )
    verdicts_b, _ = run_captioning_demo.generate_synthetic_verdicts(
        num_images=10, defect_rate=0.4, seed=7
    )
    assert verdicts_a == verdicts_b
    assert len(images_a) == 10


def test_generate_synthetic_verdicts_different_seeds_diverge() -> None:
    verdicts_a, _ = run_captioning_demo.generate_synthetic_verdicts(
        num_images=20, defect_rate=0.5, seed=1
    )
    verdicts_b, _ = run_captioning_demo.generate_synthetic_verdicts(
        num_images=20, defect_rate=0.5, seed=2
    )
    assert verdicts_a != verdicts_b


def test_run_captioning_demo_with_injected_fakes_captions_only_defects(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    report = run_captioning_demo.run_captioning_demo(
        config,
        num_images=20,
        defect_rate=0.3,
        seed=42,
        caption_fn=_fake_caption_fn,
        clip_inference_factory=lambda: _FakeClipInference(),
    )

    assert report["num_captioned"] + report["num_skipped"] == 20
    assert report["num_captioned"] > 0  # defect_rate=0.3 over 20 images -> at least some
    assert report["mean_clip_similarity"] == 1.0  # identical fake vectors -> similarity 1
    for entry in report["verdicts"]:
        if entry["verdict"] == "defect":
            assert entry["caption"] == "scratch along the edge"
        else:
            assert entry["caption"] is None


def test_run_captioning_demo_zero_defects_has_no_similarity_score(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    report = run_captioning_demo.run_captioning_demo(
        config,
        num_images=5,
        defect_rate=0.0,
        seed=42,
        caption_fn=_fake_caption_fn,
        clip_inference_factory=lambda: _FakeClipInference(),
    )
    assert report["num_captioned"] == 0
    assert report["mean_clip_similarity"] is None


def test_main_writes_reports(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model:
  name: fake-model
  type: vlm-captioning
  precision: fp32
  device: cpu
  seed: 42
  cache_dir: {tmp_path / ".cache"}
memory:
  memory_budget_mb: 2048
  device_total_mb: 8192
prompting:
  allowed_phrases: ["scratch"]
  max_new_tokens: 40
trigger:
  mode: on_defect_confirmed
paths:
  outputs_dir: {tmp_path}
  augmented_report_path: {tmp_path / "augmented_verdicts.json"}
  evaluation_report_path: {tmp_path / "evaluation_report.json"}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_captioning_demo,
        "parse_args",
        lambda: run_captioning_demo.argparse.Namespace(
            config=str(config_path), num_images=5, defect_rate=1.0, seed=1
        ),
    )
    monkeypatch.setattr(
        run_captioning_demo,
        "run_captioning_demo",
        lambda config, num_images, defect_rate, seed: {
            "num_images": num_images,
            "defect_rate": defect_rate,
            "seed": seed,
            "trigger": config.trigger.mode,
            "num_captioned": num_images,
            "num_skipped": 0,
            "mean_clip_similarity": 0.5,
            "verdicts": [{"image": "a.png", "verdict": "defect", "caption": "x"}],
        },
    )

    run_captioning_demo.main()

    augmented = json.loads((tmp_path / "augmented_verdicts.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert augmented == [{"image": "a.png", "verdict": "defect", "caption": "x"}]
    assert summary["num_captioned"] == 5
    assert "verdicts" not in summary
