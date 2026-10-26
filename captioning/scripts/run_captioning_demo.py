#!/usr/bin/env python3
"""Week 15's whole non-interactive deliverable, one command.

Generates a deterministic SYNTHETIC set of week-04-pipeline-shaped
verdicts (some "ok", some "defect"), captions the confirmed defects
through `captioning.model.CaptioningLoader` (a real small instruction-
tuned VLM, not a mock), and scores each caption against its image with
`captioning.evaluation.clip_caption_similarity` (CLIP text-image
similarity -- reference-free, not human review, see
`captioning/README.md`'s Honesty note for exactly what that does and
doesn't prove).

Both model calls (`caption_fn`/`clip_inference_factory` parameters of
`run_captioning_demo`) are injectable, defaulting to the real ones here
in `main()` -- tests inject fakes instead, the same separation
`peft/scripts/02_train_adapter.py`'s tests use for the real
`CLIPLoader.load()` network call.

Usage:
    PYTHONPATH=. python3 scripts/run_captioning_demo.py [--num-images 20] \\
        [--defect-rate 0.3] [--seed 42] [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_CAPTIONING_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CAPTIONING_ROOT.parent))

from PIL import Image  # noqa: E402

from captioning.evaluation import clip_caption_similarity  # noqa: E402
from captioning.integration import augment_verdicts_with_captions  # noqa: E402
from captioning.model import CaptioningLoader, generate_caption_text  # noqa: E402
from captioning.prompting import CaptionFn  # noqa: E402
from captioning.utils.config import CaptioningConfig, load_config  # noqa: E402

DEFAULT_CONFIG_PATH = "config/config.yaml"
_SYNTHETIC_LABELS = ("ok", "defect")


def generate_synthetic_verdicts(
    num_images: int, defect_rate: float, seed: int
) -> tuple[list[dict[str, object]], dict[str, Image.Image]]:
    """Build a deterministic set of week-04-pipeline-shaped verdicts + their images.

    Why synthetic, not a real inspection run: this week's actual new code
    (captioning/model.py, prompting.py, integration.py, evaluation.py)
    operates on week 04's verdict/image SHAPE, not on a real detector --
    feeding it a real pipeline run would ALSO require a real detector
    with real weights (same network boundary `peft/README.md`'s Honesty
    note already documents) just to produce the verdicts this demo
    already needs as INPUT. A synthetic set with a known "ok"/"defect"
    mix is a real, non-trivial test of the trigger/captioning/evaluation
    wiring on its own terms -- it just can't speak to real defect-
    description quality on a real part (see `captioning/README.md`'s
    Honesty note).

    Args:
        num_images: How many synthetic verdicts to generate.
        defect_rate: Fraction (roughly) labeled `"defect"`.
        seed: Deterministic RNG seed.

    Returns:
        `(verdicts, images)` -- `verdicts` is a list of week-04-shaped
        `{"image", "defect_score", "verdict"}` dicts; `images` maps each
        verdict's `"image"` key to an in-memory synthetic `PIL.Image`
        (distinct solid colors, not meaningful content -- same
        "zero-setup" synthetic-fixture style `peft/tests/conftest.py::
        make_tiny_dataset` already uses).
    """
    rng = random.Random(seed)
    verdicts: list[dict[str, object]] = []
    images: dict[str, Image.Image] = {}

    for i in range(num_images):
        is_defect = rng.random() < defect_rate
        verdict = "defect" if is_defect else "ok"
        score = rng.uniform(0.6, 0.95) if is_defect else rng.uniform(0.0, 0.4)
        image_path = f"synthetic/image_{i:03d}.png"
        color = (i * 17 % 256, i * 29 % 256, i * 53 % 256)
        images[image_path] = Image.new("RGB", (32, 32), color=color)
        verdicts.append({"image": image_path, "defect_score": score, "verdict": verdict})

    return verdicts, images


def run_captioning_demo(
    config: CaptioningConfig,
    num_images: int,
    defect_rate: float,
    seed: int,
    caption_fn: CaptionFn | None = None,
    clip_inference_factory: object | None = None,
) -> dict[str, object]:
    """Run the full trigger -> caption -> evaluate pipeline over a synthetic verdict set.

    Args:
        config: Loaded `CaptioningConfig`.
        num_images: Passed to `generate_synthetic_verdicts`.
        defect_rate: Passed to `generate_synthetic_verdicts`.
        seed: Passed to `generate_synthetic_verdicts`.
        caption_fn: Injected caption call (see `captioning.prompting`).
            `None` builds a REAL `CaptioningLoader` from `config.model`/
            `config.memory` and binds `captioning.model.generate_caption_text`
            to it -- this is the one real network-reaching call path.
        clip_inference_factory: Zero-argument callable returning an
            already-loaded CLIP-text/image encoder (see
            `captioning.evaluation`). `None` builds a real
            `src.models.clip.CLIPLoader`/`CLIPInference` -- the OTHER
            real network-reaching call path.

    Returns:
        A JSON-serializable report: per-image caption/score entries, and
        aggregate stats (`num_captioned`, `num_skipped`, `mean_similarity`).
    """
    verdicts, images = generate_synthetic_verdicts(num_images, defect_rate, seed)

    if caption_fn is None:
        loader = CaptioningLoader(config.model, config.memory)
        caption_fn = lambda image, prompt: generate_caption_text(  # noqa: E731
            loader, image, prompt, max_new_tokens=config.prompting.max_new_tokens
        )

    augmented = augment_verdicts_with_captions(
        verdicts=verdicts,
        load_image=lambda path: images[path],
        allowed_phrases=config.prompting.allowed_phrases,
        model_name=config.model.name,
        caption_fn=caption_fn,
        trigger=config.trigger.mode,
    )

    if clip_inference_factory is None:
        from src.config import MemorySection, ModelSection
        from src.models.clip import CLIPInference, CLIPLoader

        clip_model = ModelSection(
            name="openai/clip-vit-base-patch32",
            type="clip",
            precision="fp32",
            device="cpu",
            seed=config.model.seed,
            cache_dir=config.model.cache_dir,
        )
        clip_memory = MemorySection(memory_budget_mb=2048, device_total_mb=8192)
        clip_inference_factory = lambda: CLIPInference(CLIPLoader(clip_model, clip_memory))  # noqa: E731

    clip_inference = clip_inference_factory()

    similarities: list[float] = []
    for entry in augmented:
        if entry.get("caption") is None:
            continue
        image = images[str(entry["image"])]
        similarity = clip_caption_similarity(clip_inference, image, str(entry["caption"]))
        entry["clip_similarity"] = similarity
        similarities.append(similarity)

    num_captioned = len(similarities)
    num_skipped = len(augmented) - num_captioned
    mean_similarity = sum(similarities) / num_captioned if num_captioned > 0 else None

    return {
        "num_images": num_images,
        "defect_rate": defect_rate,
        "seed": seed,
        "trigger": config.trigger.mode,
        "num_captioned": num_captioned,
        "num_skipped": num_skipped,
        "mean_clip_similarity": mean_similarity,
        "verdicts": augmented,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_captioning_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--defect-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the demo, write the augmented report + evaluation summary, print the headline numbers."""
    args = parse_args()
    config = load_config(args.config)

    report = run_captioning_demo(
        config, num_images=args.num_images, defect_rate=args.defect_rate, seed=args.seed
    )

    config.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.paths.augmented_report_path.write_text(
        json.dumps(report["verdicts"], indent=2), encoding="utf-8"
    )
    summary = {k: v for k, v in report.items() if k != "verdicts"}
    config.paths.evaluation_report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"{report['num_images']} synthetic verdicts (defect_rate={report['defect_rate']}, "
        f"trigger={report['trigger']}): {report['num_captioned']} captioned, "
        f"{report['num_skipped']} skipped."
    )
    if report["mean_clip_similarity"] is not None:
        print(
            "Mean CLIP text-image similarity (reference-free): "
            f"{report['mean_clip_similarity']:.3f}"
        )
    print(f"Augmented verdicts -> {config.paths.augmented_report_path}")
    print(f"Evaluation summary -> {config.paths.evaluation_report_path}")


if __name__ == "__main__":
    main()
