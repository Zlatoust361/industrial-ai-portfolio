"""CLI: sweep prompt phrasings against a labeled validation set, rank by AUC separability.

For every distinct phrase across all styles in `config/prompts.yaml`,
scores every validation image by cosine similarity to that phrase alone,
and measures how well that score separates `defect` from `ok` (ROC AUC,
reported as separability = max(auc, 1-auc) so a phrase that ranks `ok`
higher is just as informative, with inverted polarity).

Runs the whole sweep twice against the *same* loaded weights: once as
loaded in fp32, and once after calling `model.half()` in place to emulate
the Jetson Orin Nano's fp16 runtime -- not by reloading with a different
config, so both passes score the identical checkpoint. Reports whether the
top-K best-separating phrases changed between precisions.

Usage:
    PYTHONPATH=. python scripts/optimize_prompts.py --validation-dir path/to/val \\
        [--config config/model_config.yaml] [--prompts config/prompts.yaml] \\
        [--top-k 3] [--output outputs/week02/optimize_prompts_report.json]

`--validation-dir` must contain `ok/` and `defect/` subfolders of images.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import load_config
from src.data.dataset import LabeledImageFolderDataset
from src.data.prompts import PromptManager
from src.evaluation.prompt_optimization import (
    PromptCandidate,
    PromptEvaluation,
    collect_candidate_phrases,
    diff_top_k,
    evaluate_prompt_candidates,
    labels_to_binary,
    top_k_by_separability,
)
from src.models.clip import CLIPLoader, extract_embedding_tensor

DEFAULT_OUTPUT = "outputs/week02/optimize_prompts_report.json"
DEFAULT_TOP_K = 3


def encode_images_matching_model_dtype(
    model: Any, processor: Any, images: list[Image.Image], device: str
) -> np.ndarray:
    """Encode images via `model.get_image_features`, casting pixel_values to the model's dtype.

    Needed because `model.half()` mutates the model's weights in place
    without touching the processor's (always fp32) tensor output -- a naive
    call would dtype-mismatch after the fp16 emulation step below.
    """
    import torch

    inputs = processor(images=images, return_tensors="pt").to(device)
    model_dtype = next(model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    with torch.no_grad():
        features = extract_embedding_tensor(model.get_image_features(**inputs))
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.float().cpu().numpy()


def encode_texts_matching_model_dtype(
    model: Any, processor: Any, texts: list[str], device: str
) -> np.ndarray:
    """Encode texts via `model.get_text_features` (token ids are dtype-independent)."""
    import torch

    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = extract_embedding_tensor(model.get_text_features(**inputs))
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.float().cpu().numpy()


def run_precision_pass(
    model: Any,
    processor: Any,
    device: str,
    images: list[Image.Image],
    labels: np.ndarray,
    candidates: list[PromptCandidate],
) -> list[PromptEvaluation]:
    """Encode the validation images + every candidate phrase, and evaluate all of them.

    Args:
        model: The (already loaded, possibly `.half()`-ed) CLIP model.
        processor: Its matching `CLIPProcessor`.
        device: Device the model lives on.
        images: Validation images, in the same order as `labels`.
        labels: Binary ground truth per image (see `labels_to_binary`).
        candidates: Candidate phrases to score (see `collect_candidate_phrases`).

    Returns:
        One `PromptEvaluation` per candidate.
    """
    image_vectors = encode_images_matching_model_dtype(model, processor, images, device)
    phrases = [candidate.phrase for candidate in candidates]
    text_vectors = encode_texts_matching_model_dtype(model, processor, phrases, device)
    candidate_vectors = dict(zip(phrases, text_vectors, strict=True))
    return evaluate_prompt_candidates(image_vectors, labels, candidates, candidate_vectors)


def evaluation_to_dict(evaluation: PromptEvaluation) -> dict[str, Any]:
    """JSON-friendly view of a `PromptEvaluation`."""
    origins = evaluation.candidate.origins
    return {
        "phrase": evaluation.candidate.phrase,
        "origins": [{"style": style, "class_label": label} for style, label in origins],
        "auc": evaluation.auc,
        "separability": evaluation.separability,
        "polarity": evaluation.polarity,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `optimize_prompts.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--validation-dir", required=True, help="Folder with ok/ and defect/ image subfolders."
    )
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Sweep every prompt candidate in fp32 and fp16 (via `.half()`), report top-K stability."""
    args = parse_args()

    config = load_config(args.config)
    # fp16 is emulated via model.half() below, on the SAME loaded weights --
    # force fp32 for the initial load so there's exactly one download/load.
    fp32_model_section = replace(config.model, precision="fp32")

    prompt_manager = PromptManager.from_yaml(args.prompts)
    candidates = collect_candidate_phrases(prompt_manager)

    try:
        dataset = LabeledImageFolderDataset(args.validation_dir, class_labels=["ok", "defect"])
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if len(dataset) == 0:
        print(f"Error: no images found under '{args.validation_dir}'.", file=sys.stderr)
        sys.exit(1)

    images = [sample.load() for sample in dataset.samples]
    labels = labels_to_binary([sample.label for sample in dataset.samples])

    loader = CLIPLoader(fp32_model_section, config.memory)
    loader.load()

    print(
        f"Evaluating {len(candidates)} candidate phrase(s) "
        f"against {len(dataset)} validation image(s)..."
    )

    fp32_evaluations = run_precision_pass(
        loader.model, loader.processor, fp32_model_section.device, images, labels, candidates
    )
    fp32_top = top_k_by_separability(fp32_evaluations, args.top_k)

    loader.model.half()
    fp16_evaluations = run_precision_pass(
        loader.model, loader.processor, fp32_model_section.device, images, labels, candidates
    )
    fp16_top = top_k_by_separability(fp16_evaluations, args.top_k)

    diff = diff_top_k(fp32_top, fp16_top)

    report = {
        "metadata": {
            "model_name": config.model.name,
            "validation_dir": args.validation_dir,
            "num_validation_images": len(dataset),
            "num_candidates": len(candidates),
            "top_k": args.top_k,
        },
        "fp32": {
            "top_k": [evaluation_to_dict(e) for e in fp32_top],
            "all": [
                evaluation_to_dict(e)
                for e in sorted(fp32_evaluations, key=lambda e: e.separability, reverse=True)
            ],
        },
        "fp16_via_half": {
            "top_k": [evaluation_to_dict(e) for e in fp16_top],
            "all": [
                evaluation_to_dict(e)
                for e in sorted(fp16_evaluations, key=lambda e: e.separability, reverse=True)
            ],
        },
        "top_k_diff": asdict(diff),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(f"fp32       top-{args.top_k}: {diff.fp32_phrases}")
    print(f"fp16(.half) top-{args.top_k}: {diff.fp16_phrases}")
    changed = "yes" if not diff.same_set else "no"
    print(f"Top-{args.top_k} changed between precisions: {changed} (same order: {diff.same_order})")
    print(f"Report written to '{output_path}'.")


if __name__ == "__main__":
    main()
