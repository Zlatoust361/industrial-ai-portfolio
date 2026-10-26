"""CLI: inpaint masked defects, quality-assess, keep the best fraction (week 05, day 3).

Consumes `scripts/generate_defect_masks.py`'s (week 05 day 2) output
manifest -- one mask per entry, each paired with a base image either from
`--image-dir` (real images, matched by filename stem) or, if omitted, a
deterministic synthetic textured placeholder
(`src.synthetic.diffusion_pipeline.synthesize_placeholder_surface`) sized
to match. Runs the configured inpainting backend
(`config/diffusion.yaml`'s `backend:`, override with `--backend`) on
each, scores every result with `src.synthetic.quality_assessment`
(PSNR/SSIM outside the mask, blur/contrast inside it), and keeps the top
`quality_gate.keep_fraction` (default 75%) by composite score. Selected
and rejected images are both written (for inspection), alongside a copy
of each result's base image under `originals/` (week 05 day 4's
comparison report needs the exact before image, not just the after --
see `_Payload`), plus one `manifest.json` recording every result's
metrics/score/status/image paths.

Usage:
    PYTHONPATH=. python scripts/run_diffusion_pipeline.py \\
        --mask-manifest outputs/synthetic_defects/masks/manifest.json \\
        [--image-dir path/to/base_images] \\
        [--diffusion-config config/diffusion.yaml] \\
        [--output-dir outputs/synthetic_defects/inpainted] \\
        [--backend cv2|diffusers|stability_api] [--seed 42]

The default backend (`cv2`, see `config/diffusion.yaml`) is the only one
this sandbox/CI can actually run -- see `README.md`'s "Week 05" section
for why, and for how to switch to `diffusers`/`stability_api` on a real
dev GPU / with a real API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocessing import load_image
from src.synthetic import quality_assessment as qa
from src.synthetic.diffusion_config import load_diffusion_config
from src.synthetic.diffusion_pipeline import build_backend, synthesize_placeholder_surface

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_diffusion_pipeline.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mask-manifest", default="outputs/synthetic_defects/masks/manifest.json")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Folder of real base images, matched to masks by filename stem. If omitted, "
        "a synthetic textured placeholder is generated per mask instead.",
    )
    parser.add_argument("--diffusion-config", default="config/diffusion.yaml")
    parser.add_argument("--output-dir", default="outputs/synthetic_defects/inpainted")
    parser.add_argument(
        "--backend", default=None, choices=("cv2", "diffusers", "stability_api"),
        help="Override diffusion.yaml's backend: for this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for synthetic placeholder surfaces (see --image-dir).",
    )
    return parser.parse_args()


def _find_base_image(image_dir: Path, stem: str) -> Path | None:
    """First `{image_dir}/{stem}{ext}` that exists, trying `IMAGE_EXTENSIONS` in order."""
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


@dataclass
class _Payload:
    """Carried through `qa.composite_scores`' generic `payload` field.

    Both images (not just the inpainted result) so day 4's comparison
    report can render a real before/after -- without this, it would have
    no way to recover the exact base image a given result was inpainted
    from (`base_image` is never otherwise persisted to disk).
    """

    result_image: Image.Image
    base_image: Image.Image
    mask_path: str


def main() -> None:
    """Inpaint every masked base image, quality-assess, select top fraction, write outputs."""
    args = parse_args()

    mask_manifest_path = Path(args.mask_manifest)
    if not mask_manifest_path.exists():
        print(
            f"Error: mask manifest not found: {mask_manifest_path} -- run "
            "scripts/generate_defect_masks.py first (week 05 day 2).",
            file=sys.stderr,
        )
        sys.exit(1)
    mask_manifest = json.loads(mask_manifest_path.read_text())

    diffusion_config = load_diffusion_config(args.diffusion_config)
    if args.backend is not None:
        diffusion_config = replace(diffusion_config, backend=args.backend)

    try:
        backend = build_backend(diffusion_config)
    except (ImportError, RuntimeError) as exc:
        print(
            f"Error: could not build backend {diffusion_config.backend!r}: {exc}", file=sys.stderr
        )
        sys.exit(1)

    image_dir = Path(args.image_dir) if args.image_dir is not None else None

    identifiers: list[str] = []
    metrics_list: list[qa.QualityMetrics] = []
    payloads: list[_Payload] = []
    skipped = 0

    for i, entry in enumerate(mask_manifest["masks"]):
        mask_path = Path(entry["mask"])
        stem = mask_path.stem
        width, height = entry["image_size"]
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        if image_dir is not None:
            base_path = _find_base_image(image_dir, stem)
            if base_path is None:
                print(
                    f"Warning: no base image found for mask '{stem}' under {image_dir} -- "
                    "skipping.",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            base_image = load_image(base_path)
        else:
            base_image = synthesize_placeholder_surface((width, height), seed=args.seed + i)

        result_image = backend.inpaint(
            base_image, mask, diffusion_config.prompt.positive, diffusion_config.prompt.negative
        )

        original_rgb = np.array(base_image.convert("RGB"), dtype=np.uint8)
        inpainted_rgb = np.array(result_image.convert("RGB"), dtype=np.uint8)
        metrics = qa.assess(
            original_rgb, inpainted_rgb, mask, ssim_window=diffusion_config.quality_gate.ssim_window
        )

        identifiers.append(stem)
        metrics_list.append(metrics)
        payloads.append(
            _Payload(result_image=result_image, base_image=base_image, mask_path=str(mask_path))
        )

    if not metrics_list:
        print("Error: no inpainting results produced -- nothing to select.", file=sys.stderr)
        sys.exit(1)

    scored = qa.composite_scores(
        identifiers, metrics_list, diffusion_config.quality_gate.weights, payloads=payloads
    )
    kept, rejected = qa.select_top_fraction(scored, diffusion_config.quality_gate.keep_fraction)

    output_dir = Path(args.output_dir)
    selected_dir = output_dir / "selected"
    rejected_dir = output_dir / "rejected"
    originals_dir = output_dir / "originals"
    selected_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    originals_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for status, group, group_dir in (
        ("selected", kept, selected_dir),
        ("rejected", rejected, rejected_dir),
    ):
        for result in group:
            payload: _Payload = result.payload
            image_path = group_dir / f"{result.identifier}.png"
            original_path = originals_dir / f"{result.identifier}.png"
            payload.result_image.save(image_path)
            payload.base_image.save(original_path)
            manifest_entries.append(
                {
                    "identifier": result.identifier,
                    "status": status,
                    "composite_score": result.composite_score,
                    "metrics": result.metrics.to_dict(),
                    "image": str(image_path),
                    "original_image": str(original_path),
                    "mask": payload.mask_path,
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": backend.name,
                "diffusion_config": str(args.diffusion_config),
                "mask_manifest": str(mask_manifest_path),
                "keep_fraction": diffusion_config.quality_gate.keep_fraction,
                "weights": diffusion_config.quality_gate.weights,
                "results": manifest_entries,
            },
            indent=2,
        )
    )

    print(f"Backend: {backend.name}")
    print(
        f"Processed {len(scored)} mask(s) ({skipped} skipped, no base image found): "
        f"{len(kept)} selected, {len(rejected)} rejected "
        f"(keep_fraction={diffusion_config.quality_gate.keep_fraction})."
    )
    print(f"Manifest written to '{manifest_path}'.")


if __name__ == "__main__":
    main()
