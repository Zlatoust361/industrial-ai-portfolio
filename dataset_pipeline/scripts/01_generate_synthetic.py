#!/usr/bin/env python3
"""Stage 1/7: generate synthetic ok/defect scenes (week 06).

Two selectable, real generation sources (`--source dr|diffusion`,
default `dr`) -- see `course/TECH-DEBT.md` C.1 / `course/DECISIONS.md`
ADR-XXX for why this pipeline previously only reconciled with one of
them:

- `--source dr` (default, byte-for-byte the original stage-1 behavior):
  reuses week 07's DR scene renderer (`src.domain_randomization.scene.
  render_scene`) for the base scene and week 05's parametric defect-mask
  generator (`src.synthetic.mask_generation.generate_defect_mask`) for
  `"defect"`-labeled samples, rather than reimplementing scene
  compositing a second time. A `"defect"` sample here is a DR-rendered
  scene with a generated defect shape blended visibly onto it
  (`overlay_defect_mask`), not a photorealistic inpaint. Every sample is
  independently randomized (this stage isn't producing week 07's
  `no_dr`/`with_dr` comparison branches -- just one diverse dataset).
- `--source diffusion`: reuses week 05's *full* days 2-3 pipeline instead
  -- `generate_defect_mask` for the paint mask, then
  `src.synthetic.diffusion_pipeline.build_backend`/`.inpaint` for a real
  (non-mocked) inpainting pass, then `src.synthetic.quality_assessment`'s
  composite-score top-fraction gate to drop low-quality inpaints, exactly
  the same code `scripts/run_diffusion_pipeline.py` already runs. Uses
  `config/diffusion.yaml`'s default `cv2` backend (OpenCV classical
  inpainting) -- no `torch`/GPU/API key/network needed, same
  zero-setup-demo default as `make defect-pipeline`.

**Decision criterion (this is the actual resolution of C.1, not a
preference)**: these are two real, valid strategies with a genuine
tradeoff, not one obsoleting the other.
  - Pick **`dr`** when geometry/pose diversity (viewpoint, lighting,
    background) and automatically-correct-by-construction GT matter most
    -- e.g. training a detector that needs precise object/defect
    localization. `object_bbox`/`defect_boxes` come directly from the
    3D-scene compositor and mask generator, so they're exactly right by
    construction, with no verification step needed.
  - Pick **`diffusion`** when photorealistic surface/texture fidelity on
    a real (or real-textured) background image matters most, and an
    approximate/manually-spot-checked mask is acceptable -- e.g.
    training a classifier/anomaly detector on how a real material
    surface actually looks with a defect, not a flat-color overlay.
  This is a reasoned engineering judgment, not an experimentally
  validated one -- no comparative study (e.g. downstream detector mAP
  trained on one source vs. the other) has been run in this repo to
  prove either source is objectively better for any specific metric.

**Schema honesty note (`--source diffusion`'s real, load-bearing
differences from `dr` -- not papered over)**:
  - `object_bbox` is the *whole frame* `(0, 0, width, height)`, not a
    localized object placement -- diffusion inpainting here has no
    separate "object cutout on a background" concept the way
    `render_scene` does; the entire image *is* the imaged surface. Any
    consumer that expects a tight object-localization box from
    `object_bbox` gets a trivial full-frame box for these samples.
  - `defect_boxes` come from the *input paint mask*'s drawn instances
    (`generate_defect_mask`, same function `dr` uses) -- exactly correct
    for "this is the region we told the inpainter to alter", but, unlike
    `dr`'s overlay (which forcibly makes the masked region visibly
    different), this is NOT a verified post-hoc segmentation of what the
    backend actually painted. A backend could paint a result that blends
    in well enough to not look like a visible defect at all inside that
    same box -- `quality_assessment`'s gate filters the worst offenders
    (low SSIM outside mask, low blur/contrast inside it) but doesn't
    re-segment or guarantee visible defect content, the way `dr`'s
    forced color-blend does by construction.
  - `dr_params` is NOT real per-sample domain-randomization output for
    `diffusion` samples (there is no DR renderer in this path) --
    `light.brightness`/`background_color` are approximated from the
    actual generated image's pixels (mean intensity / border-pixel mean)
    purely so stage 2's stratification code keeps working, and
    `camera.rotation_deg` is hardcoded `0.0` since a 2D inpainted surface
    has no camera-angle concept at all. Real consequence: stage 2's
    "angle" stratification axis is degenerate (always `"center"`) for a
    diffusion-sourced dataset, and its brightness/background diversity
    is far lower than `dr`'s (derived from one synthetic placeholder-
    surface generator, not real per-sample lighting variation) --
    `dr_params` also carries an explicit `"source": "diffusion"` marker
    so this is traceable in the manifest itself, not silently implied.
  - The quality gate can make `--source diffusion` emit FEWER than
    `generation.count` samples: `"ok"` samples are never gated (nothing
    was inpainted to score), but a `"defect"` candidate whose inpaint is
    rejected by `select_top_fraction` is dropped entirely, not retried
    or replaced -- unlike `dr`, where every requested sample is always
    emitted.

Input: none (generates from `generation.dr_parameters_path`/
`generation.defects_config_path`, week 05/07's own config files; `--source
diffusion` additionally reads `generation.diffusion_config_path`, week
05's `diffusion.yaml`).
Output: `<data_dir>/images/*.jpg`, `<data_dir>/manifest_generated.json`
-- same manifest schema (`utils.manifest.Sample`) for both sources, so
every downstream stage (2-8) runs unmodified regardless of `--source`.

Usage:
    python3 scripts/01_generate_synthetic.py [--config config/dataset_config.yaml]
    python3 scripts/01_generate_synthetic.py --source diffusion \\
        [--backend cv2|diffusers|stability_api]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

_DATASET_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DATASET_PIPELINE_ROOT))
# This stage reuses week 05/07's generators from the repo root, one level
# up from dataset_pipeline/ -- see this module's docstring.
sys.path.insert(0, str(_DATASET_PIPELINE_ROOT.parent))

from utils.config import DatasetPipelineConfig, GenerationConfig, load_config  # noqa: E402
from utils.manifest import Bbox, Sample, save_manifest  # noqa: E402

from src.domain_randomization.config import DRParameters, load_dr_parameters  # noqa: E402
from src.domain_randomization.scene import render_scene, synthesize_placeholder_object  # noqa: E402
from src.synthetic import quality_assessment as qa  # noqa: E402
from src.synthetic.config import DefectsConfig, load_defects_config  # noqa: E402
from src.synthetic.diffusion_config import DiffusionConfig, load_diffusion_config  # noqa: E402
from src.synthetic.diffusion_pipeline import (  # noqa: E402
    InpaintingBackend,
    build_backend,
    synthesize_placeholder_surface,
)
from src.synthetic.mask_generation import generate_defect_mask  # noqa: E402


def overlay_defect_mask(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> Image.Image:
    """Blend `color` onto `image` wherever `mask` is set, alpha-weighted.

    Args:
        image: The base RGB scene.
        mask: `(height, width)` `uint8` array, `0`/`255` (see
            `src.synthetic.mask_generation.generate_defect_mask`).
        color: `(r, g, b)` blended in where `mask` is nonzero.
        alpha: Blend strength, `[0, 1]` -- `0` leaves `image` untouched, `1`
            replaces masked pixels with `color` outright.

    Returns:
        A new `RGB` image, same size as `image`.
    """
    base = np.asarray(image, dtype=np.float32)
    overlay = np.array(color, dtype=np.float32)
    mask_3d = (mask > 0)[:, :, None]
    blended = np.where(mask_3d, base * (1 - alpha) + overlay * alpha, base)
    return Image.fromarray(blended.astype(np.uint8))


def generate_sample(
    sample_index: int,
    object_pool: list[Image.Image],
    dr_parameters: DRParameters,
    defects_config: DefectsConfig,
    generation_config: GenerationConfig,
    rng: np.random.Generator,
) -> tuple[Image.Image, Sample]:
    """Render one sample: a DR scene, with a blended-in defect for `"defect"`-labeled ones.

    Args:
        sample_index: Used to build `sample_id` and to cycle through `object_pool`.
        object_pool: Object cutouts to place (cycled by index).
        dr_parameters: Parsed `parameters.yaml`.
        defects_config: Parsed `defects.yaml`.
        generation_config: `generation:` section of `dataset_config.yaml`.
        rng: Shared source of randomness (label, DR sampling, defect mask, blend).

    Returns:
        `(image, sample)`. `sample.image_path` names where the caller
        should save `image` under `<data_dir>` -- this function only
        renders, it doesn't write to disk.
    """
    sample_id = f"sample_{sample_index:06d}"
    object_rgba = object_pool[sample_index % len(object_pool)]
    render_result = render_scene(object_rgba, dr_parameters, rng)

    is_defect = rng.random() < generation_config.defect_fraction
    image = render_result.image
    defect_boxes: list[Bbox] = []
    if is_defect:
        mask, instances = generate_defect_mask(dr_parameters.canvas.size, defects_config, rng)
        image = overlay_defect_mask(
            image,
            mask,
            generation_config.defect_overlay_color,
            generation_config.defect_overlay_alpha,
        )
        defect_boxes = [instance.bbox for instance in instances]

    width, height = image.size
    sample = Sample(
        sample_id=sample_id,
        image_path=f"images/{sample_id}.jpg",
        width=width,
        height=height,
        label="defect" if is_defect else "ok",
        object_bbox=render_result.bbox,
        defect_boxes=tuple(defect_boxes),
        dr_params=render_result.params.to_dict(),
    )
    return image, sample


def approximate_diffusion_dr_params(image: Image.Image) -> dict[str, object]:
    """Best-effort, honestly-approximate `dr_params` for a `--source diffusion` sample.

    There is no real domain-randomization renderer in the diffusion path
    -- this exists purely so stage 2 (`02_stratify_dataset.py`), which
    reads `dr_params["light"]["brightness"]`/`["camera"]["rotation_deg"]`/
    `["background_color"]` unconditionally, keeps working unmodified for
    both sources (see this module's docstring's "Schema honesty note" for
    the real consequences of this approximation, spelled out rather than
    hidden).

    Args:
        image: The final (post-inpaint, for a `"defect"` sample; the raw
            placeholder surface, for an `"ok"` one) RGB image.

    Returns:
        A `dr_params`-shaped dict: `light.brightness` from the image's
        actual mean pixel intensity (normalized around
        `synthesize_placeholder_surface`'s own ~160 mean, so a neutral
        `~1.0` matches its default output), `camera.rotation_deg` fixed
        at `0.0` (no camera/pose concept for a flat inpainted surface --
        always buckets to `"center"`), `background_color` from the
        image's border-pixel mean (no separate background layer exists
        to read a ground-truth color from), and `source: "diffusion"` so
        this approximation is traceable in the manifest itself.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    brightness = float(array.mean() / 160.0)

    border_pixels = np.concatenate(
        [array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]]
    )
    background_color = [int(round(channel)) for channel in border_pixels.mean(axis=0)]

    return {
        "light": {"brightness": brightness},
        "camera": {"rotation_deg": 0.0},
        "background_color": background_color,
        "source": "diffusion",
    }


@dataclass
class _DiffusionCandidate:
    """One `--source diffusion` sample before the quality gate decides its fate.

    Attributes:
        sample: The candidate `Sample` (schema-complete already).
        image: The candidate's final pixels (inpainted, for `"defect"`;
            the raw placeholder surface, for `"ok"`).
        base_image: The pre-inpaint base image, needed by
            `quality_assessment.assess` -- `None` for `"ok"` samples
            (nothing was inpainted, so there's nothing to quality-gate).
        mask: The paint mask used, needed by `quality_assessment.assess`
            -- `None` for `"ok"` samples, for the same reason.
    """

    sample: Sample
    image: Image.Image
    base_image: Image.Image | None
    mask: np.ndarray | None


def generate_sample_diffusion(
    sample_index: int,
    image_size: tuple[int, int],
    defects_config: DefectsConfig,
    generation_config: GenerationConfig,
    diffusion_config: DiffusionConfig,
    backend: InpaintingBackend,
    rng: np.random.Generator,
) -> _DiffusionCandidate:
    """Render one candidate sample via week 05's mask-generation + inpainting pipeline.

    Reuses `generate_defect_mask` (same function `--source dr` uses) for
    the paint mask and `backend.inpaint` (`src.synthetic.diffusion_pipeline`,
    week 05 day 3 -- the exact code `scripts/run_diffusion_pipeline.py`
    runs) for the fill. Base images are `synthesize_placeholder_surface`
    placeholders (no real facility/product imagery), same as
    `run_diffusion_pipeline.py`'s own `--image-dir`-omitted default.

    Args:
        sample_index: Used to build `sample_id` and to seed the
            placeholder surface.
        image_size: `(width, height)` for the generated surface.
        defects_config: Parsed `defects.yaml`.
        generation_config: `generation:` section of `dataset_config.yaml`.
        diffusion_config: Parsed `diffusion.yaml`.
        backend: An `InpaintingBackend` from `build_backend`.
        rng: Shared source of randomness (label, defect mask).

    Returns:
        A `_DiffusionCandidate` -- not yet written to disk or added to
        the manifest; the caller quality-gates `"defect"` candidates
        first (see this module's docstring).
    """
    sample_id = f"sample_{sample_index:06d}"
    base_image = synthesize_placeholder_surface(
        image_size, seed=generation_config.seed + sample_index
    )

    is_defect = rng.random() < generation_config.defect_fraction
    image = base_image
    mask: np.ndarray | None = None
    defect_boxes: list[Bbox] = []
    if is_defect:
        mask, instances = generate_defect_mask(image_size, defects_config, rng)
        image = backend.inpaint(
            base_image, mask, diffusion_config.prompt.positive, diffusion_config.prompt.negative
        )
        defect_boxes = [instance.bbox for instance in instances]

    width, height = image.size
    sample = Sample(
        sample_id=sample_id,
        image_path=f"images/{sample_id}.jpg",
        width=width,
        height=height,
        label="defect" if is_defect else "ok",
        # Whole-frame, not a localized object placement -- see this
        # module's docstring's "Schema honesty note".
        object_bbox=(0, 0, width, height),
        defect_boxes=tuple(defect_boxes),
        dr_params=approximate_diffusion_dr_params(image),
    )
    return _DiffusionCandidate(
        sample=sample, image=image, base_image=(base_image if is_defect else None), mask=mask
    )


def select_diffusion_candidates(
    candidates: list[_DiffusionCandidate], diffusion_config: DiffusionConfig
) -> tuple[list[_DiffusionCandidate], int]:
    """Quality-gate `"defect"` candidates, keep every `"ok"` one unconditionally.

    Reuses `src.synthetic.quality_assessment.composite_scores`/
    `select_top_fraction` verbatim -- the same ranking/selection
    `scripts/run_diffusion_pipeline.py` already runs. `"ok"` candidates
    (no `mask`/`base_image` -- nothing was inpainted) never had anything
    to score, so they're never gated.

    Args:
        candidates: Every generated candidate, in original generation order.
        diffusion_config: Parsed `diffusion.yaml` (reads `quality_gate`).

    Returns:
        `(kept, num_rejected)`: `kept` is the accepted candidates, in
        original generation order (not re-sorted by score) -- so
        `sample_id`s stay in generation order in the final manifest even
        though some may be missing (a rejected sample's id is simply
        absent, not reused). `num_rejected` is how many `"defect"`
        candidates the gate dropped.
    """
    defect_candidates = [c for c in candidates if c.mask is not None]
    if not defect_candidates:
        return list(candidates), 0

    identifiers = [c.sample.sample_id for c in defect_candidates]
    metrics = [
        qa.assess(
            np.array(c.base_image.convert("RGB"), dtype=np.uint8),
            np.array(c.image.convert("RGB"), dtype=np.uint8),
            c.mask,
            ssim_window=diffusion_config.quality_gate.ssim_window,
        )
        for c in defect_candidates
    ]
    scored = qa.composite_scores(
        identifiers, metrics, diffusion_config.quality_gate.weights, payloads=defect_candidates
    )
    kept, rejected = qa.select_top_fraction(scored, diffusion_config.quality_gate.keep_fraction)
    kept_ids = {result.identifier for result in kept}

    final = [c for c in candidates if c.mask is None or c.sample.sample_id in kept_ids]
    return final, len(rejected)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `01_generate_synthetic.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    parser.add_argument(
        "--source",
        default="dr",
        choices=("dr", "diffusion"),
        help="'dr' (default, unchanged behavior): week 07 DR scene render + week 05 "
        "defect-mask blend. 'diffusion': week 05's mask-generation + real inpainting + "
        "quality-gate pipeline. See this script's module docstring for the decision "
        "criterion between the two.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=("cv2", "diffusers", "stability_api"),
        help="--source diffusion only: override diffusion.yaml's backend: for this run "
        "(default: config/diffusion.yaml's own 'cv2' -- no GPU/API key/network needed).",
    )
    return parser.parse_args()


def _generate_via_diffusion(
    config: DatasetPipelineConfig, backend_override: str | None
) -> tuple[list[Sample], str, int, float]:
    """Run the `--source diffusion` generation + quality-gate flow.

    Args:
        config: The loaded `dataset_config.yaml`.
        backend_override: `--backend`, or `None` to use `diffusion.yaml`'s
            own `backend:` (default `cv2`, no GPU/API key/network needed).

    Returns:
        `(samples, backend_name, num_rejected, keep_fraction)` -- `samples`
        are the accepted candidates only (see `select_diffusion_candidates`),
        already written to disk under `config.paths.data_dir`.

    Raises:
        SystemExit: Exit code 1, if the configured backend can't be built
            (e.g. `--backend diffusers` without `torch`/a real CUDA GPU) --
            same behavior as `scripts/run_diffusion_pipeline.py`.
    """
    generation = config.generation
    dr_parameters = load_dr_parameters(generation.dr_parameters_path)
    defects_config = load_defects_config(generation.defects_config_path)
    diffusion_config = load_diffusion_config(generation.diffusion_config_path)
    if backend_override is not None:
        diffusion_config = replace(diffusion_config, backend=backend_override)

    try:
        backend = build_backend(diffusion_config)
    except (ImportError, RuntimeError) as exc:
        print(
            f"Error: could not build backend {diffusion_config.backend!r}: {exc}", file=sys.stderr
        )
        sys.exit(1)

    rng = np.random.default_rng(generation.seed)
    image_size = dr_parameters.canvas.size

    candidates = [
        generate_sample_diffusion(
            sample_index, image_size, defects_config, generation, diffusion_config, backend, rng
        )
        for sample_index in range(generation.count)
    ]
    kept, num_rejected = select_diffusion_candidates(candidates, diffusion_config)

    samples: list[Sample] = []
    for candidate in kept:
        candidate.image.save(config.paths.data_dir / candidate.sample.image_path, quality=95)
        samples.append(candidate.sample)

    return samples, backend.name, num_rejected, diffusion_config.quality_gate.keep_fraction


def main() -> None:
    """Generate `generation.count` samples and write `manifest_generated.json`."""
    args = parse_args()
    config = load_config(args.config)
    generation = config.generation

    images_dir = config.paths.data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    backend_name: str | None = None
    if args.source == "diffusion":
        samples, backend_name, num_rejected, keep_fraction = _generate_via_diffusion(
            config, args.backend
        )
    else:
        # --source dr (default) -- unchanged from before --source existed.
        dr_parameters = load_dr_parameters(generation.dr_parameters_path)
        defects_config = load_defects_config(generation.defects_config_path)

        rng = np.random.default_rng(generation.seed)
        object_pool = [
            synthesize_placeholder_object() for _ in range(generation.object_pool_size)
        ]

        samples = []
        for sample_index in range(generation.count):
            image, sample = generate_sample(
                sample_index, object_pool, dr_parameters, defects_config, generation, rng
            )
            image.save(config.paths.data_dir / sample.image_path, quality=95)
            samples.append(sample)

    manifest_path = config.paths.data_dir / "manifest_generated.json"
    save_manifest(manifest_path, samples)

    num_defect = sum(1 for sample in samples if sample.label == "defect")
    print(
        f"Generated {len(samples)} sample(s): {num_defect} defect, "
        f"{len(samples) - num_defect} ok."
    )
    if backend_name is not None:
        print(
            f"Source: diffusion (backend={backend_name}). Quality gate kept "
            f"{keep_fraction:.0%} by composite score: rejected {num_rejected} inpainted "
            "'defect' candidate(s) (never applied to 'ok' samples -- nothing was inpainted "
            "to score)."
        )
    print(f"Manifest written to '{manifest_path}'.")


if __name__ == "__main__":
    main()
