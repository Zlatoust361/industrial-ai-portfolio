"""CLI: real-vs-synthetic defect comparison report + edge-deployment path (week 05, day 4).

Reads day 3's `run_diffusion_pipeline.py` output manifest (Method A:
inpainting) and, if available, week 06's committed `outputs/week06/`
demo artifacts (Method B: the DR-scene + blended parametric shape stage 6
dataset generation currently uses instead of real diffusion, per
`README.md`'s "Week 06" section). Writes a self-contained Markdown report
with real visual examples (copied PNGs, not links into gitignored
scratch directories) and metrics computed by real code -- no numbers in
`report.md` are hand-typed.

"Real vs synthetic" is honestly reframed here, not glossed over: this
repo has no photographed industrial-defect imagery at all (no client
data, session-wide rule -- see `README.md`). What IS real and comparable:

- Method A's own before/after pair -- the exact base image an inpainting
  result came from vs. the result itself, which IS a meaningful
  real-vs-synthetic-defect-region comparison (real surface, synthetic
  defect blended in).
- Two independently-real SYNTHETIC generation *methods* against each
  other -- Method A (day 3's mask + inpainting) vs Method B (weeks 06/07's
  DR-rendered scene + a blended-in parametric shape, currently what
  `dataset_pipeline/` actually uses for its `"defect"` samples).

Usage:
    PYTHONPATH=. python scripts/generate_defect_comparison_report.py \\
        [--inpainted-manifest outputs/synthetic_defects/inpainted/manifest.json] \\
        [--week06-sample-dir outputs/week06/sample_images] \\
        [--week06-manifest outputs/week06/manifest_split_demo.json] \\
        [--output-dir outputs/report]

Requires day 3's `outputs/synthetic_defects/inpainted/manifest.json` to
already exist -- run `make generate-masks && make inpaint` first if not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.synthetic.quality_assessment import METRIC_NAMES, blur_inside_mask, contrast_inside_mask

EXAMPLE_LABELS = ("best", "median", "worst")
THUMB_HEIGHT = 256


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `generate_defect_comparison_report.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inpainted-manifest", default="outputs/synthetic_defects/inpainted/manifest.json"
    )
    parser.add_argument("--week06-sample-dir", default="outputs/week06/sample_images")
    parser.add_argument("--week06-manifest", default="outputs/week06/manifest_split_demo.json")
    parser.add_argument("--output-dir", default="outputs/report")
    return parser.parse_args()


def _load_inpainted_manifest(path: Path) -> dict:
    """Load day 3's inpainting manifest, or exit with clear instructions if missing."""
    if not path.exists():
        print(
            f"Error: inpainted manifest not found: {path}\n"
            "Run day 2 + day 3 first, e.g.:\n"
            "    make generate-masks ARGS='--image-size 512x512 --count 12 --seed 7'\n"
            "    make inpaint",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(path.read_text())


def pick_examples(results: list[dict]) -> list[tuple[str, dict]]:
    """Pick 3 examples spanning the batch's composite-score range: best, median, worst.

    Deliberately not "the 3 best-looking results" -- a report that only
    shows cherry-picked wins would misrepresent what the quality gate
    actually filters. `worst` is very likely a `rejected` entry, showing
    concretely what gets filtered out and why.

    Args:
        results: `manifest.json`'s `"results"` list (day 3's output),
            non-empty.

    Returns:
        `[("best", entry), ("median", entry), ("worst", entry)]`. If
        `results` has fewer than 3 entries, some labels may repeat the
        same entry (e.g. a single-result batch returns it 3 times) rather
        than raising -- this report should degrade gracefully on a tiny
        demo batch, not fail outright.
    """
    ranked = sorted(results, key=lambda r: r["composite_score"], reverse=True)
    best = ranked[0]
    worst = ranked[-1]
    median = ranked[len(ranked) // 2]
    return list(zip(EXAMPLE_LABELS, (best, median, worst), strict=True))


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.BILINEAR)


def _label_strip(text: str, width: int) -> Image.Image:
    """A short white strip with `text` centered -- goes under each triptych panel."""
    strip = Image.new("RGB", (width, 20), color=(255, 255, 255))
    draw = ImageDraw.Draw(strip)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) / 2, 4), text, fill=(0, 0, 0), font=font)
    return strip


def make_triptych(
    original: Image.Image, mask: Image.Image, inpainted: Image.Image, labels: tuple[str, str, str]
) -> Image.Image:
    """Horizontally concatenate original | mask | inpainted, each with a caption strip.

    Args:
        original: The pre-inpaint base image.
        mask: The binary inpainting mask (rendered as a viewable grayscale panel).
        inpainted: The backend's output.
        labels: Caption text for each of the 3 panels, in order.

    Returns:
        One RGB image, all 3 panels resized to `THUMB_HEIGHT` px tall.
    """
    panels = [
        _resize_to_height(original.convert("RGB"), THUMB_HEIGHT),
        _resize_to_height(mask.convert("RGB"), THUMB_HEIGHT),
        _resize_to_height(inpainted.convert("RGB"), THUMB_HEIGHT),
    ]
    strips = [_label_strip(label, panel.width) for label, panel in zip(labels, panels, strict=True)]

    gap = 8
    total_width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (total_width, THUMB_HEIGHT + 20 + 4), color=(255, 255, 255))
    x = 0
    for panel, strip in zip(panels, strips, strict=True):
        canvas.paste(panel, (x, 0))
        canvas.paste(strip, (x, THUMB_HEIGHT + 4))
        x += panel.width + gap
    return canvas


def aggregate_metric_stats(results: list[dict]) -> dict[str, dict[str, float]]:
    """Mean/min/max per metric across `results`, plus per-status counts.

    Args:
        results: `manifest.json`'s `"results"` list.

    Returns:
        `{"count": {"selected": N, "rejected": N}, "metrics": {metric_name: {"mean",
        "min", "max"}}}`.
    """
    selected_count = sum(1 for r in results if r["status"] == "selected")
    stats = {
        "count": {"selected": selected_count, "rejected": len(results) - selected_count},
        "metrics": {},
    }
    for name in METRIC_NAMES:
        values = [r["metrics"][name] for r in results]
        stats["metrics"][name] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return stats


def method_b_metrics(
    week06_sample_dir: Path, week06_manifest_path: Path
) -> dict | None:
    """Blur/contrast inside week 06's real `defect_sample_000000`'s defect boxes.

    No PSNR/SSIM here: unlike Method A's inpainting (which by construction
    starts from a stored, identical base image), week 06's `"ok"`/`"defect"`
    samples are independent DR renders with different randomized
    camera/light/material parameters -- there is no pixel-aligned "before"
    image for `defect_sample_000000` to compare against, so an
    outside-mask fidelity metric wouldn't mean anything here. Blur/contrast
    (absolute, unpaired proxies) are still meaningful.

    Args:
        week06_sample_dir: `outputs/week06/sample_images/`.
        week06_manifest_path: `outputs/week06/manifest_split_demo.json`
            (for `sample_000000`'s real `defect_boxes`).

    Returns:
        `None` if any required file is missing (this report degrades
        gracefully rather than failing outright -- Method B is a bonus
        comparison, not this day's core deliverable). Otherwise a dict
        with `blur_inside_mask`/`contrast_inside_mask` and the source
        `defect_boxes` used to build the mask.
    """
    defect_path = week06_sample_dir / "defect_sample_000000.jpg"
    if not (defect_path.exists() and week06_manifest_path.exists()):
        return None

    manifest = json.loads(week06_manifest_path.read_text())
    sample = next(
        (s for s in manifest["samples"] if s["sample_id"] == "sample_000000"), None
    )
    if sample is None or not sample.get("defect_boxes"):
        return None

    image = np.array(Image.open(defect_path).convert("RGB"))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in sample["defect_boxes"]:
        mask[y0:y1, x0:x1] = 255

    return {
        "blur_inside_mask": blur_inside_mask(image, mask),
        "contrast_inside_mask": contrast_inside_mask(image, mask),
        "defect_boxes": sample["defect_boxes"],
        "source_sample_id": sample["sample_id"],
    }


EDGE_PATH_SECTION = """\
## The path to the edge: how the *dataset* (not this pipeline) reaches Jetson

The point of `diffusion_pipeline`/`quality_assessment` isn't to run on a
Jetson Orin Nano — it never will (see `README.md`'s "Week 05" section).
The point is the *dataset* it produces, which rides the same path every
other week's training data does, ending up serving a real detector on the
device. Concretely, by real file/module name in this repo:

1. **Generation** (this week). `scripts/run_diffusion_pipeline.py`'s
   `selected/` output (`outputs/synthetic_defects/inpainted/selected/` --
   the quality-gated top 75%) is a set of `(image, defect region)` pairs,
   the same shape `dataset_pipeline/scripts/01_generate_synthetic.py`'s
   `"defect"`-labeled samples already produce via week 05's mask generator
   + week 07's DR renderer (see `README.md`'s "Week 06" section). **Not
   yet wired together**: stage 1 doesn't call `diffusion_pipeline` today
   — adding a `--generation-method diffusion` mode there that reads from
   `selected/` instead of (or alongside) the DR+blended-shape path is the
   real next integration step, not attempted in this pass.
2. **Dataset assembly** (week 06, `dataset_pipeline/`). Once stage 1
   accepts diffusion-sourced images, stages 2-8 need NO changes at all --
   that's the actual payoff of the manifest-per-stage design
   (`dataset_pipeline/README.md`): `02_stratify_dataset.py` balances by
   brightness/angle/background regardless of how pixels were generated,
   `03_augment_dataset.py`/`04_convert_coco.py`/`05_convert_yolo.py`
   convert format, `06_split_dataset.py` does the stratified train/val/test
   split, `07_export_jetson_val_subset.py` produces the same
   200-300-image/~16 MB Jetson-sized val subset it already does today, and
   `08_validate_dataset.py` gates the whole manifest (bbox/empty-mask/
   duplicate integrity) before anything downstream trusts it.
3. **Training** (week 07). `scripts/train_dr_yolo.py` trains the
   `no_dr`/`with_dr` YOLOv8 branches. Today it reads directly from
   `render_dr_dataset.py`'s raw per-branch output, not from
   `dataset_pipeline`'s YOLO export — unifying the two into one training
   input (so a diffusion-augmented dataset can actually reach this step)
   is, again, real described future work, not implemented here.
4. **On-device export** (week 07 day 2 / week 09). `export_and_compare_jetson.py`
   exports the trained `.pt` checkpoint to a TensorRT `.engine` and
   compares fp32-vs-fp16 mAP on the device — the YOLO-detector analog of
   what `quantization/`'s 8-script INT4 pipeline does more thoroughly
   (calibration, INT4/INT8, a 0.98 cosine-similarity quality gate) for
   week 01's CLIP image encoder. Two independent model families, same
   destination hardware and the same "build the engine ON the Jetson,
   architecture/TensorRT-version-locked" constraint (`course/HARDWARE.md`).
5. **Deployment + ops** (weeks 08 and 10). `docker-compose.jetson.yml`/
   `Dockerfile.jetson` (week 08) is the deployment shape this repo already
   uses for a Jetson-resident inference service — today serving CLIP
   embeddings (`src/api/app.py`), not yet the YOLO detector; standing up a
   detector-serving container the same way is a natural next step, not
   done here. Whenever that happens, `monitoring/` and `backup/` (week 10)
   already apply unmodified — neither is CLIP-specific, both are generic
   to "a container running on this Jetson."

**What this section is not claiming**: an end-to-end pipeline that
today, right now, turns a `diffusion_pipeline` mask into a deployed
Jetson detector. It doesn't exist yet. What's real is every individual
link above, and the fact that each one's input/output shape already
matches its neighbor's — the integration gaps are specific, named, and
small (a new stage-1 generation mode; unifying two YOLO-format training
inputs), not an architectural rewrite.
"""


def render_report(
    week05_stats: dict,
    examples: list[tuple[str, dict]],
    method_b: dict | None,
    inpainted_manifest_path: Path,
) -> str:
    """Build the full `report.md` Markdown text.

    Args:
        week05_stats: `aggregate_metric_stats()`'s output for Method A.
        examples: `pick_examples()`'s output, with each entry's `triptych`
            path already filled in by `main()`.
        method_b: `method_b_metrics()`'s output, or `None` if unavailable.
        inpainted_manifest_path: For a "generated from" provenance line.

    Returns:
        The complete Markdown text.
    """
    lines: list[str] = []
    lines.append("# Real vs. synthetic defects — comparison report (week 05, day 4)")
    lines.append("")
    lines.append(f"Generated from `{inpainted_manifest_path}`.")
    lines.append("")
    lines.append(
        "**Honesty note up front**: this repository has no photographed real-world "
        "defect imagery of any kind (no client data — a session-wide rule, see "
        "`README.md`). \"Real vs. synthetic\" is therefore two things, both real "
        "comparisons, neither a claim of comparing against a real photograph:"
    )
    lines.append(
        "1. Method A's own **before/after** — the real base image an inpainting "
        "result came from, vs. the synthetic defect blended into it."
    )
    lines.append(
        "2. Two independently-real **synthetic generation methods** compared to "
        "each other — inpainting (this week) vs. the DR-scene + blended parametric "
        "shape `dataset_pipeline/` currently uses instead (weeks 06/07)."
    )
    lines.append("")

    lines.append("## Method A: mask + inpainting (week 05, days 2-3)")
    lines.append("")
    lines.append(
        f"{week05_stats['count']['selected']} selected / "
        f"{week05_stats['count']['rejected']} rejected out of "
        f"{week05_stats['count']['selected'] + week05_stats['count']['rejected']} total "
        "results, backend `cv2` (see `README.md`'s \"Week 05\" section for why the "
        "diffusion backends can't run in this sandbox)."
    )
    lines.append("")
    lines.append("Aggregate metrics across the whole batch:")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("|---|---|---|---|")
    for name in METRIC_NAMES:
        m = week05_stats["metrics"][name]
        lines.append(f"| `{name}` | {m['mean']:.4f} | {m['min']:.4f} | {m['max']:.4f} |")
    lines.append("")

    lines.append("### Examples: best, median, and worst by composite score")
    lines.append("")
    lines.append(
        "Deliberately not just the best-looking results — `worst` shows concretely "
        "what the quality gate filters out and why."
    )
    lines.append("")
    for label, entry in examples:
        lines.append(f"#### {label.capitalize()}: `{entry['identifier']}` ({entry['status']})")
        lines.append("")
        lines.append(f"![{label} example]({entry['triptych_relative']})")
        lines.append("")
        lines.append(f"Composite score: **{entry['composite_score']:.4f}**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for name in METRIC_NAMES:
            lines.append(f"| `{name}` | {entry['metrics'][name]:.4f} |")
        lines.append("")

    lines.append("## Method B: DR scene + blended parametric shape (weeks 06/07)")
    lines.append("")
    if method_b is None:
        lines.append(
            "**Unavailable**: `outputs/week06/sample_images/defect_sample_000000.jpg` "
            "and/or `outputs/week06/manifest_split_demo.json` were not found. Run "
            "week 06's demo pipeline first if you want this section populated "
            "(`dataset_pipeline/README.md`)."
        )
    else:
        lines.append(
            "Real committed demo artifacts from week 06's `dataset_pipeline/` run "
            f"(`{method_b['source_sample_id']}`) — the method `dataset_pipeline/`'s "
            "stage 1 currently actually uses for `\"defect\"` samples, not this "
            "week's inpainting."
        )
        lines.append("")
        lines.append("![Method B ok vs. defect](images/method_b_side_by_side.png)")
        lines.append("")
        lines.append(
            "No PSNR/SSIM outside-mask here — unlike Method A's inpainting, "
            "`\"ok\"`/`\"defect\"` are independent DR renders with different "
            "randomized camera/light/material parameters, not a pixel-aligned "
            "before/after pair. Blur/contrast inside the real `defect_boxes` "
            f"(from `{method_b['source_sample_id']}`'s manifest entry) are still "
            "meaningful, unpaired:"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| `blur_inside_mask` | {method_b['blur_inside_mask']:.4f} |")
        lines.append(f"| `contrast_inside_mask` | {method_b['contrast_inside_mask']:.4f} |")
        lines.append("")

    lines.append("## Method A vs. Method B: what's actually different")
    lines.append("")
    lines.append(
        "- **Texture consistency**: Method A's `cv2.inpaint` fills the masked "
        "region by propagating the *surrounding pixels'* texture inward — the "
        "generated defect region shares statistics with its own neighborhood by "
        "construction. Method B blends a flat-colored parametric shape onto a "
        "DR-rendered scene — visibly a shape overlay, not a texture-consistent "
        "surface flaw (see `dataset_pipeline/README.md`'s own \"Why generation "
        "looks the way it does\" section)."
    )
    lines.append(
        "- **What each is a stand-in for**: Method A's `cv2` backend stands in for "
        "real Stable Diffusion inpainting (day 3's `diffusers`/`stability_api` "
        "backends are real code, just unrunnable in this sandbox). Method B's "
        "blended shape is week 06's own stand-in for the SAME thing — both are "
        "honestly documented substitutions for the same missing piece, arrived at "
        "independently in two different weeks."
    )
    lines.append(
        "- **What's actually wired into `dataset_pipeline/` today**: Method B, "
        "exclusively — see \"The path to the edge\" below for what changes to "
        "connect Method A instead."
    )
    lines.append("")

    lines.append(EDGE_PATH_SECTION)
    return "\n".join(lines)


def main() -> None:
    """Build the comparison report end to end."""
    args = parse_args()

    inpainted_manifest_path = Path(args.inpainted_manifest)
    inpainted_manifest = _load_inpainted_manifest(inpainted_manifest_path)
    results = inpainted_manifest["results"]
    if not results:
        print("Error: inpainted manifest has no results -- nothing to report.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    week05_stats = aggregate_metric_stats(results)

    examples = pick_examples(results)
    for label, entry in examples:
        original = Image.open(entry["original_image"])
        mask = Image.open(entry["mask"])
        inpainted = Image.open(entry["image"])
        triptych = make_triptych(
            original, mask, inpainted, labels=("original", "mask", "inpainted")
        )
        triptych_path = images_dir / f"{label}_{entry['identifier']}_triptych.png"
        triptych.save(triptych_path)
        entry["triptych_relative"] = f"images/{triptych_path.name}"

    method_b = method_b_metrics(Path(args.week06_sample_dir), Path(args.week06_manifest))
    if method_b is not None:
        defect_image = Image.open(Path(args.week06_sample_dir) / "defect_sample_000000.jpg")
        ok_image = Image.open(Path(args.week06_sample_dir) / "ok_sample_000003.jpg")
        # A plain ok|defect side-by-side (not the 3-panel triptych helper,
        # which assumes a mask panel Method B doesn't have in the same shape).
        ok_panel = _resize_to_height(ok_image.convert("RGB"), THUMB_HEIGHT)
        defect_panel = _resize_to_height(defect_image.convert("RGB"), THUMB_HEIGHT)
        gap = 8
        side_by_side = Image.new(
            "RGB", (ok_panel.width + defect_panel.width + gap, THUMB_HEIGHT), color=(255, 255, 255)
        )
        side_by_side.paste(ok_panel, (0, 0))
        side_by_side.paste(defect_panel, (ok_panel.width + gap, 0))
        side_by_side.save(images_dir / "method_b_side_by_side.png")

    metrics_summary = {
        "inpainted_manifest": str(inpainted_manifest_path),
        "method_a": week05_stats,
        "method_b": method_b,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))

    report_text = render_report(week05_stats, examples, method_b, inpainted_manifest_path)
    (output_dir / "report.md").write_text(report_text)

    print(f"Report written to '{output_dir / 'report.md'}'.")
    print(f"Metrics summary written to '{output_dir / 'metrics_summary.json'}'.")
    print(f"Method B: {'included' if method_b is not None else 'unavailable, skipped'}.")


if __name__ == "__main__":
    main()
