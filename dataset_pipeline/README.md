# dataset_pipeline/ -- synthetic dataset pipeline (week 06)

A separate top-level directory, same reasoning as `pipeline/` (week 04,
see `course/DECISIONS.md` ADR-014): its numbered
`scripts/01_generate_synthetic.py` ... `08_validate_dataset.py`
would collide with the root `scripts/01_download_models.py`/
`02_convert_onnx.py`/`03_benchmark.py` (week 01), and its own schema
(generation/stratify/augment/split/Jetson-subset parameters) has nothing
in common with `src/config.py` or `pipeline/utils/config.py`.

Not a fully isolated workspace like `ros2_ws/`, though: stage 1
(`scripts/01_generate_synthetic.py`) legitimately imports
`src.domain_randomization.scene`/`src.synthetic.mask_generation` from the
repo root -- reusing week 07's DR scene renderer and week 05's parametric
defect-mask generator directly, rather than reimplementing scene
compositing a second time (same pattern as `pipeline/scripts/04_detect.py`
importing `src.models.zero_shot` for its `clip` backend).

## Why generation looks the way it does

Real Stable Diffusion inpainting (week 05 days 3-5) was never
implemented -- needs `torch`/`diffusers` + GPU, unavailable in the
sandbox this course was built in (see `course/DECISIONS.md` ADR-004 and
`README.md`'s "Week 05" section). Stage 1 doesn't wait for that: a
`"defect"` sample here is a DR-rendered scene
(`src.domain_randomization.scene.render_scene`, `rng != None` --
independently randomized per sample, not week 07's `no_dr`/`with_dr`
comparison branches) with a generated defect mask
(`src.synthetic.mask_generation.generate_defect_mask`) blended visibly
onto it (`overlay_defect_mask`) -- a real, checkable synthetic defect
signal, not a photorealistic inpaint, and not a hardcoded marker either
(each defect's shape/size/position is actually sampled).

That's stage 1's **default** (`--source dr`, or no flag at all) --
see the next section for its selectable alternative.

## Two synthetic-data sources: DR rendering vs. diffusion inpainting

`course/TECH-DEBT.md` C.1 named this gap: week 05's diffusion-inpainting
pipeline (`src/synthetic/diffusion_pipeline.py`,
`scripts/run_diffusion_pipeline.py` -- fully real, tested, working) was
never connected to stage 1, which only had the DR-rendering path above.
Stage 1 now takes `--source dr|diffusion` (default `dr`, so every
existing invocation, script, and test keeps its exact prior behavior
unless `--source diffusion` is passed explicitly):

- **`--source dr`** (default): `src.domain_randomization.scene.render_scene`
  + `src.synthetic.mask_generation.generate_defect_mask`, as above.
- **`--source diffusion`**: reuses week 05's *full* days 2-3 pipeline --
  `generate_defect_mask` for the paint mask (same function, same config,
  `config/defects.yaml`), then `src.synthetic.diffusion_pipeline.
  build_backend`/`.inpaint` for a real (non-mocked) inpainting pass over
  a synthetic textured placeholder surface
  (`synthesize_placeholder_surface` -- the same zero-setup base
  `scripts/run_diffusion_pipeline.py` uses when no `--image-dir` of real
  images is given), then `src.synthetic.quality_assessment`'s
  composite-score top-fraction gate to drop the worst inpaints. Uses
  `config/diffusion.yaml`'s default `cv2` backend (OpenCV classical
  inpainting) -- **no `torch`/GPU/API key/network needed**, exactly
  `make defect-pipeline`'s own zero-setup demo default.

### Decision criterion (the actual resolution of C.1)

These are two real, valid synthetic-data strategies with a genuine
engineering tradeoff -- neither replaces the other:

- Pick **`dr`** when geometry/pose diversity (viewpoint, lighting,
  background) and automatically-correct-by-construction ground truth
  matter most -- e.g. training a detector that needs precise
  object/defect localization. `object_bbox`/`defect_boxes` come
  directly from the 3D-scene compositor and mask generator, exactly
  right by construction, no verification step needed.
- Pick **`diffusion`** when photorealistic surface/texture fidelity on a
  real (or real-textured) background image matters most, and an
  approximate/manually-spot-checked mask is acceptable -- e.g. training
  a classifier/anomaly detector on how a real material surface actually
  looks with a defect, not a flat-color overlay.

This is a **reasoned engineering judgment, not an experimentally
validated one** -- no comparative study (e.g. downstream detector mAP
trained on one source vs. the other) has been run in this repo to prove
either source is objectively better for any specific metric. See
`course/DECISIONS.md` ADR-XXX.

### Schema honesty: real, load-bearing differences (not papered over)

Both sources write the exact same `utils.manifest.Sample` schema, so
stages 2-8 run unmodified either way -- verified for real, end to end,
0 validation errors, not just "the field names line up". But the two
sources do NOT produce equivalent data, and pretending otherwise would
be dishonest:

- **`object_bbox`**: `dr` gives a real, localized object-placement box.
  `diffusion` has no separate "object on a background" concept -- the
  whole generated image *is* the surface -- so `object_bbox` is the
  trivial full-frame `(0, 0, width, height)`. Anything downstream that
  wants a tight object-localization box gets nothing useful from a
  diffusion-sourced sample on that field.
- **`defect_boxes`**: `dr`'s overlay forcibly makes the masked region
  visibly different (a real, checkable signal by construction).
  `diffusion`'s `defect_boxes` come from the *input paint mask* handed
  to the inpainter (same `generate_defect_mask` function) -- correct for
  "this is the region we told it to alter", but NOT a verified post-hoc
  segmentation of what the backend actually painted there. A backend
  could blend a result that doesn't visually read as a defect at all
  inside that same box; the quality gate filters the worst offenders by
  fidelity/sharpness/contrast, it doesn't re-segment or guarantee
  visible defect content.
- **`dr_params`**: real per-sample DR output for `dr`; for `diffusion`
  it's *approximated* from the generated image's own pixels
  (`light.brightness` from mean intensity, `background_color` from
  border-pixel mean) purely so stage 2's stratification keeps working,
  with `camera.rotation_deg` hardcoded `0.0` (no camera-angle concept
  for a flat inpainted surface) and an explicit `"source": "diffusion"`
  marker for traceability. Real consequence, verified: stage 2's angle
  axis is degenerate (always `"center"`) for a diffusion-sourced
  dataset, and its brightness/background diversity is far lower than
  `dr`'s.
- **Sample count**: `dr` always emits exactly `generation.count`
  samples. `diffusion`'s quality gate can emit *fewer* -- a rejected
  `"defect"` candidate is dropped, not retried or replaced (`"ok"`
  samples are never gated -- nothing was inpainted to score).

See `scripts/01_generate_synthetic.py`'s module docstring for the same
detail alongside the code itself.

## The 8 stages

Each stage is a thin CLI wrapper around `utils/`'s tested pure logic,
reading the previous stage's output and writing a new, more complete
artifact (never mutating one file in place -- same "immutable record, one
file per stage" shape as `pipeline/utils/manifest.py`, week 04):

1. **`01_generate_synthetic.py`** -- `<data_dir>/images/*.jpg` +
   `manifest_generated.json`. See above.
2. **`02_stratify_dataset.py`** -- discretizes each sample's *actual* DR
   render parameters (not the config's sampling ranges) into
   brightness/angle/background buckets (`utils/stratify.py`), joined with
   the `ok`/`defect` label into one `stratum` string. Also computes each
   axis's *marginal* distribution independently (not just the joint
   stratum) plus a `[0, 1]` balance ratio per axis (day 4) -- logs a
   warning (stderr + the report) when an axis drops below
   `stratify.balance_warn_ratio` (default `0.3`), informational, not a
   hard gate (stage 8 owns correctness). Written to
   `outputs/metrics/stratify_distribution.json`.
3. **`03_augment_dataset.py`** -- adds one augmented variant for
   `augment.augment_fraction` of samples: horizontal flip (geometric,
   boxes flipped to match exactly) or brightness/contrast jitter/light
   gaussian noise (photometric, boxes untouched) -- see
   `utils/augment.py`. Deliberately not rotation/crop, which need real
   bbox-clipping math this stage doesn't need.
4. **`04_convert_coco.py`** -- `outputs/coco/annotations.json`. Two
   categories: `generation.object_class_name` (every sample's whole-object
   bbox) and `generation.defect_class_name` (each defect shape's bbox).
5. **`05_convert_yolo.py`** -- `outputs/yolo/{images,labels,classes.txt}`.
   Same `class_id cx cy w h` normalized-line convention as
   `src.evaluation.detection_metrics` (week 07) -- this output is
   directly loadable by `load_yolo_test_set`/consumable by week 07's
   `train_dr_yolo.py`/`compare_dr_branches.py`, closing the "week 06 val
   subset doesn't exist yet" gap those scripts' docstrings named (see
   `course/DECISIONS.md` ADR-022).
6. **`06_split_dataset.py`** -- stratified train/val/test split
   (`utils/split.py`, largest-remainder apportionment *per stratum*, not
   a flat percentage -- with 26-36 strata and a demo-sized dataset, most
   strata have 1-2 samples, and a naive "round down, dump the leftover in
   test" implementation would send every single-sample stratum to test
   unconditionally, badly skewing the aggregate split). An augmented
   sample (stage 3) always inherits its *source* sample's split, so a
   flipped/jittered twin can never leak across train/val/test. Then
   physically reorganizes stage 4/5's output to match: the flat YOLO
   `images/`/`labels/` become `{train,val,test}/` (each a complete,
   independent YOLO root), and the COCO file is partitioned into
   `annotations_{train,val,test}.json`.
7. **`07_export_jetson_val_subset.py`** -- picks
   `jetson_subset.target_count` (200-300 by default) images from the val
   split, proportionally across strata (same largest-remainder logic), and
   copies just those into `outputs/jetson_val_subset/`. The full dataset
   is never copied to the device -- limited NVMe/microSD on-device storage
   is exactly the constraint this stage exists for.
8. **`08_validate_dataset.py`** (day 3) -- dataset-integrity gate over
   everything stages 1-7 produced (`utils/validate.py`): bbox validity
   (in-bounds, non-degenerate), empty masks (a `"defect"` sample with no
   defect boxes, or an `"ok"` sample carrying one -- both a contract
   violation given `generate_defect_mask`'s `count_range` always places
   >=1 shape), duplicates (`sample_id`, `image_path`, a repeated bbox
   within one sample), plus a file-level cross-check against each split's
   actually-written YOLO directory and a self-consistency check on the
   Jetson subset. Writes `outputs/validation_report.json` and **exits 1
   if any error-level issue is found** -- a real gate (same "fail loud,
   don't ship broken data" stance as `pipeline/scripts/06_quality_gate.py`,
   week 04), not just an advisory report; warnings (e.g. a duplicate bbox
   within one sample) are reported but don't fail the gate. Verified for
   real both ways: 0 errors on a genuine pipeline run, and a real exit-1
   failure against a deliberately corrupted copy of the same manifest --
   see `outputs/week06/`.

   **Caught a real, pre-existing bug on its first run**: week 05's
   `generate_defect_mask` could place a `"crack"` (polyline) shape whose
   bbox spilled outside the image, because `_shape_extent` budgeted
   margin for it as if the random walk were centered on its sampled point
   (correct for `line`/`ellipse`, which actually are) -- but
   `_draw_polyline`'s walk *starts* there and can wander up to its full
   length away in one direction. Fixed in
   `src/synthetic/mask_generation.py::_shape_extent` (see
   `outputs/week06/report.md` for the exact repro numbers), with a
   200-seed regression test added to `tests/test_mask_generation.py`.

## Running it

```bash
make setup   # install runtime + dev deps (shared with the repo root)
make test    # 92 passed -- utils/ pure logic + script-level CLI wiring, real files/images throughout
make lint    # ruff check .
make pipeline                        # ./run_all.sh, default config/dataset_config.yaml
make pipeline CONFIG=path/to/config.yaml
make jetson-subset                   # re-run just stage 7
make clean                           # rm -rf data outputs

# Stage 1 only, with the diffusion-inpainting source instead of DR rendering
# (see "Two synthetic-data sources" above) -- everything else (stages 2-8)
# runs exactly the same either way:
PYTHONPATH=. python3 scripts/01_generate_synthetic.py --source diffusion
```

`config/dataset_config.yaml`'s default `generation.count: 60` is a fast,
zero-setup demo -- `make pipeline`/`./run_all.sh` runs all 8 stages for
real (no `torch`/GPU/network needed anywhere in this chain), but stage 7
will clearly fail (exit 1, not silently skip) past that point:
`jetson_subset.min_count` needs at least 200 val samples, and 60 total
samples spread across up to 36 strata (2 labels x 3 brightness buckets x
3 angle buckets x 2 background buckets) doesn't produce that many; stage
8 still runs and validates whatever stages 1-6 produced. A real run needs
`generation.count` in the thousands (~1500 comfortably clears
`min_count: 200` with the default split fractions) -- verified for real in
this sandbox: 1500 generated -> 1950 after augmentation -> 296 val samples
-> 250-image Jetson subset, all 8 stages genuinely executed, not
smoke-tested to an import boundary (see `outputs/week06/` at the repo
root for the actual demo artifact from that run, including stage 8's
real validation report both clean and against a deliberately corrupted
manifest).

## Tests

`tests/test_*.py` cover each `utils/` module's pure logic in isolation
(config parsing, manifest round-tripping, stratum bucketing, bbox-safe
augmentation, COCO/YOLO writers, stratified split apportionment, Jetson
subset selection, dataset-integrity validation) -- real files/images
throughout (`PIL`/`numpy`/`cv2` aren't mocked anywhere).
`tests/test_script_pipeline.py` runs the actual 8-script chain end to end
via each script's `main()`, plus each stage's "clear error if the
previous stage hasn't run yet" contract, and stage 8's exit-1-on-real-
corruption path specifically. `tests/test_generate_synthetic_diffusion.py`
(C.1) covers `--source diffusion` specifically: `approximate_diffusion_dr_params`/
`generate_sample_diffusion`/`select_diffusion_candidates`'s pure logic in
isolation, plus its output feeding cleanly into stage 2 (stratification)
and passing stage 8 (validation) end to end -- real `cv2` inpainting
throughout, no mocking (the default backend needs neither GPU nor
network). Run with `make test` (from `dataset_pipeline/`, `PYTHONPATH=.`).
