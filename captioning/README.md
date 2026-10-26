# captioning/ — week 15: VLM captioning/VQA on edge

Status: **first implementation pass**, part of
[`course/ROADMAP-WEEKS-13-24.md`](../course/ROADMAP-WEEKS-13-24.md)'s
volume 2 (not wired into `course.yaml`/`validate_course.py` — see that
roadmap's open question #1). Lives at the repo root, next to
`tracking/`/`depth_fusion/`/`peft/`/`fleet/`, not under `course/weeks/`
(same split every other week already has: `course/weeks/weekNN-.../`
documents the week, this directory IS the week's code).

## Idea

Every detector in this course (weeks 02-09) outputs a class and a
confidence score, never a sentence. A quality-control report that says
`defect, score=0.87` doesn't tell a human what's actually wrong with the
part — the formulation in WORDS ("scol на кромке, ~3 мм, вдоль шва") is
what an ОТК report actually needs, and that requires a generative VLM
(captioning/VQA), not CLIP's embeddings (weeks 01-02 are similarity
search, not text generation).

This week's answer: a small instruction-tuned image-to-text model, called
ONLY on a confirmed defect (not every frame — a second generative model
can't realistically run alongside the detector on every frame of an 8 GB
Jetson Orin Nano), constrained to a narrow industrial vocabulary (not open
dialogue), feeding its output straight into week 04's existing pipeline
report as one more field.

## What's new here vs. reused from other weeks

| Piece | Status |
|---|---|
| `model.py::CaptioningLoader` | **New**, but same `VLMLoader` contract as `src.models.clip.CLIPLoader` — load/unload, memory-budget-checked before and after loading weights via `src.models.memory`. No `VLMInference` subclass: captioning is generative, not embedding, so forcing it onto `encode_image`/`encode_text` would be the wrong fit (same judgment call `tracking/adapters.py` makes about not force-fitting an interface). |
| `prompting.py` | **New** — narrow-vocabulary prompt construction + `describe_defect()`, with the caption call itself INJECTED (`CaptionFn`), same pattern `assistant/vision_bridge.py`'s `client` parameter established. |
| `triggers.py` | **New** — `on_defect_confirmed` (default) vs `on_every_frame`, the Jetson-budget tradeoff this week names explicitly. |
| `integration.py` | **New**, small — augments week 04's `pipeline.utils.interpret.interpret_detections` verdict shape with a `caption` field, WITHOUT modifying `pipeline/` itself (reads the same shape, writes its own output). |
| `evaluation.py` | **New**, reuses `src.models.clip.CLIPInference` (weeks 01/02's text branch) directly for a reference-free caption-quality signal — not a new embedding model. |
| Config (`utils/config.py`) | Reuses `src.config.ModelSection`/`MemorySection` directly, same shape `peft/utils/config.py` already established. |

## Pipeline

```
scripts/run_captioning_demo.py
        │  generates a deterministic SYNTHETIC set of week-04-pipeline-
        │  shaped verdicts (some "ok", some "defect") -- see its own
        │  docstring for exactly why synthetic, not a real inspection run
        ▼
    triggers.should_caption() per verdict
        │  on_defect_confirmed (default): only "defect" verdicts proceed
        ▼
    prompting.describe_defect()  ->  model.generate_caption_text()
        │  REAL small instruction-tuned VLM call (CaptioningLoader)
        ▼
    evaluation.clip_caption_similarity()
        │  REAL CLIP call (src.models.clip) -- reference-free quality signal
        ▼
    outputs/captioning_demo/{augmented_verdicts.json,evaluation_report.json}
        │
        ▼
    NEXT (real hardware): wire augment_verdicts_with_captions() directly
    into week 04's 07_generate_report.py, against a REAL detector's
    verdicts, not synthetic ones -- not yet done in this pass (see
    Honesty note below).
```

## Why `on_defect_confirmed`, not `on_every_frame`

Named explicitly in `course/ROADMAP-WEEKS-13-24.md`'s week 15 entry: a
second (generative) model next to the detector probably can't share an
8 GB Jetson Orin Nano with it running concurrently every frame. Captioning
is called only once a defect is already confirmed by the (cheaper)
detector — the trigger is a config value (`captioning/triggers.py`), not
a hardcoded assumption, and `on_every_frame` stays as a documented,
selectable alternative (not deleted) so the cost of the "wrong" choice is
visible in code, not hidden.

## Why the vocabulary constraint is a prompt hint, not an enforced grammar

`prompting.build_prompt()` lists allowed phrases (`scratch`, `crack`, ...)
INSIDE the instruction prompt — a soft constraint the model can still
ignore. A hard constraint (grammar-guided/constrained decoding) was
considered and left out of this first pass: it adds a real decoding-time
dependency for a course that otherwise keeps every model call a plain
`generate()`, and the course's own research step for this week
(`course/PROMPTS-WEEKS-13-24.md`'s P15.2) explicitly asks to weigh this
tradeoff before committing to either — not yet resolved with a real
comparison in this sandbox (no real model available to compare against,
see Honesty note).

## Honesty note — not yet run against a real VLM checkpoint in this sandbox

Same network boundary every other week's own README documents: this
sandbox has no route to `huggingface.co`
(`CaptioningLoader.load()`'s real `AutoModelForImageTextToText.from_pretrained()`
call reaches it and fails with `httpx2.ProxyError: 403 Forbidden` —
confirmed by actually running `make demo`, not assumed). What IS verified
for real in this sandbox (24 tests, all passing):

- `model.py::CaptioningLoader`: memory-budget check is wired up correctly
  (a too-low available-RAM value raises `MemoryBudgetExceededError`
  BEFORE any network call, confirmed by monkeypatching
  `available_system_memory_mb`); the real load path is confirmed to reach
  exactly the real network boundary, not an unrelated bug — this is also
  how the `AutoModelForVision2Seq` → `AutoModelForImageTextToText` rename
  (this installed `transformers==5.20.0` removed the former) was actually
  caught: the first test run failed with `ImportError`, not a network
  error, which `assert_reaches_network_boundary` correctly flagged as a
  real bug in this week's own code, not a sandbox limitation.
- `prompting.py`: prompt construction includes the label and allowed
  vocabulary; `describe_defect()` with an injected fake `CaptionFn`
  returns the right `DefectCaption` shape and measures real latency.
- `triggers.py`: full coverage of both modes plus the invalid-trigger
  error path.
- `integration.py`: `on_defect_confirmed` captions only `"defect"`
  verdicts (confirmed `"ok"` entries get `caption=None`, not skipped
  silently from the output); `on_every_frame` captions everything;
  input list is never mutated; original verdict fields survive untouched.
- `evaluation.py`: `clip_caption_similarity()` against a duck-typed fake
  encoder with known vectors — identical vectors give similarity `1.0`,
  orthogonal give `0.0`, opposite give `-1.0` (exact arithmetic, not
  "roughly plausible").
- `scripts/run_captioning_demo.py`: determinism for a fixed seed, seed
  divergence, the zero-defect boundary case (`mean_clip_similarity is
  None`, not a crash or a fabricated `0.0`), and `main()`'s report-writing
  — all with injected fakes; the REAL path (`make demo`) is confirmed to
  reach exactly the real network boundary described above.

**Update (real dev-machine run, reported 2026-10-24)**: `make demo`
reached a real network, downloaded the real `HuggingFaceTB/
SmolVLM-256M-Instruct` checkpoint, and hit a real crash in
`generate_caption_text` before a single caption was produced:
`ValueError: The total number of <image> tokens in the prompts should
be the same as the number of images passed. Found [0] <image> tokens
and [1] images per sample.` Root cause: SmolVLM is built on the
Idefics3 architecture, whose processor requires the prompt TEXT to
already contain an `<image>` placeholder token matching the images
passed — unlike some other VLM processors, it does not insert this
automatically. `generate_caption_text` was calling
`loader.processor(images=image, text=prompt, ...)` with `prompt` as
plain text from `build_prompt()` (no placeholder at all). Fixed by
building a proper chat-formatted prompt via
`loader.processor.apply_chat_template()` with an explicit `{"type":
"image"}` content block before the text, the standard/documented usage
pattern for Idefics3-family processors — `build_prompt()`/`CaptionFn`'s
own plain-string contract is unchanged, the chat-template wrapping
happens only at this one real call site. **Not independently verified
end-to-end by this sandbox** (still no network route to
`huggingface.co` here) — the fix is grounded in the real traceback and
SmolVLM's documented usage pattern, not re-confirmed by a second real
run; the next real dev-machine run of `make demo` is what actually
closes this out.

**Not yet done**: a reference-free similarity score computed against
real generated captions (now unblocked by the fix above, still not
actually run end-to-end — see the update directly above). Also not
done: wiring
`augment_verdicts_with_captions()` directly into week 04's
`07_generate_report.py`/`render_report()` so a real pipeline run's own
JSON report carries captions end-to-end (this week's `integration.py`
proves the augmentation logic against week-04-SHAPED data, not a live
`pipeline/` run); a hard vocabulary constraint via constrained decoding
(see the section above); and running two real models (detector +
captioning) concurrently on one real Jetson to measure the ACTUAL peak
memory/latency cost this week's `on_defect_confirmed` default is
justified by — currently a documented assumption, not a measurement.

## Usage

```bash
cd captioning
make setup
make demo                         # defaults: 20 images, defect_rate=0.3, seed=42
make demo ARGS="--num-images 50 --defect-rate 0.4 --seed 7"
```

See `captioning/Makefile` for each target's exact CLI, and
`course/weeks/week15-edge-vlm-captioning/MANUAL-RUN.md` for the
code-level walkthrough and known-failure-boundary details.
