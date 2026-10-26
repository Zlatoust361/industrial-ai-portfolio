# pipeline/ — inspection pipeline (week 04)

Week 04 deliverable: a 7-stage industrial visual-inspection pipeline —
`scripts/01_environment_check.py` … `scripts/07_generate_report.py` — built
on top of week 01/02's CLIP scaffold (`../src/`) and week 03's YOLOv8
comparison work. A separate top-level directory from `../src`/`../scripts`
on purpose: this week's own numbered scripts (`01_...` … `07_...`) would
otherwise collide with week 01's `../scripts/01_download_models.py`/
`02_convert_onnx.py`/`03_benchmark.py` in the same folder, and this week's
config schema (7 pipeline stages, quality-gate thresholds, a report title)
has nothing in common with `../config/model_config.yaml`'s.

## Running the whole pipeline

```bash
cd pipeline
make setup      # installs ../requirements.txt + ../requirements-dev.txt (skip if the repo root already ran this -- same venv, same files)
make pipeline   # all 7 stages via run_all.sh, demo dataset + stub backend, zero setup
make test       # pytest tests/  (74 passed)
make lint       # ruff check .
make report     # regenerate outputs/report.md alone (e.g. after tweaking quality_gate thresholds)
make pipeline-jetson   # like `make pipeline`, but --device jetson (fp16 + batch=1 in 04_detect.py)
make clean      # remove data/outputs/logs/.cache
```

`make pipeline`/`make pipeline-jetson` default to `config/demo_config.yaml`
(`detect.backend: stub`) so a fresh checkout runs end to end with **zero
setup** — no network, no model weights, no real inspection images (see
"CI/demo backend" below). Pass `CONFIG=config/pipeline_config.yaml` (or
your own config) for a real `clip`/`yolo` run against real `data/raw/`:

If you do this with `make pipeline-jetson` **on the Jetson itself**, `make
setup`'s plain `pip install -r ../requirements.txt` is the wrong dependency
source for that run — it pulls torch from a CPU-only x86_64 wheel index
that isn't valid on aarch64. Use a venv with `--system-site-packages` so
`torch` resolves to JetPack's own CUDA-enabled build instead (see
`course/VENV.md`'s Jetson section for the exact steps); the `stub` backend
(default, no `CONFIG=`) doesn't import torch at all, so this only matters
once you pass a real `clip`/`yolo` config.

```bash
make pipeline CONFIG=config/pipeline_config.yaml
```

### Running a single stage

Every stage is also runnable directly, config paths resolved relative to
the invocation cwd (`pipeline/`):

```bash
cd pipeline
python3 scripts/01_environment_check.py [--config ...]
python3 scripts/02_ingest.py [--config ...]
python3 scripts/03_preprocess.py [--config ...]
python3 scripts/04_detect.py [--config ...] [--device dev|jetson]
python3 scripts/05_interpret.py [--config ...]
python3 scripts/06_quality_gate.py [--config ...]
python3 scripts/07_generate_report.py [--config ...]
```

Each script inserts its own `import` path (`pipeline/` for `utils.*`, the
repo root for `04_detect.py`'s `src.*` reuse) at the top, so no
`PYTHONPATH=` juggling is needed to run them directly.

## `run_all.sh`

Chains all 7 stages, generating a synthetic demo dataset first if
`<data_dir>/raw/` is empty (`scripts/generate_demo_dataset.py`). Stops
immediately if stages 1-5 fail; stage 6 (`06_quality_gate.py`) is handled
specially -- its exit code (`0` GREEN/YELLOW, `1` RED) becomes
`run_all.sh`'s own exit code, but stage 7 always runs first regardless, so
a RED verdict still produces a report explaining why, instead of silently
skipping it. This is exactly what CI's `pipeline` job runs to gate the
build on `outputs/metrics/quality_gate.json`.

Every stage's stdout+stderr is duplicated (via `tee`) into
`<logs_dir>/NN_stage_name.log` as well as the terminal -- the individual
stage scripts don't know about `logs_dir` at all; `run_all.sh` is the one
place that already sequences all 7, so it's also the one place that needs
to know logging exists. Running a stage directly (see above) only prints,
it doesn't write a log file.

## GitHub Actions

`.github/workflows/ci.yml`'s `pipeline` job (parallel to the existing
`lint-and-test` job for weeks 01-02) lints `pipeline/`, runs its 51 unit
tests, then runs `./run_all.sh` (demo dataset + `stub` backend -- no
`torch`/`transformers`/`ultralytics` install needed, so this job stays fast
and independent of model-weight network access) and fails the build if the
resulting quality gate is RED. `outputs/report.md` and
`outputs/metrics/quality_gate.json` are uploaded as a build artifact either way.

## Layout

```
pipeline/
  run_all.sh                 # chains all 7 stages -- see "run_all.sh" below
  Makefile                   # setup/test/lint/pipeline/pipeline-jetson/report/clean
  config/
    pipeline_config.yaml      # real config -- detect.backend: clip
    demo_config.yaml           # same schema, detect.backend: stub -- run_all.sh's default
  utils/
    config.py                # PipelineConfig loader + apply_device_target(dev|jetson)
    environment.py             # is_jetson / L4T+JetPack parsing / memory / power mode
    manifest.py                  # image discovery + PIL validation (02, 03)
    preprocess.py                  # letterbox-resize geometry, pure math (03)
    interpret.py                     # detection scores -> ok/defect verdicts (05)
    quality_gate.py                    # defect_rate -> GREEN/YELLOW/RED (06)
    report.py                            # renders the final Markdown report (07)
  scripts/
    01_environment_check.py    # writes outputs/env_report.json
    02_ingest.py                 # data/raw/* -> data/manifest.json
    03_preprocess.py               # -> data/processed/* + data/processed_manifest.json
    04_detect.py                     # -> outputs/detections.json (clip | yolo | stub backend)
    05_interpret.py                    # -> outputs/interpretations.json
    06_quality_gate.py                   # -> outputs/metrics/quality_gate.json
    07_generate_report.py                  # -> outputs/report.md
    generate_demo_dataset.py                 # synthetic data/raw/ for run_all.sh's demo mode
  tests/                        # pytest, PYTHONPATH=. pytest tests/  (see "Testing")
  data/, outputs/, logs/        # created at runtime, gitignored (see ../.gitignore)
```

## Data flow

```
data/raw/*.jpg                                    (you provide these)
  -> 01_environment_check.py -> outputs/env_report.json      (informational, no data/ input)
  -> 02_ingest.py            -> data/manifest.json
  -> 03_preprocess.py        -> data/processed/*, data/processed_manifest.json
  -> 04_detect.py            -> outputs/detections.json
  -> 05_interpret.py         -> outputs/interpretations.json
  -> 06_quality_gate.py      -> outputs/metrics/quality_gate.json
  -> 07_generate_report.py   -> outputs/report.md
```

Every stage reads only the previous stage's artifact (plus the config) and
fails with a clear "run `NN_previous.py` first" error if it's missing —
nothing implicitly re-runs an earlier stage.

## Config (`utils/config.py`)

`PipelineConfig` mirrors week 01's `src.config.AppConfig` pattern: one
frozen dataclass per YAML section, `load_config()` parses and validates
(`device.target`/`detect.backend` are checked against a fixed set, same as
week 01's `TARGETS`), `apply_device_target(config, "jetson")` forces
`fp16` + `batch_size=1` the same way week 01's `apply_target_overrides`
forces `fp16` (see `course/DECISIONS.md` ADR-002 for why `memory_budget_mb`
defaults to 6144 MB on an 8192 MB Orin Nano). Not reused directly from
`src/config.py` because the schemas don't overlap — this week's is about
pipeline stages and thresholds, not a single CLIP checkpoint's inference
settings.

## Stage 1: environment check

`01_environment_check.py` is the one stage day 2's prompt calls out
explicitly: it must determine whether it's running on a Jetson (checked via
`/etc/nv_tegra_release`, JetPack's own marker file — see
`utils.environment.is_jetson`) and log the JetPack/L4T version, available
memory, and current `nvpmodel` power mode to `outputs/env_report.json`.

Two honesty notes baked into `utils/environment.py`:

- **JetPack version is reported as a coarse generation (`"6.x"`), not an
  exact minor version.** `/etc/nv_tegra_release` reliably gives the L4T/BSP
  version (`"R36.4.0"`-style); the L4T-major → JetPack-generation mapping
  (`R36` → `6.x`) is well known, but the L4T-minor → JetPack-minor mapping
  (e.g. exactly `6.1` vs `6.2`) changes between releases and isn't reliably
  inferable from this file alone. The report's `l4t_version` field is
  exact; `jetpack_generation` is explicitly labeled a coarser derived
  value, and `course/VENV.md`'s `sudo apt show nvidia-jetpack` remains the
  authoritative source for the precise version.
- **Detecting the device does not set `device.target`.** `01_environment_check.py`
  only *reports*; `04_detect.py`'s dev/jetson behavior is driven by the
  config's `device.target` (or its `--device` flag, see "Stage 4" below),
  set explicitly by whoever runs the pipeline. Auto-switching stage behavior
  off what stage 1 observes would make a dev machine's "pretend I'm a
  Jetson" dry run (fp16, batch=1, for testing the jetson profile without
  the hardware) impossible to express.

## Stage 4: detection backends

`04_detect.py` supports three backends, chosen by `detect.backend`:

- **`clip`** (default, real): zero-shot ok/defect classification. Reuses
  week 02's already-tested `src.models.zero_shot`/`src.data.prompts`/
  `src.models.clip` code directly (`../config/prompts.yaml`'s
  `basic`/`industrial`/`detailed` styles) rather than reimplementing
  zero-shot classification a second time — this stage just batches it over
  `data/processed/` instead of stdout. Raw CLIP cosine similarities sit
  around 0.15-0.35, nowhere near a `[0, 1]` threshold, so `defect_score` is
  a plain 2-way softmax over the ok/defect similarities (see
  `_softmax_defect_probability`'s docstring) — not a calibrated
  probability, just enough structure for `interpret.defect_score_threshold`
  to compare against.
- **`yolo`** (real): `ultralytics.YOLO` object detection. A stock
  COCO-pretrained checkpoint has no `"defect"` class, so
  `detect.yolo_defect_classes: []` means "every detected box counts as a
  defect" — fine for exercising the wiring, not a real signal; point it at
  a defect-trained checkpoint's actual class name(s) for real use.
- **`stub`** (CI/demo only, see below): a deterministic, dependency-light
  pixel-marker check.

All three lazily import whatever they need (`torch`/`transformers`,
`ultralytics`, or nothing) only inside `detect_batch()`, so importing
`04_detect.py` itself never requires any of them installed. All three
also process images in batches of `device.batch_size` (forced to `1` by
`--device jetson`) — for `clip` this is a real batched `encode_image` call,
not a loop calling a single-image method one at a time; the memory-vs-
throughput tradeoff this controls is exactly week 01's `memory_budget_mb`
guardrail (`course/DECISIONS.md` ADR-002).

**`clip`/`yolo` not runnable in this sandbox**: no network access to
download CLIP/YOLO weights, and `torch`/`ultralytics` themselves aren't
installed here (see `course/DECISIONS.md` ADR-004). Smoke-tested to the
exact expected boundary instead: with a real `data/processed_manifest.json`
in place, `detect.backend: clip` reaches `ModuleNotFoundError: No module
named 'torch'` inside `CLIPLoader.load()` — including with `--device
jetson` set, proving the batching/override plumbing around it is correct
too — and `detect.backend: yolo` reaches `ModuleNotFoundError: No module
named 'ultralytics'`.

### CI/demo backend: `stub`

`StubDetector` (in `04_detect.py`) needs only `numpy`/`PIL` — no model, no
weights, no network. It checks each image for a bright-red marker rectangle
(RGB ~ `(220, 30, 30)`) and scores `0.9` if enough pixels match, `0.05`
otherwise. `scripts/generate_demo_dataset.py` paints that exact marker onto
a configurable fraction of its synthetic images (`--defects N`), so the two
together form a real, if trivial, end-to-end signal — not a hardcoded
constant — that exercises `05_interpret.py`/`06_quality_gate.py`/
`07_generate_report.py` against genuine (synthetic) image content. This is
what `run_all.sh`'s default run and CI's `pipeline` job use; it is **not**
a substitute for `clip`/`yolo` in a real inspection run.

## Testing

All 7 stages run for real end to end in this sandbox via `make pipeline`
(demo dataset + `stub` backend, no heavy deps needed) — a clean 12-image
demo run produces a genuine GREEN verdict and exit code `0`; regenerating
the demo dataset with `--defects 2` produces a genuine RED verdict, exit
code `1`, and `outputs/report.md` still lists the two flagged images (see
`run_all.sh`'s exit-code handling). `make report` was verified to
regenerate `outputs/report.md` alone from already-produced outputs.
`clip`/`yolo` (real backends) were separately smoke-tested to their exact
expected import boundary — see "Stage 4" above.

`utils/`'s pure logic — `config.py`, `environment.py`, `manifest.py`,
`preprocess.py`, `interpret.py`, `quality_gate.py`, `report.py` — has unit
tests, independent of `../tests/` and `../ros2_ws/`'s own pytest roots (no
shared imports):

```bash
cd pipeline
PYTHONPATH=. pytest tests/    # 74 passed
```

### `test_script_*.py`: CLI/`main()` coverage, not just `utils/`

`test_<module>.py` files test `utils/`'s pure logic in isolation;
`test_script_*.py` files test each stage's actual `main()`/CLI wiring end
to end (`conftest.py`'s `load_script_module` imports a numbered
`scripts/*.py` file as a module — its name isn't a valid Python
identifier, so a plain `import` can't reach it — and `write_pipeline_config`
writes a minimal valid config to `tmp_path`):

- **`test_script_01_environment_check.py`** is the mocked "are we on a
  Jetson Orin Nano" test: `01_environment_check.py` exposes `--marker-path`
  specifically so tests can point it at a fixture file instead of the real
  `/etc/nv_tegra_release` (`build_environment_report()`'s default argument
  binds once at import time — a real Python gotcha — so monkeypatching the
  *function's* default wouldn't have worked here). One run with the marker
  file present (`is_jetson: true`, real L4T/JetPack fields populated), one
  with it absent (`is_jetson: false`, everything else `null`), plus the
  off-device stderr warning appearing in exactly one of the two.
- **`test_script_04_detect.py`** is where "config actually switches stage
  behavior" is verified concretely: `StubDetector.detect_batch` is spied on
  while running `04_detect.py` with and without `--device jetson` against
  the same 5-image demo manifest and `device.batch_size: 3` — without the
  flag, batches come in as `[3, 2]` (the configured size); with it, `[1, 1,
  1, 1, 1]` (forced to `1`). That's `apply_device_target`'s effect actually
  reaching the detection loop, not just the config dataclass (already
  covered by `test_config.py`).
- **`test_script_03_preprocess.py`**/**`test_script_06_quality_gate.py`**/
  **`test_script_07_generate_report.py`** run the real script against
  hand-written upstream artifacts: real PIL letterboxing verified pixel-size
  and manifest-geometry correct, `06_quality_gate.py`'s exit code checked
  against all three GREEN/YELLOW/RED cases (the part `run_all.sh`/CI
  actually depend on, not just `evaluate_quality_gate`'s return value), and
  `07_generate_report.py`'s Markdown output checked against real upstream
  JSON. Each also covers its "run the earlier stage first" error path.
