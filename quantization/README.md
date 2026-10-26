# quantization/ — INT4 quantization + TensorRT (week 09)

Week 09 deliverable: baseline FP16 → calibration → INT4/INT8 TensorRT
engine → FP16-vs-INT4 measurement → final `summary.md`, as 8 numbered
scripts (`01_baseline_inference.py` … `08_profile.py`, all 8 now
implemented as of day 4). A separate top-level directory from
`../src`/`../scripts` on purpose: this week's own numbered scripts would
otherwise collide with week 01's
`../scripts/01_download_models.py`/`02_convert_onnx.py`/`03_benchmark.py`,
and this week's config schema (calibration settings, TensorRT builder
profiles) has nothing in common with `../config/model_config.yaml`'s —
same reasoning as `../pipeline/` (week 04) and `../dataset_pipeline/`
(week 06).

## Running (the full 8-step recipe)

Steps 1-3 and 8 run on the dev machine; steps 4-7 run on the Jetson itself
— **two different machines, two different `make setup*` targets**. Don't
run plain `make setup` on the Jetson: `../requirements.txt` pulls torch
from a CPU-only x86_64 wheel index that has no aarch64 match at all, which
aborts the whole `pip install` (pyyaml/numpy/pillow included) before
anything installs — and steps 4-7 don't even need torch (tensorrt/pycuda
come from JetPack, imported lazily, never via pip).

```bash
# --- dev machine ---
cd quantization
make setup              # installs ../requirements.txt + ../requirements-dev.txt (skip if the repo root already ran this -- same venv, same files)
make baseline            # 01: FP16 latency/throughput -> outputs/baseline/
make export-onnx          # 02: ONNX export + PyTorch-vs-ONNXRuntime smoke test
make prepare-calibration   # 03: sample + resize calibration images from week 06's dataset

# --- on the Jetson itself (copy outputs/ from steps 1-3 over first) ---
make setup-jetson            # installs ../requirements-jetson.txt (no torch, no CPU wheel index)
make quantize FORCE=1       # 04: TensorRT calibration cache
make build-engine FORCE=1    # 05: serialize the quantized .engine
make build-engine-fp16 FORCE=1 # 05 --fp16-only: the FP16 reference engine
make trt-inference ENGINE=outputs/engine/image_encoder_fp16.engine LABEL=fp16 MODE=1 FORCE=1
make trt-inference ENGINE=outputs/engine/image_encoder.engine LABEL=int4 MODE=1 FORCE=1
make trt-inference ENGINE=outputs/engine/image_encoder_fp16.engine LABEL=fp16 MODE=0 FORCE=1
make trt-inference ENGINE=outputs/engine/image_encoder.engine LABEL=int4 MODE=0 FORCE=1
make evaluate-quality MODE=1 FORCE=1  # 07: FP16-vs-INT4 embedding quality gate

# --- either machine ---
make profile                            # 08: aggregate all of the above -> outputs/summary.md
make test                     # pytest tests/
make lint                      # ruff check .
```

`make quantize`/`make build-engine`/`make build-engine-fp16`/
`make trt-inference`/`make evaluate-quality` refuse to run unless
`/etc/nv_tegra_release` exists (i.e. this is a real Jetson) — `FORCE=1`
skips that check to exercise the scripts' argument handling on a dev
machine, but the result won't be a deployable/meaningful artifact (see
"Why steps 4-7 are Jetson-gated" below). `FORCE=1` on a dev machine still
uses whatever `make setup` installed there — `make setup-jetson` is only
needed on the real device.

### Running a single stage directly

```bash
cd quantization
python3 scripts/01_baseline_inference.py [--config config/config.yaml] [--image PATH]
python3 scripts/02_export_onnx.py [--config config/config.yaml] [--skip-smoke-test]
python3 scripts/03_prepare_calibration_data.py [--config config/config.yaml] [--source-dir PATH] [--num-samples N]
python3 scripts/04_quantize_model.py [--config config/config.yaml] [--profile orin_nano_8gb] [--force]
python3 scripts/05_build_engine.py [--config config/config.yaml] [--profile orin_nano_8gb] [--fp16-only] [--force]
python3 scripts/06_trt_inference.py --engine PATH --label fp16|int4 [--nvpmodel-mode 0|1|2]
    [--config config/config.yaml] [--profile orin_nano_8gb] [--force]
python3 scripts/07_evaluate_quality.py [--config config/config.yaml] [--profile orin_nano_8gb]
    [--nvpmodel-mode 0|1|2] [--force]
python3 scripts/08_profile.py [--config config/config.yaml] [--inference-dir PATH]
    [--quality-report PATH] [--baseline-report PATH] [--output PATH]
```

Each script inserts its own import path (`quantization/` for `utils.*`,
the repo root for `01`/`02`/`06`/`07`'s `src.*` reuse) at the top, so no
`PYTHONPATH=` juggling is needed to run them directly --
`test_script_imports_standalone.py` runs every script with `--help` and no
`PYTHONPATH` set at all, to actually prove this rather than assume it (see
"Testing" below for why that test exists).

## Why steps 1-3 and 8 aren't gated, and 4-7 are

Steps 1-3 (baseline inference, ONNX export, calibration-data prep) are
portable: FP16 PyTorch inference, ONNX export, and PIL image resizing all
run identically on a dev machine or a Jetson. Steps 4-7 all touch
TensorRT directly (the calibrator interface, the builder, or loading and
running a serialized `.engine`) — a `.engine` file (or a calibration cache
computed against one GPU's kernel selection) is not portable across GPU
architectures/TensorRT versions, the same hard constraint week 03's
[`ros2_ws/scripts/convert_owlvit_tensorrt.py`](../ros2_ws/scripts/convert_owlvit_tensorrt.py)
and week 07's [`scripts/export_and_compare_jetson.py`](../scripts/export_and_compare_jetson.py)
already document. `utils/environment.py::require_jetson` (checked against
JetPack's own `/etc/nv_tegra_release` marker, week 04's
`pipeline/utils/environment.py::is_jetson`) refuses to run any of the four
scripts elsewhere by default; `--force` exists purely so this repo's own
tests (and a curious dev) can exercise the argument handling without a
device, not as something you'd pass on a real run. Step 8
(`08_profile.py`, day 4) is portable again — it only reads the JSON files
steps 1/6/7 already wrote and renders Markdown, nothing TensorRT-specific
about that, so it's the one step in this whole week runnable from a plain
dev checkout with real, non-smoke-tested output (see "Day 4" below).

## `tensorrt`/`pycuda` aren't in any `requirements*.txt`

Unlike `torch`/`onnx`/`onnxruntime` (all real `../requirements.txt`
dependencies), `tensorrt` ships as part of JetPack itself — present
already on a real Jetson Orin Nano's Python environment (or the
`nvcr.io/nvidia/l4t-pytorch` base image `../Dockerfile.jetson` uses), not
something you `pip install` generically off-device. `pycuda` (used by
`utils/tensorrt_builder.py`'s calibrator, and `utils/trt_inference.py`'s
engine wrapper, for device memory allocation) is the standard companion
for TensorRT's Python API and is installed the same way, on-device.
Neither is installed in this sandbox (nor is `onnx`/`onnxruntime`, despite
being listed in `../requirements.txt` — see `../README.md`'s week 01
section) — `01`/`02` are smoke-tested to `torch`'s import boundary, `04`
to `tensorrt`'s, `05` (default mode) and `06`/`07` to `pycuda`'s
(`build_calibrator`/`TrtImageEncoder.__init__` both import `pycuda` before
`tensorrt`), `05 --fp16-only` to `tensorrt`'s (no calibrator needed, so no
`pycuda` import on that path).

## Day 3: `06_trt_inference.py` + `07_evaluate_quality.py` — FP16 vs INT4

Comparing FP16 against INT4 needs **two** engines, so `05_build_engine.py`
gained `--fp16-only`: a plain FP16 build (`trt.BuilderFlag.FP16` only, no
calibrator, no calibration cache needed at all) written to
`paths.fp16_engine_path`, alongside the existing quantized build at
`paths.engine_path`. `utils/trt_inference.py::TrtImageEncoder` wraps
either one identically (deserialize, allocate I/O device buffers via
`pycuda`, run via TensorRT's modern `set_tensor_address`/
`execute_async_v3` API — available since TensorRT 8.5, so it works on
JetPack 6.0's bundled TensorRT 8.6 as well as newer 10.x releases, unlike
`BuilderFlag.INT4` itself, which 8.6 most likely does not have) — both
`06` and `07` share this one wrapper rather than each reimplementing
TensorRT buffer management.

- **`06_trt_inference.py`** — latency (p50/p95/mean/throughput, via
  `src.evaluation.benchmark.LatencyStats`, reused rather than a third
  copy of that shape) for **one** engine, tagged `--label fp16`/`--label
  int4`. Run it once per engine per power mode — four times total for a
  full FP16-vs-INT4-at-15W-and-MAXN-Super comparison.
- **`07_evaluate_quality.py`** — per-image cosine similarity between the
  FP16 and INT4 engines' embeddings on the exact same fixed test set
  (`utils/test_set.py::discover_test_images`, sorted-and-capped, not
  randomly resampled — the point is the *same* images across every
  engine/power-mode run so the numbers are comparable). A real gate, like
  week 04/06's `06_quality_gate.py`/`08_validate_dataset.py`: exits `1` if
  the mean similarity drops below `evaluation.min_cosine_similarity`
  (`0.98` by default).

### Fixed power mode: 15W standard, MAXN Super for the upper bound

Both scripts accept `--nvpmodel-mode {0,1,2}` (`utils/power_mode.py`,
duplicated from `../scripts/jetson_benchmark.py`'s own `set_nvpmodel`/
`TegrastatsMonitor` — same "separate top-level package, no cross-import"
stance as `utils/environment.py`, reusing only the hardware-free data
model in `src.evaluation.jetson`). Passing it switches the device via
`sudo nvpmodel -m <id>` and waits `--settle-seconds` (default 3s) before
measuring; omitting it benchmarks under whatever mode the device is
already in (still recorded via `nvpmodel -q`, so the report never claims
a mode it didn't actually check). The day's ask — 15W (mode `1`) as the
standard operating point, MAXN Super (mode `0`) as the upper bound — means
running `06` four times (2 engines × 2 modes) and `07` at least once per
mode of interest; `06_trt_inference.py`'s own docstring has the exact
four-command sequence. `06` also samples `tegrastats` for the whole
benchmark window (`utils/power_mode.py::TegrastatsMonitor`) and reports
mean/max per-sensor temperatures alongside latency — day 4's
`08_profile.py` aggregates all of these JSON reports into one summary.

## Day 4: `08_profile.py` — aggregate everything into `summary.md`

`utils/profile_report.py` does the actual work (pure JSON-in, Markdown-out,
no `tensorrt`/`pycuda`/`torch` anywhere in it — the one `utils/*.py` module
this week that isn't hardware-gated); `08_profile.py` itself is a thin CLI
around it, same "logic in `utils/`, orchestration in `scripts/`" split as
every other script this week.

- **Discovers** every `outputs/trt_inference/*.json` `06_trt_inference.py`
  wrote (required — the one input `08` can't proceed without), and
  optionally loads `outputs/quality/quality_report.json` (`07`) and
  `outputs/baseline/baseline_results.json` (`01`) if they exist, rather
  than failing when a comparison run skipped one of them.
- **Groups** reports by `nvpmodel` power-mode name (from each report's own
  recorded `metadata.power_mode`, not the filename — `"unknown"` if a
  report never queried `nvpmodel` at all), then pairs `fp16`/`int4` within
  each mode.
- **Speedup table**: `fp16_mean_ms / int4_mean_ms` and the same for p95,
  per power mode. A mode missing either engine's report gets an explicit
  "missing '<label>' report for this mode" note instead of a fabricated
  or silently-skipped number.
- **Trade-off table**: one row per engine/power-mode combination — batch
  size, mean latency, throughput, and (for `int4` rows) the quality
  report's mean cosine similarity to `fp16` plus its PASSED/FAILED gate
  verdict. `fp16` rows show `1.0000 (reference)` rather than repeating a
  meaningless self-similarity computation.
- **Temperature/throttling section**: peak `tegrastats` reading *per
  sensor, across every run in the whole test cycle* (not just the last
  one), flagged against `evaluation.throttle_temp_c` (85.0°C default —
  same value and "configurable proxy, not a real DVFS trip point" caveat
  as week 08's `src/api/metrics.py::THROTTLE_TEMP_C`), plus a per-run
  breakdown table. Reports a clear "no `tegrastats` readings" note instead
  of an empty/misleading table when no run had any (e.g. this sandbox,
  which has no `tegrastats` binary at all).

Genuinely runnable and tested end to end in this sandbox (see "Testing"
below): `test_script_08_profile.py` writes real synthetic JSON fixtures to
disk and asserts on the actual rendered `summary.md`, and a manual run
against a fuller, more realistic 4-report/quality/baseline fixture set
produced a genuine, readable summary (all values traced back to their
inputs, not hand-written) — a curated version of that demo may land in
`outputs/week09/` at day 5's review, per this repo's usual pattern (see
e.g. `outputs/week07/`, `outputs/week08/`).

## Config (`utils/config.py`, `config/config.yaml`)

`QuantizationConfig` mirrors week 01's `src.config.AppConfig`/week 04's
`pipeline.utils.config.PipelineConfig` pattern: one frozen dataclass per
YAML section, `load_config()` parses and validates. The one new shape:
`profiles:` is a *named* dict of TensorRT builder-sizing presets, not a
fixed two-value switch — `QuantizationConfig.get_profile(name)` looks one
up, defaulting to `active_profile`.

### The `orin_nano_8gb` profile

```yaml
profiles:
  orin_nano_8gb:
    batch_size: 1
    workspace_size_mb: 1024
    fp16_fallback: true
```

Picked so a new engineer building this for the first time doesn't have to
hand-tune these from scratch:

- **`batch_size: 1`** — the Orin Nano's edge scenario is one video stream
  at a time, nothing to batch (same reasoning as week 04/08's
  `device.batch_size=1`/`mem_limit=6g` defaults).
- **`workspace_size_mb: 1024`** — TensorRT's builder uses this as scratch
  memory for kernel autotuning *during the one-time build*, carved out of
  the same 8 GB the OS, other services, and the model's own
  weights/activations all share (Jetson has no separate VRAM). Kept well
  under the `memory_budget_mb: 6144` ceiling this repo has used since week
  01 ([ADR-002](../course/DECISIONS.md)) rather than maximizing it — the
  builder doesn't need the whole budget to explore its kernel search
  space, and grabbing more than necessary risks starving whatever else is
  running on the device during the build.
- **`fp16_fallback: true`** — sets `trt.BuilderFlag.FP16` alongside the
  low-bit flag: not every layer in a ViT/CLIP-style model has an INT4 (or
  even INT8) TensorRT kernel on Orin's Ampere (sm_87) build, and letting
  the builder fall back to fp16 per-layer instead of failing the whole
  build outright is standard TensorRT practice.

A separate `dev` profile (`workspace_size_mb: 4096`) exists purely for
`--force`-ing steps 4-5 on a dev machine with more generous assumed
headroom — never actually reaches a real build in this sandbox either way.

## INT4 vs INT8 calibration — an honesty note

TensorRT's `IInt8EntropyCalibrator2` API (used by
`utils/tensorrt_builder.py::build_calibrator`) has been stable for years
and is what `04_quantize_model.py` uses for the calibration cache
regardless of target bit-width — despite the "Int8" name, the same
calibration data is what `05_build_engine.py` feeds the builder whether
`quantization.bits` is 4 or 8 (the bit-width itself is a separate builder
flag, applied only in step 5's real, kept build). `trt.BuilderFlag.INT4`
is newer and more version/hardware-gated than `INT8` — most likely newer
than JetPack 6.0's bundled TensorRT 8.6 (INT4 weight-only quantization
support landed around TensorRT 10.x); `configure_builder_config` checks
it exists on the installed `tensorrt` build before using it and raises a
clear error otherwise (pointing at `quantization.bits: 8` as the
fallback), rather than silently building something the config didn't ask
for. **None of this has been run against real TensorRT or a real
Jetson** — `tensorrt`/`pycuda` aren't installed in this sandbox at all
(see above), so every TensorRT-specific code path here (including
`utils/trt_inference.py`'s engine-execution wrapper, day 3) is written to
the best of this session's knowledge of the Python API's shape, not
verified against a live install. See `course/DECISIONS.md`'s week 09 day
2 and day 3 entries for the full accounting of what's real vs. unverified.

## Layout

```
quantization/
  Makefile                    # setup/test/lint/baseline/export-onnx/prepare-calibration/
                               # quantize/build-engine/build-engine-fp16/trt-inference/
                               # evaluate-quality/profile/clean
  config/
    config.yaml                 # model/paths/baseline/onnx/calibration/evaluation/quantization
                                 # + profiles: {dev, orin_nano_8gb}
  utils/
    config.py                     # QuantizationConfig loader + get_profile()
    environment.py                  # is_jetson / require_jetson (self-contained, see week 04's pipeline/utils/environment.py)
    tensorrt_builder.py               # calibrator, ONNX parsing, builder-config setup (tensorrt/pycuda, lazy-imported)
    trt_inference.py                    # TrtImageEncoder, benchmark_trt_encoder, cosine_similarity (day 3)
    power_mode.py                         # nvpmodel switching + tegrastats sampling (day 3, dup. of jetson_benchmark.py's)
    test_set.py                             # discover_test_images -- deterministic, capped (day 3)
    profile_report.py                         # JSON aggregation + summary.md rendering, no hardware deps (day 4)
  scripts/
    01_baseline_inference.py    # -> outputs/baseline/baseline_results.json
    02_export_onnx.py             # -> outputs/onnx/image_encoder.onnx
    03_prepare_calibration_data.py  # -> outputs/calibration_data/*.png + manifest.json
    04_quantize_model.py              # ON THE JETSON ONLY -> outputs/calibration/int4_calibration.cache
    05_build_engine.py                  # ON THE JETSON ONLY -> outputs/engine/image_encoder[_fp16].engine
    06_trt_inference.py                   # ON THE JETSON ONLY -> outputs/trt_inference/<label>_<mode>.json
    07_evaluate_quality.py                  # ON THE JETSON ONLY -> outputs/quality/quality_report.json
    08_profile.py                             # portable -> outputs/summary.md
  tests/                        # pytest, PYTHONPATH=. pytest tests/
  outputs/                      # created at runtime, gitignored (see ../.gitignore)
```

## Testing

Steps 1/2 (torch-dependent), 4 (tensorrt-dependent), and 5-7
(tensorrt/pycuda-dependent) are smoke-tested to their exact expected
import boundary, same as every other hardware/network-gated script in
this repo (see `../course/DECISIONS.md` ADR-004 for the original
pattern). Step 3 (calibration data prep) needs only `PIL`/`numpy`/stdlib
— both real dependencies of this sandbox — so it runs for real end to
end: sampling, resizing, determinism across repeated runs with the same
seed, the "fewer candidates than requested" warning path, and the "source
directory doesn't exist" error path are all genuinely exercised, not just
asserted. `utils/trt_inference.py::cosine_similarity` (pure NumPy) and
`utils/test_set.py::discover_test_images` (pure `pathlib`) are likewise
tested for real, independent of `tensorrt`/`pycuda`.

Step 8 (`08_profile.py`/`utils/profile_report.py`) needs no hardware deps
at all and is the most thoroughly real-tested script this week: grouping
by power mode, speedup computation (including the "missing engine for
this mode" and "zero latency" edge cases), peak-temperature aggregation
across a whole synthetic test cycle, the thermal-warning threshold, and
the full rendered `summary.md` (including the "no quality report"/"no
`tegrastats` readings" degraded-gracefully paths) are all asserted against
real output, not mocked.

`test_script_imports_standalone.py` runs every `scripts/*.py` with
`--help` and `PYTHONPATH` explicitly stripped from the subprocess
environment — a real regression test, added after it caught a real bug
during day 3: `07_evaluate_quality.py` initially only added
`quantization/` to `sys.path`, not the repo root, so its (new) import of
`utils.power_mode` (which itself reuses `src.evaluation.jetson`) failed
for a genuine standalone invocation. Every `load_script_module`-based
test in this directory missed it, because root `pyproject.toml`'s
`[tool.pytest.ini_options] pythonpath = ["."]` puts the repo root on
`sys.path` for any pytest run regardless of cwd — masking exactly this
class of bug. Fixed by adding the missing `sys.path.insert`; the new test
now guards against it recurring, here or in any future script.

```bash
cd quantization
PYTHONPATH=. pytest tests/   # 86 passed
```
