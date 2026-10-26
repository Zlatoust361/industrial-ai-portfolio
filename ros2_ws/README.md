# ros2_ws/ — vision_bridge ROS2 workspace

Week 03 deliverable: a ROS2 package that bridges an image stream + text
query into open-vocabulary detections. Separate top-level workspace from
the rest of this repo (`src/`, `scripts/`, `course/`) on purpose — ROS2
packages are built and tested with `colcon`/`ament`, not `pip`/`pytest`
directly, and a colcon build only ever sees files inside this directory.

## Layout

```
ros2_ws/src/
  vision_bridge_msgs/          # ament_cmake: BoundingBox, DetectionResult, DetectionArray
    msg/*.msg
    CMakeLists.txt
    package.xml
  vision_bridge/                # ament_python: the node + launch file
    vision_bridge/
      vision_bridge_node.py      # the rclpy Node (VisionBridgeNode) + main()
      detection.py                 # Detection/BoundingBox dataclasses, Detector protocol, StubDetector
      detector_factory.py           # build_detector(): auto/tensorrt/pytorch/stub backend selection
      owlvit_pytorch.py               # PyTorchOwlViTDetector: eager transformers OWL-ViT
      owlvit_tensorrt.py                # TensorRTOwlViTDetector: TensorRT vision tower + PyTorch heads
      owlvit_postprocess.py              # score/box arrays -> Detection (shared by both OWL-ViT backends)
      tensorrt_engine.py                  # TensorRTEngine: load a .engine, run inference
      image_codec.py                       # sensor_msgs/Image -> numpy, no cv_bridge dependency
      power_mode.py                         # nvpmodel -q query/parse (used by the node's launch file)
    launch/vision_bridge.launch.py
    test/                          # pure-Python unit tests (see "Testing" below)
    package.xml, setup.py, setup.cfg, resource/vision_bridge
ros2_ws/scripts/
  convert_owlvit_tensorrt.py    # ONNX export + TensorRT engine build -- run ON the Jetson (see below)
  compare_owlvit_yolov8.py      # OWL-ViT vs YOLOv8 benchmark -- run ON the Jetson, 15W (see below)
  map_metric.py                  # pure mAP@IoU / AP math (Pascal VOC-style), no dependencies
  test_set.py                     # loads a YOLO-style test-set (images/, labels/, classes.txt)
  resource_monitor.py              # tegrastats system-RAM + psutil process-RSS monitors
  owlvit_multi.py                   # OwlViTMultiClassDetector: multi-query OWL-ViT for the benchmark
  yolov8_detector.py                # YOLOv8Detector: thin ultralytics.YOLO wrapper
  test_map_metric.py, test_test_set.py, test_resource_monitor.py   # unit tests for the above
```

**Why two packages:** custom `.msg` interfaces require the
`rosidl_generate_interfaces()` CMake macro, which only exists in
`ament_cmake` packages — it's not available from a pure `ament_python`
package's `setup.py`. `vision_bridge_msgs` (messages only) and
`vision_bridge` (the Python node, depending on `vision_bridge_msgs`) is the
standard ROS2 split for this reason.

## Dependencies (installing on the Jetson)

None of this workspace's Python dependencies are pip-installable the normal
way on the Jetson Orin Nano — see `course/VENV.md` for the full explanation
of why (short version: JetPack ships CUDA/cuDNN/TensorRT/PyTorch as system
packages tied to the JetPack version; a plain `pip install torch` either
fails or silently installs a CPU-only build with no GPU).

```bash
# 1. ROS2 itself -- via apt/rosdep, never pip (JetPack 6.x officially
#    supports ROS2 Humble/Jazzy -- see week 03's Perplexity research
#    prompts, course/weeks/week03-.../prompts/perplexity.md, for the
#    current recommendation and the exact apt package names).

# 2. A venv that INHERITS the system JetPack packages -- must use
#    --system-site-packages, or torch.cuda.is_available() silently
#    comes back False:
python3 -m venv --system-site-packages ~/.venv-jetson
source ~/.venv-jetson/bin/activate

# 3. Everything else this workspace needs (numpy, pillow, psutil,
#    transformers, ultralytics -- ordinary pip packages that layer fine
#    on top of the system torch/tensorrt):
pip install -r ../course/requirements/requirements-jetson.txt

# 4. Verify the venv actually sees the GPU before running anything below:
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

`torch`/`tensorrt`/`pycuda` themselves are **not** in
`requirements-jetson.txt` — they come from the system JetPack install (step
2's `--system-site-packages`), not pip; see `course/VENV.md` section 3 for
where to get a matching `torch`/`torchvision` wheel if JetPack's own isn't
new enough for a given `transformers`/`ultralytics` version.

| Script | Needs beyond ROS2/system JetPack |
|---|---|
| `vision_bridge_node.py` (`pytorch`/`auto` backend) | `torch`, `transformers`, `pillow>=10.0` (Ubuntu's apt `python3-pil` predates `Image.Resampling`, see below) |
| `vision_bridge_node.py` (`tensorrt`/`auto` backend) | + `tensorrt`, `pycuda` (system) |
| `scripts/convert_owlvit_tensorrt.py` | `torch`, `transformers`, `tensorrt` (system) |
| `scripts/compare_owlvit_yolov8.py` | `torch`, `transformers`, `ultralytics`, `psutil`, `pillow`, `numpy` |

## Building and running (on a machine with ROS2 installed)

```bash
# JetPack 6.x officially supports ROS2 Humble/Jazzy -- see week 03's
# Perplexity research prompts (course/weeks/week03-.../prompts/perplexity.md)
# for the current recommendation.
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

ros2 launch vision_bridge vision_bridge.launch.py \
    image_topic:=/camera/image_raw \
    text_query:="a photo of a defect"
    # add check_power_mode:=false to skip the nvpmodel check off-device
```

**On a plain dev machine (no JetPack), not just Jetson:** `colcon
build`/`ros2 run`/`ros2 launch` all run under ROS2's own system Python
(e.g. `/opt/ros/humble`'s Python 3.10 on Ubuntu 22.04 -- confirmed by a
real run: the traceback for a missing dependency showed
`/opt/ros/humble/local/lib/python3.10/dist-packages`), which is a
**different Python than this repo's main `.venv`** (`requirements.txt`
targets a newer, unrelated Python version and was never meant to be
importable from inside `rclpy`). `detector_backend:=stub` needs
nothing beyond ROS2 itself and works out of the box (see "Detection
backend" below); `pytorch`/`auto` (the default) needs `transformers`
(and `torch`, pulled in by it) importable from *that* ROS2 Python, or
`vision_bridge_node` crashes with `ModuleNotFoundError: No module
named 'transformers'` as soon as it tries to build the PyTorch
backend. Two ways to get there, same idea as the Jetson
`--system-site-packages` venv above but without any JetPack-specific
constraint:

```bash
# Option A: install straight into ROS2's system Python (simplest for a
# one-off dev-machine check):
pip3 install --user transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

# Option B: a venv that still sees rclpy (system site-packages) plus
# transformers/torch layered on top -- closer to the Jetson recipe,
# better if you don't want to touch system Python at all:
python3 -m venv --system-site-packages ~/.venv-ros2-dev
source ~/.venv-ros2-dev/bin/activate
pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
# then colcon build / ros2 run / ros2 launch from inside this activated venv
```

**Use `--extra-index-url`, not `--index-url`** (confirmed by a real
failure: `--index-url` *replaces* the default PyPI index outright, and
`download.pytorch.org/whl/cpu` only hosts `torch`/`torchvision`/
`torchaudio` -- `pip install transformers torch --index-url
https://download.pytorch.org/whl/cpu` fails with `ERROR: Could not
find a version that satisfies the requirement transformers (from
versions: none)` because `transformers` was never going to be there.
`--extra-index-url` *adds* that index on top of PyPI instead, so
`torch` still resolves to the CPU wheel while `transformers` resolves
normally from PyPI -- the same reasoning `requirements.txt`'s own
`--extra-index-url` line at the top of this repo already follows.

**Do not `pip install transformers`/`torch` into this repo's main
`.venv` and expect `ros2 launch` to see them** -- confirmed by a real
failure: that plain venv (no `--system-site-packages`) isn't the
Python `vision_bridge_node` actually runs under even while it's
active in your shell. `colcon`'s generated console-script wrapper
(`install/vision_bridge/lib/vision_bridge/vision_bridge_node`) is an
ordinary `setuptools` entry point, whose `#!` shebang line hard-codes
the *absolute path* of whatever Python built it -- baked in once, at
build time, not re-resolved later. Activating a different venv
afterwards changes what a bare `python3`/`pip` resolves to on `PATH`;
it does **not** rewrite that already-generated file's shebang, and
`ros2 launch` executes it directly. A `colcon build` that reports
finishing in ~2s (an incremental no-op, not a real rebuild) will keep
the old shebang indefinitely. If you switch which Python you're
installing dependencies into (Option A vs B above, or just activating
a venv for the first time on an already-built workspace), force a
real rebuild first:

```bash
cd ros2_ws
rm -rf build install log    # colcon's own generated output -- safe to delete
colcon build --symlink-install
```

**`pip3 install --user transformers torch ...` into system Python can
hit `AssertionError` in pip's own dependency resolver** (confirmed by
a real failure: `pip/_internal/resolution/resolvelib/resolver.py`'s
`get_topological_weights`, `assert len(weights) ==
expected_node_count`) -- a known pip bug, not anything in this repo,
triggered by Ubuntu's apt-installed `pip3` being old enough to carry
it, combined with a pre-existing `--user`-installed package (e.g. an
earlier partial `torch` install) already in the resolution graph.
Fix: upgrade pip itself before retrying:

```bash
python3 -m pip install --user --upgrade pip
pip3 install --user transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
```

**`transformers` importing can still fail afterwards with
`AttributeError: module 'PIL.Image' has no attribute 'Resampling'`**
(confirmed by a real failure, traceback rooted in
`transformers/image_utils.py`) -- Ubuntu ships an `apt`-installed
`python3-pil` (e.g. 9.0.1 on 22.04) that predates Pillow 9.1.0's
`Image.Resampling` enum, and `pip install --user transformers` doesn't
touch it: `transformers` only declares an unversioned `Pillow`
dependency, so pip's resolver treats the already-installed system one
as satisfying it and leaves it alone. Fix is the same
`requirements.txt` already pins (`pillow>=10.0`), just applied to
`--user` site explicitly so it actually overrides the system one:

```bash
pip3 install --user --upgrade "pillow>=10.0" --extra-index-url https://download.pytorch.org/whl/cpu
python3 -c "from PIL import Image; print(Image.__file__)"   # should now print a path under ~/.local, not /usr/lib/python3/dist-packages
```

## Detection backend

`VisionBridgeNode` takes a `Detector` (see `detection.py`) rather than
hardcoding a model, built by `detector_factory.build_detector()` from the
`detector_backend`/`engine_path`/`owlvit_model_name`/`score_threshold`
parameters:

- `auto` (default): use a TensorRT engine at `engine_path` if one is
  configured and present; otherwise, or if it fails to load, fall back to
  the PyTorch backend. A TensorRT load failure here is *expected*, not a
  bug — the near-universal cause is a `.engine` built for a different
  GPU/TensorRT version (see "Building the TensorRT engine" below), and
  `TensorRTEngineError` exists specifically to make that the fallback
  trigger rather than a crash.
- `tensorrt` — require a working engine; raise if unavailable.
- `pytorch` — force the eager `transformers` OWL-ViT backend.
- `stub` — always `StubDetector` (zero detections), for dev/testing
  without any model.

Only the OWL-ViT **vision tower** runs through TensorRT
(`owlvit_tensorrt.py`): it's the part that runs on every video frame. Text
encoding and the detection heads stay in `transformers`/PyTorch — cheap,
only rerun when the query changes, and running the model's own real
box/class-head code instead of a hand reimplementation of it. Both
backends funnel through the same `owlvit_postprocess.build_detections()`
and HuggingFace's own `post_process_object_detection()`, so results are
comparable regardless of which backend produced them.

### Building the TensorRT engine

`.engine` files are **not portable** across GPU architectures or TensorRT
versions — build on the exact device you'll run inference on:

```bash
# ON the Jetson Orin Nano itself, JetPack's own tensorrt (not pip):
cd ros2_ws
PYTHONPATH=src/vision_bridge python3 scripts/convert_owlvit_tensorrt.py \
    --onnx-path models/owlvit_vision.onnx \
    --engine-path models/owlvit_vision.engine
```

The script checks `/etc/nv_tegra_release` (the JetPack marker file) and
refuses to run elsewhere by default (`--force` overrides this, but the
resulting engine will only be useful for testing the ONNX-export step, not
for actually loading anywhere).

**Not verified in this sandbox**: no GPU/CUDA/TensorRT here, and
`huggingface.co` is network-blocked (see `course/DECISIONS.md`, ADR-004),
so neither the conversion script nor the TensorRT runtime path
(`tensorrt_engine.py`, `owlvit_tensorrt.py`) has actually been executed.
Both were written against documented TensorRT 8.5+/10.x Python API usage
and OWL-ViT's public `vision_model(pixel_values) -> .last_hidden_state`
contract — reviewed carefully, not run. Verify on the actual Jetson before
relying on them; see the module docstrings for exactly what to check.

## Executor choice

`main()` spins the node on `rclpy.executors.SingleThreadedExecutor`, not
`MultiThreadedExecutor` — reasoning is in `vision_bridge_node.py`'s module
docstring (short version: the Orin Nano's 6 cores are shared with the rest
of the ROS2 graph, the only expensive callback is inherently serial behind
the shared inference accelerator anyway, and the text-query callback is
cheap/infrequent enough that briefly queuing behind an image callback is a
non-issue).

## Power-mode check

`vision_bridge.launch.py` runs `nvpmodel -q` before starting the node and
logs a warning (does not block launch) if the current mode isn't
`15W`/`MAXN_SUPER` — `7W` caps GPU/DLA clocks and may not sustain stable
detection FPS under continuous load. Off-device, `nvpmodel` doesn't exist
and the check is a silent no-op; pass `check_power_mode:=false` to skip it
explicitly.

## OWL-ViT vs YOLOv8 comparison

`scripts/compare_owlvit_yolov8.py` benchmarks the open-vocabulary OWL-ViT
backend against closed-set YOLOv8 on one test-set, and writes a Markdown
comparison table:

```bash
# ON the Jetson Orin Nano, in 15W mode:
cd ros2_ws/scripts
python3 compare_owlvit_yolov8.py \
    --test-set path/to/test_set \
    --yolo-model yolov8n.pt \
    --owlvit-model google/owlvit-base-patch32 \
    --output outputs/report.md
```

The test-set is YOLO-format (`images/*.jpg`, `labels/*.txt` with normalized
`class_id cx cy w h` lines, `classes.txt`); `test_set.py` loads it into
pixel-space `GroundTruth`s using only `PIL` for image dimensions. For a
meaningful YOLOv8 number, `classes.txt` should use the same class-name
strings the loaded YOLOv8 checkpoint predicts, since `map_metric.py`
matches predictions to ground truths by class-name equality, not index.

**15W is a hard gate, not a warning.** Unlike the launch file's soft
power-mode check (logs and continues), this script `sys.exit(1)`s if
`nvpmodel -q` doesn't report `15W` — `--force` overrides it. The launch
file only needs "detection won't be unreasonably slow"; this script exists
specifically to produce mAP/FPS/memory numbers that stay comparable across
runs, and `MAXN_SUPER`'s burst clocks or `7W`'s throttled floor would both
silently invalidate that comparison.

**Two distinct memory metrics, not one.** The task asks for "peak process
memory (via tegrastats or jtop)", but `tegrastats` actually reports
system-wide RAM, not a single process's — so `resource_monitor.py`
implements both, and the report table keeps them in separate columns
rather than conflating them: `TegrastatsRamMonitor` parses `RAM
used/totalMB` out of the `tegrastats` subprocess's stdout (peak system
RAM), while `PeakRssTracker` samples `psutil.Process().memory_info().rss`
after each image (peak process RSS). Neither substitutes for the other —
system RAM includes everything else running on the device, and process
RSS misses e.g. GPU/unified memory `tegrastats` also reports.

**mAP@0.5** is `map_metric.py`'s `mean_average_precision()`: standard
Pascal VOC 2010+-style per-class AP (greedy highest-IoU matching per
image, monotonic precision envelope, area-under-the-envelope integration
by recall step), averaged across classes present in the ground truth. It
depends on nothing beyond stdlib and is fully unit-tested (see "Testing").

**Not runnable in this sandbox**: no GPU/Jetson/`tegrastats`, and no
network access to download OWL-ViT or YOLOv8 weights (`course/DECISIONS.md`
ADR-004). `map_metric.py` and `test_set.py` are pure and unit-tested
without any of that. `owlvit_multi.py`/`yolov8_detector.py`/
`compare_owlvit_yolov8.py`'s model-running and monitoring glue has been
reviewed carefully and smoke-tested up to the exact expected import
boundary (`ModuleNotFoundError: No module named 'torch'` after `--force`
clears the power-mode gate and the test-set loads) but not actually run
against real models or hardware — verify on the Jetson before trusting the
numbers it produces.

## Testing

This sandbox has no ROS2/colcon installed (a full ROS2 install is a large,
system-level apt dependency, impractical here — same kind of environment
constraint as `torch`/`huggingface.co` access documented in
`course/DECISIONS.md` for weeks 01-02). So:

- `power_mode.py`, `detection.py`, `image_codec.py`, `owlvit_postprocess.py`,
  and `tensorrt_engine.py`/`detector_factory.py`'s *decision logic*
  deliberately don't import `rclpy`, `torch`, `transformers`, or
  `tensorrt`/`pycuda` at module level (all lazily imported inside the
  functions/constructors that actually need them) — they're unit-tested
  directly with `pytest`, no ROS2/CUDA/TensorRT/torch needed:

  ```bash
  cd ros2_ws/src/vision_bridge
  PYTHONPATH=. pytest test/    # 27 passed
  ```

  Several `detector_factory`/`tensorrt_engine` tests specifically lean on
  this sandbox lacking tensorrt/pycuda/torch: they assert `build_detector`
  reaches the *expected* environment boundary (`FileNotFoundError` for a
  missing engine, `TensorRTEngineError`/`ModuleNotFoundError` for missing
  tensorrt/torch) for each backend and fallback path, proving the
  auto-fallback branching itself is exercised correctly even though no
  actual model ever loads here.

- `vision_bridge_node.py`, `owlvit_pytorch.py`'s/`owlvit_tensorrt.py`'s
  actual model-loading code paths, and the TensorRT builder/runtime need a
  real ROS2 install (+ GPU/CUDA/TensorRT for the latter) to import/run/test
  — not verified in this session. They're thin layers over the
  already-tested pure logic above, reviewed carefully but not executed
  here; see "Building the TensorRT engine" above for the specific caveat
  on the OWL-ViT TensorRT path.

- `scripts/`'s pure logic (`map_metric.py`, `test_set.py`,
  `resource_monitor.py`'s `parse_tegrastats_ram_mb`) is unit-tested the
  same way, independently of `ros2_ws/src/vision_bridge/test/` (a separate
  `pytest` root, no shared imports):

  ```bash
  cd ros2_ws/scripts
  pytest    # 23 passed
  ```

  `compare_owlvit_yolov8.py`'s orchestration (power-mode gate, test-set
  loading, backend construction) is reviewed and smoke-tested to its exact
  expected environment boundary, per the "Not runnable in this sandbox"
  note above, rather than covered by these unit tests.
