# Runbook — Jetson Orin Nano incidents (week 10, day 4)

Five incident types specific to running this repo's VLM defect-detection
service on a Jetson Orin Nano, with concrete recovery steps against this
repo's real scripts, endpoints, and config keys — not generic advice.
Each section states plainly what this repo can and can't detect
automatically; several gaps below are real and not yet closed (see each
section's "Gap" note).

Conventions used throughout:
- Commands marked **[Jetson]** must run on the device itself (they're
  gated by `require_jetson()`/`/etc/nv_tegra_release` checks, or are
  meaningless off-device).
- `$REPO` = the repo root on the device (wherever `deploy_to_jetson.sh`
  or a manual checkout put it).

---

## 1. Camera / video source loss (RTSP stream or local camera feed)

**Symptom:** `POST /v1/embed/rtsp` returns errors, or the ROS2
`vision_bridge` node stops publishing detections.

**What this repo does and doesn't detect:** there is no camera
reconnect/retry logic anywhere in this repo, and `GET /health` only
reports that the FastAPI *process* is alive — it says nothing about
whether any camera or RTSP stream is reachable (`src/api/app.py`'s
`/v1/embed/rtsp` only touches the stream per-request, not continuously).
"Service healthy" does **not** mean "camera reachable."

**Two independent code paths, diagnosed differently:**

### 1a. FastAPI service (`POST /v1/embed/rtsp`)

`src/data/video.py::open_video_capture()` wraps `cv2.VideoCapture` and
raises `VideoSourceError("Could not open video source: ...")` if
`capture.isOpened()` is `False`; `src/api/app.py` catches this and
returns **HTTP 503** with that message (a second 503 case: the stream
opened but yielded zero frames — `"No frames could be read from the RTSP
stream."`).

1. Reproduce and confirm it's really the camera, not the API:
   ```bash
   curl -X POST http://localhost:8000/v1/embed/rtsp \
     -H "Content-Type: application/json" \
     -d '{"url": "rtsp://<camera-ip>:554/stream1", "max_frames": 8, "frame_stride": 1}'
   ```
   A `503` with `"Could not open video source"` confirms the container
   genuinely can't reach the stream — not an application bug.
2. Check the stream from the Jetson host directly (outside the
   container), since a container network/firewall issue looks identical
   to a dead camera from inside the app:
   ```bash
   ffprobe -v error rtsp://<camera-ip>:554/stream1   # or: ffplay, vlc --intf dummy
   ping <camera-ip>
   ```
3. If the camera itself is unreachable: power-cycle it, check its own
   network link/PoE, confirm its RTSP URL/credentials haven't changed
   (the URL is caller-supplied per-request in this API — there is no
   stored/default camera URL in `config/model_config*.yaml` to check).
4. Once the stream answers again, retry the same `curl` above — there's
   no service restart needed, since the FastAPI process never held the
   camera open between requests.

### 1b. ROS2 `vision_bridge` node

`ros2_ws/src/vision_bridge/vision_bridge/vision_bridge_node.py` does not
open the camera itself — it subscribes to a ROS2 topic (default
`/camera/image_raw`, a `sensor_msgs/Image`) published by a separate
camera-driver node (e.g. a RealSense D435i driver per `course/HARDWARE.md`
— that driver is not part of this repo). If the publisher dies, callbacks
just stop; nothing in `vision_bridge_node.py` detects "no frames
arriving."

1. Confirm frames are actually arriving and at what rate: **[Jetson]**
   ```bash
   ros2 topic hz /camera/image_raw
   ros2 topic info /camera/image_raw   # confirms a publisher exists at all
   ros2 node list
   ```
2. If `ros2 topic hz` reports nothing: the upstream camera driver node
   has died or was never launched — restart it (its own process/launch
   file, outside this repo).
3. If the topic is publishing but `vision_bridge` still isn't producing
   detections: restart the bridge itself:
   ```bash
   ros2 launch vision_bridge vision_bridge.launch.py \
     image_topic:=/camera/image_raw text_query:="a photo of a defect"
   ```

**Gap:** neither path has automated alerting on camera loss — this is a
documentation-only mitigation (know what to run), not a monitored signal.
A future improvement would be a periodic `/v1/embed/rtsp` synthetic
health check against a known-good stream, wired into the Prometheus stack
in `monitoring/`.

---

## 2. Model quality degradation (embeddings drift, wrong predictions)

**Symptom:** the service still runs (no crashes, no 5xx spike), but
predictions are visibly wrong, or a scheduled quality check fails.

**What this repo does and doesn't detect:** `src/api/metrics.py`'s
`MetricsCollector`/`render_prometheus` track only latency, throughput,
status codes, and Jetson temperature — nothing about embedding quality or
prediction distribution. **There is no automated drift alarm in this
repo.** Degradation is caught either by a human noticing wrong
predictions, or by deliberately re-running the quality gates below on a
schedule.

### 2a. Embedding quality gate — FP16 vs quantized (INT4/INT8)

`quantization/scripts/07_evaluate_quality.py` compares per-image cosine
similarity between the FP16 reference `.engine` and the deployed
quantized `.engine`'s image embeddings on a fixed, deterministic test set
(`quantization/config/config.yaml`'s `evaluation.test_set_dir`,
`evaluation.num_samples: 64`), gated at
`evaluation.min_cosine_similarity: 0.98`.

```bash
cd $REPO/quantization
make evaluate-quality              # [Jetson] add MODE=1 to fix nvpmodel to 15W first
# writes outputs/quality/quality_report.json
```

Prints: `Quality gate {PASSED|FAILED}: mean cosine similarity {X:.4f}
(threshold {threshold}), {N}/{total} images below threshold.` Inspect
`outputs/quality/quality_report.json`'s `per_image` field to find which
specific images regressed.

**If FAILED:** the quantized engine has drifted from its own FP16
reference — this is a quantization-artifact problem, not a model
problem. Rebuild both engines fresh (see §4's rebuild sequence — the same
steps apply whether the trigger was a JetPack upgrade or a quality-gate
failure) and re-run this gate.

**If the FP16 reference itself looks wrong** (predictions are bad even
before comparing to the quantized engine): that's genuine upstream model
drift, not a quantization artifact. This repo has no automated
re-training/fine-tuning tooling for that scenario — re-validate
`config/model_config.yaml`'s `model.name`
(`openai/clip-vit-base-patch32`) against a known-good baseline
(`outputs/weekNN/` demo artifacts from earlier weeks are a fixed
reference point), and escalate to a human decision about re-training or
re-selecting the checkpoint.

### 2b. Defect-rate quality gate (pipeline)

Separate from 2a: `pipeline/scripts/06_quality_gate.py` checks the
*defect detection rate* (not embedding similarity) against
`pipeline/config/*.yaml`'s `quality_gate.green_max_defect_rate`/
`yellow_max_defect_rate`, writing a GREEN/YELLOW/RED verdict to
`outputs/metrics/quality_gate.json`. This gate is CI-wired
(`.github/workflows/ci.yml` fails the build on RED).

```bash
cd $REPO/pipeline
./run_all.sh --config config/pipeline_config.yaml   # real run, all 7 stages
cat outputs/metrics/quality_gate.json                # inspect the verdict
```

A RED verdict here means the detection pipeline's defect rate has moved
outside the expected band — check `outputs/interpretations.json` (stage
5's output) for which images/classes are driving it before assuming the
model itself is at fault (a dataset/labeling shift can produce the same
symptom).

**Gap:** no scheduled/cron trigger exists for either gate in this repo —
both are manual (or CI-on-push, for 2b only) commands. Running 2a
periodically on-device is the realistic way to catch quantized-model
drift before a human notices bad predictions.

---

## 3. Disk overflow on microSD/NVMe

**Symptom:** writes start failing (model download, engine build,
inference logs), or `backup_data.sh` itself warns about low space.

Per `course/HARDWARE.md`: an NVMe SSD is recommended over microSD alone —
weeks 5-6 already generate tens/hundreds of MB of synthetic data copied
to the device for inference tests, and microSD becomes an I/O bottleneck
under that load. This section assumes both are possible: a microSD-only
device that's simply full, or an NVMe-equipped device where the wrong
device filled up.

1. **Confirm and locate** — check every mounted filesystem, not just `/`:
   ```bash
   df -h
   ```
2. **Find what's actually eating space** — this repo's own biggest
   consumers, per `backup/backup_data.sh`'s default backup paths:
   ```bash
   du -sh .cache/models outputs/onnx outputs/engine outputs/calibration \
         quantization/outputs/onnx quantization/outputs/engine \
         quantization/outputs/calibration 2>/dev/null
   docker system df   # image/layer/volume disk usage, if running containerized
   ```
   `.cache/models` (the HuggingFace CLIP checkpoint cache,
   `config/model_config*.yaml`'s `model.cache_dir`) and TensorRT `.engine`
   files are typically the largest single items.
3. **Reclaim space from regeneratable artifacts** — everything below is
   a build product, not source (confirmed by `.gitignore`'s own
   `outputs/*`, `*.onnx`, `*.engine`, `*.pt` exclusions): it's always safe
   to delete and regenerate via the relevant week's scripts if truly
   desperate for space and no backup exists yet.
   ```bash
   docker system prune          # unused images/containers/build cache
   docker volume prune          # ONLY if you're sure model-cache isn't needed live
   rm -rf quantization/outputs/calibration_data   # copied input images, cheap to regenerate from dataset_pipeline/
   ```
4. **Prefer restoring from backup over deleting-and-regenerating** when a
   backup exists — regenerating `.engine` files means a full TensorRT
   build (§4's rebuild sequence, minutes to tens of minutes on-device),
   while a restore is a plain file copy:
   ```bash
   ./backup/backup_data.sh --backup-root /mnt/nvme/backups --keep 14   # check first: warns if <512 MiB free at the destination
   ./backup/restore.sh --backup-root /mnt/nvme/backups --restore-root $REPO --force
   ```
5. **Move the backup destination to the faster device** if backups
   themselves are on the same slow microSD being protected — `--backup-root`
   should point at NVMe when the device has one, per `backup/README.md`'s
   own guidance (backing up onto the same slow medium still writes every
   changed byte twice to it).
6. **After freeing space**, re-run whatever failed (a `docker compose up
   -d` that couldn't write a layer, a `05_build_engine.py` that couldn't
   serialize, etc.) — nothing in this repo auto-retries a disk-full
   failure.

**Gap:** no log rotation exists in this repo. `src/api/logging_config.py`
writes structured JSON to **stdout only** (for a container log collector,
not a file) — a fresh checkout has no `logs/` directory at all. If an
operator redirects `docker compose logs` to a file per `backup/README.md`'s
suggested cron pattern, that file grows unbounded with nothing in-repo to
rotate or truncate it; pair any such redirect with `logrotate` or an
equivalent, not left to this repo's own tooling. Also: Prometheus's
`node-exporter` (`monitoring/jetson/docker-compose.exporters.yml`, port
`:9100`) already exposes `node_filesystem_avail_bytes` for
Grafana-side alerting on disk pressure before it becomes an incident, but
`monitoring/server/grafana/dashboards/vlm-inference-overview.json` has no
panel for it yet — worth adding rather than only reacting to a full disk
after the fact.

---

## 4. TensorRT engine failure after a JetPack upgrade

**Symptom:** the quantized/FP16 `.engine` fails to load or produces
garbage output after the device's JetPack/L4T version changed.

**Root cause, documented in-repo:** TensorRT engines are **not portable**
across GPU architecture or TensorRT version — `course/HARDWARE.md` states
outright that changing JetPack mid-course means rebuilding every
`.engine` file, and `Dockerfile.jetson`'s header comment is explicit that
its `l4t-pytorch:r36.2.0` base tag (JetPack 6.0) "MUST match the device's
actually-flashed L4T version." An engine built under one JetPack/TensorRT
version is expected to fail — or silently misbehave — under another.

1. **Confirm the JetPack/L4T version actually on the device:** **[Jetson]**
   ```bash
   cat /etc/nv_tegra_release
   ```
   This is the same marker file `quantization/utils/environment.py`'s
   `require_jetson()`/`is_jetson()` checks before letting any
   TensorRT-touching script run at all — if this file is missing or
   reports an unexpected release, that alone explains an engine failure.
2. **Rebuild the plain FP16 reference engine first** — fastest smoke test
   that TensorRT itself works post-upgrade, skips the INT4/INT8
   calibration cache entirely: **[Jetson]**
   ```bash
   cd $REPO/quantization
   make build-engine-fp16
   ```
   A `RuntimeError("TensorRT engine build failed (build_serialized_network
   returned None) -- check the builder log above for the underlying
   error.")` here points at a genuine TensorRT/driver incompatibility, not
   a calibration problem.
3. **Rebuild the quantized engine** — re-run calibration too, since a
   stale calibration cache from before the upgrade may not be valid
   either: **[Jetson]**
   ```bash
   make quantize        # scripts/04_quantize_model.py -- regenerates outputs/calibration/int4_calibration.cache
   make build-engine     # scripts/05_build_engine.py
   ```
4. **Re-validate quality against the freshly-built FP16 baseline** (not
   the old one — §2a's gate compares against whatever FP16 engine
   currently exists on disk):
   ```bash
   make evaluate-quality
   ```
5. **If `vision_bridge`'s OWL-ViT detector is also affected**
   (`ros2_ws/`, a separate TensorRT engine from the CLIP path above):
   note that `ros2_ws/src/vision_bridge/vision_bridge/detector_factory.py`'s
   `auto` backend is *designed* to fall back to the PyTorch OWL-ViT
   backend automatically on a `TensorRTEngineError` — a JetPack upgrade
   breaking this engine is degraded, not down, if `auto` mode is in use.
   Rebuild it the same way: **[Jetson]**
   ```bash
   PYTHONPATH=src/vision_bridge python3 scripts/convert_owlvit_tensorrt.py \
     --onnx-path models/owlvit_vision.onnx --engine-path models/owlvit_vision.engine
   ```
6. **Redeploy the full service** once engines are confirmed good, so the
   running container picks up the new files and its base image matches
   the new JetPack (if `Dockerfile.jetson`'s base tag also needed
   updating for the new release):
   ```bash
   ./scripts/deploy_to_jetson.sh --host user@jetson-ip
   ```

Every command above except step 6 has a `--force`/`FORCE=1` escape hatch
to skip the `/etc/nv_tegra_release` check for a dev-machine smoke test of
the script's argument handling — never use it for a real fix; an engine
built off-device (or under `--force` while genuinely on a mismatched
JetPack) is not a valid deployment artifact.

---

## 5. Thermal throttling (fan/heatsink failure)

**Symptom:** inference latency climbs, `vlm_jetson_thermal_warning`
metric flips to 1, or the device is audibly/visibly running hot with a
stalled fan.

**Detection — two independent, overlapping sources:**

1. **In-process** (`src/api/metrics.py`'s `TegrastatsSampler`, running
   inside the `vlm-inference` container for the life of the FastAPI
   process):
   ```bash
   curl -s http://localhost:8000/metrics.json | jq .tegrastats
   # latest_temps_c, max_temps_c (peak since process start), thermal_warning, throttle_temp_c
   curl -s http://localhost:8000/metrics | grep vlm_jetson
   # vlm_jetson_temperature_celsius{sensor="..."}
   # vlm_jetson_temperature_max_celsius{sensor="..."}
   # vlm_jetson_thermal_warning        (1/0)
   # vlm_jetson_tegrastats_available   (1/0 -- whether tegrastats itself was even found)
   ```
2. **Standalone exporter** (`monitoring/jetson_exporter/exporter.py`,
   `:9835` — tails a `tegrastats --logfile` written by the privileged
   `tegrastats-writer` sidecar, independent of the inference process):
   ```bash
   curl -s http://localhost:9835/metrics | grep jetson_
   ```
   Includes `jetson_tegrastats_stale` (1/0) — if this is `1`, the
   `tegrastats-writer` container itself has died, which is a *different*
   incident (monitoring outage) from real thermal throttling; check
   `docker logs tegrastats-writer` first before assuming a fan failure.
3. **Grafana**: the "Jetson Temperature (°C)" panel in
   `monitoring/server/grafana/dashboards/vlm-inference-overview.json`
   overlays both sources above, colored orange at 75°C / red at 85°C.

Both `vlm_jetson_thermal_warning` and the dashboard's 85°C threshold are
sourced from `THROTTLE_TEMP_C = 85.0` in `src/api/metrics.py` — its own
docstring is explicit that this is a **conservative software proxy, not
a confirmed hardware trip point**. For the real board-specific numbers:
```bash
cat /sys/devices/virtual/thermal/thermal_zone*/trip_point_*_temp   # [Jetson]
```
Treat `thermal_warning=True` as "worth investigating directly via
`tegrastats`/`jetson_clocks`," not as an authoritative throttle event on
its own.

**Recovery steps, in order:**

1. **Physically inspect the fan/heatsink** — this is a hardware failure
   this repo's software can only detect the symptom of, not fix. Confirm
   the fan spins, the heatsink isn't caked with dust, and thermal paste
   hasn't degraded (long-running 24/7 deployments especially).
2. **Immediately reduce thermal load** by dropping the power mode — the
   fastest software mitigation while the hardware issue is being fixed:
   ```bash
   nvpmodel -q          # [Jetson] confirm current mode first
   sudo nvpmodel -m 2    # [Jetson] 7W -- lowest power/heat (NVPMODEL_MODES: 0=MAXN_SUPER, 1=15W, 2=7W)
   ```
   `quantization/utils/power_mode.py::set_nvpmodel(mode_id,
   settle_seconds)` is the programmatic equivalent used by
   `06_trt_inference.py`/`07_evaluate_quality.py --nvpmodel-mode 2` if
   running a benchmark/quality check *during* the incident and you want
   the reduced-power reading recorded.
3. **Check clocks directly** for confirmation of active throttling (not
   automated anywhere in this repo — bare CLI):
   ```bash
   sudo tegrastats                # [Jetson] live temps/freqs
   sudo jetson_clocks --show      # [Jetson] current clock state vs. max
   ```
4. **Monitor recovery** via the same `/metrics.json`/`/metrics`/`:9835`
   endpoints above until `thermal_warning` clears and `max_temps_c`
   stabilizes below threshold.
5. **Once the hardware fix is confirmed** (fan replaced, heatsink
   reseated, dust cleared) and temperatures are stable, restore normal
   operating mode:
   ```bash
   sudo nvpmodel -m 1    # [Jetson] 15W -- this repo's standard working mode (course/HARDWARE.md)
   ```
   Only return to `-m 0` (MAXN Super) deliberately, for benchmarking —
   `ros2_ws/scripts/compare_owlvit_yolov8.py` hard-gates on exactly 15W
   (`sys.exit(1)` if `nvpmodel -q` doesn't report it, `--force` to
   override) specifically so benchmark numbers stay comparable across
   runs; don't leave a device at 7W or MAXN Super as its new steady
   state without updating that expectation.

**Gap:** no automated alert fires on `thermal_warning`/`jetson_tegrastats_stale`
today — Grafana's dashboard is dashboard-only (someone has to be looking
at it), no Alertmanager or equivalent is wired into
`monitoring/server/docker-compose.monitoring.yml`. A real 24/7 deployment
should add one rather than relying on a human noticing the panel turn
red.

---

## Honesty note

This runbook was written by tracing this repo's actual code (endpoints,
config keys, CLI flags, Makefile targets) rather than from generic
Jetson/MLOps advice, and every command above was verified to exist in the
codebase as written. It has **not** been exercised against a real
incident on real hardware — this sandbox has no camera, no Jetson, no
fan to fail, and no TensorRT/CUDA installed (the same standing limitation
documented throughout `course/DECISIONS.md` since week 08's ADR-028).
Several sections above call out concrete, real monitoring/automation gaps
(no camera health check, no scheduled drift detection, no log rotation,
no disk-usage dashboard panel, no thermal/staleness alerting) rather than
implying this runbook alone makes the deployment self-healing — it
doesn't; it tells a human what to run.
