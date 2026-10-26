# Final architecture — weeks 1-11 unified under `src/` (week 12)

Eight layers, one for each named in day 2's task: `vision/`, `nlp/`,
`robot/`, `database/`, `api/`, `scenarios/`, `logging/`, `config/`. Six
of the eight (`vision`/`nlp`/`robot`'s two week-11 bridges/`database`/
`logging`, plus `config`'s `environment.py` addition) are **composition,
not new code** — each is a thin package that re-exports an
already-built, already-tested implementation from the week that built
it, so nothing below is a rewrite or a second copy of eleven weeks of
work. `scenarios/` is the one genuinely new layer, and `robot/` gained
one genuinely new module of its own on day 3
(`manipulator.py` — SO-ARM 101 control, no prior week built this);
`api/` already existed as its own top-level package since week 08 and
didn't need a new one.

```
                         operator (chat text)
                                │
                                ▼
                        src/nlp  (assistant.llm_parser)
                    two-tier: local regex → Claude API
                                │
                                ▼
              assistant.action_executor.ActionExecutor
           (dispatches ONE parsed command; week 11 day 3)
                    │           │            │
                    ▼           ▼            ▼
              src/vision   src/robot    src/database
           (VisionBridge) (ROS2+OPC-UA)  (SQLite)
                    │           │            │
                    ▼           ▼            ▼
            src/api/app.py  physical world  outputs/*/detections.db
          (CLIP embed, wk08)  (or simulated)

                         ── separately ──

              src/scenarios  (base.py: day 2, concrete: day 3)
     ColorSortingScenario / DefectDetectionScenario -- a standalone
     watch→decide→act loop, one cycle per Scenario.run(context) call:
     image_source() → context.vision.classify_by_prompts() →
     context.manipulator.move_to()/close_gripper()/open_gripper()/home()
     → context.database.record() -- built on the SAME shared layers
     above (src.scenarios.build_default_context), not triggered by chat
     text, driven by scripts/run_scenario_demo.py instead.
     config/scenarios.yaml's dev/jetson camera source (synthetic frames
     vs a real device) is where "конфиг на окружения" actually changes
     behavior, resolved once via src.config.environment.

                  src/logging + src/config (+ environment.py)
        cross-cutting: every layer above uses these, not part of
        either request path.
```

## The eight layers, what each one is, and where it actually lives

| Layer | New in week 12? | What it re-exports (or is) | Real source |
|---|---|---|---|
| `src/vision/` | Composition | `VisionBridge` (HTTP to the deployed CLIP service + in-process OWL-ViT detector reuse; `classify_by_prompts`, day 3, generalizes `classify_defect` to an arbitrary class set), `CLIPLoader`/`CLIPInference`, `classify_by_cosine_similarity` | `assistant/vision_bridge.py` (wk11d3, extended wk12d3), `src/models/clip.py` (wk01/08), `src/models/zero_shot.py` (wk02) |
| `src/nlp/` | Composition | `LLMParser`, `match_locally`, `call_claude_api`, `ParsedCommand`, rate-limiter | `assistant/llm_parser.py`, `assistant/schemas.py`, `assistant/rate_limiter.py` (wk11d2) |
| `src/robot/` | Composition + one new module | `ROSInterface`, `OPCUABridge` (composition, wk11d3); `Manipulator`/`JointPosition` (**genuinely new**, wk12d3 — SO-ARM 101 control, see "The manipulator" below) | `assistant/ros_interface.py`, `assistant/opc_ua_bridge.py`; `src/robot/manipulator.py` (this day) |
| `src/database/` | Composition | `Database`, `DetectionRecord`, `DetectionLogProtocol` | `assistant/database.py`, `assistant/database_models.py` (wk11d4), `assistant/action_executor.py` (wk11d3) |
| `src/api/` | Already existed | `src/api/app.py` (detection/embedding HTTP API) — `assistant/chat_app.py` is the second, separate API surface (chat + WebSocket), deliberately NOT moved under `src/api/` (see "Why `assistant/chat_app.py` didn't move" below) | week 08, week 11 day 4 |
| `src/scenarios/` | **Genuinely new** | `Scenario` (ABC), `ScenarioContext`, `ScenarioResult`, `build_default_context` (day 2); `ColorSortingScenario`/`DefectDetectionScenario`, `config.py`'s env-aware loaders, `image_sources.py`'s dev/jetson frame sources (day 3) | `src/scenarios/base.py` (d2), `color_sorting.py`/`defect_detection.py`/`config.py`/`image_sources.py` (d3) |
| `src/logging/` | Composition | `configure_logging`, `JSONFormatter` | `src/api/logging_config.py` (week 08) |
| `src/config/` | Composition + one new module | `AppConfig`/`load_config`/`apply_target_overrides` unchanged (moved from `src/config.py` into `src/config/__init__.py`, same public API, zero call sites changed); `environment.py` (`Environment`, `resolve_environment`) is new | `src/config.py` → `src/config/__init__.py` (week 01), `src/config/environment.py` (day 2) |

## The manipulator: SO-ARM 101 control (day 3)

`src/robot/manipulator.py` — genuinely new code, unlike everything else
`src/robot/` re-exports: no prior week built manipulator control at all.
Real facts about the hardware (not guessed): six Feetech STS3215 servos
(`shoulder_pan`=1, `shoulder_lift`=2, `elbow_flex`=3, `wrist_flex`=4,
`wrist_roll`=5, `gripper`=6) daisy-chained on one USB-serial bus, per
LeRobot's own published SO-101/SO-ARM101 documentation.

- `Manipulator.move_to`/`open_gripper`/`close_gripper`/`home` — the
  public interface `scenarios/` calls, same config-driven
  `simulate: true|false` split as `ROSInterface`/`OPCUABridge` (week 11):
  `_SimulatedManipulatorBackend` is real, independently-testable
  in-memory arm state (not a mock); `_SerialManipulatorBackend`
  frames real Feetech SCServo (protocol 0) packets over `pyserial`
  (lazy-imported — not installed in this sandbox, verified only to the
  exact `ModuleNotFoundError` boundary, same pattern as `rclpy`/`asyncua`).
- Real motion requires BOTH `config.simulate == False` AND
  `config.enabled == True` — mirrors `OPCUABridge.write_tag`'s
  `read_only`/`allow_write` double opt-in (ADR-043): a deployment must
  explicitly arm the manipulator before any real hardware moves.
- **Honesty note**: `config/manipulator.yaml`'s `register_map`
  (`torque_enable`/`goal_position`/`present_position` addresses) are the
  commonly-published Feetech STS/SMS control-table addresses — NOT
  independently re-verified against a physical SO-ARM 101's firmware in
  this sandbox (no serial port, no device). Confirm before a first real
  run; override in config if this unit's table differs.

## The two scenarios (day 3)

`ColorSortingScenario`/`DefectDetectionScenario` (`src/scenarios/`) each
implement one `run(context)` cycle: pull a frame from `image_source`,
`context.vision.classify_by_prompts(...)` against an env-independent
class set (colors; `src.nlp.DEFECT_TYPES` + `"ok"`), then drive
`context.manipulator` through pick → place → home on a confident result,
logging to `context.database` via the same `DetectionRecord` shape week
11 already established. Both return a `ScenarioResult` — never raise —
for every failure mode (no frame, vision service down, an uncertain
classification below `margin_threshold`, a manipulator failure
mid-sequence).

`config/scenarios.yaml`'s `camera:` section is where "конфиг на
окружения (dev/prod=jetson)" is concrete, not just a label: `dev`
resolves to a deterministic synthetic frame generator
(`image_sources.py`, no camera needed — colored noise textures for
`color_sorting`, reusing week 05's grayscale placeholder generator for
`defect_detection`); `jetson` resolves to a real device path/RTSP URL
`src.data.video.open_video_capture` opens for real on-device. Resolved
once via `src.config.environment.Environment` at config-load time, not
branched on inside either scenario.

`scripts/run_scenario_demo.py` (`make scenario-color-sorting`/
`scenario-defect-detection`) drives N cycles against a real
`ScenarioContext`. Run for real in this sandbox against the actually
deployed `src/api/app.py`: a real HTTP round trip reaching a real `500`
(no `torch` to load CLIP weights) — the same honest boundary
`assistant/vision_bridge.py` already established, now proven reachable
from a scenario too, not just from chat.

## Deployment topology (day 4): two images, one k3s namespace, one node

The two real FastAPI processes each get their own Dockerfile and their
own k3s Deployment/Service/PVC — `k3s/` (day 4) is the deployment-layer
mirror of this document's two-API-process architecture, not a third
thing:

- `Dockerfile.jetson` (week 08) → `vlm-inference` Deployment, `l4t-pytorch`
  base (needs CUDA for real CLIP inference), `runtimeClassName: nvidia`,
  6144Mi memory request=limit (mirrors `docker-compose.jetson.yml`'s
  `mem_limit: 6g`, ADR-002).
- `Dockerfile.chat.jetson` (day 4, new) → `chat-backend` Deployment,
  plain `python:3.11-slim` base — chat_app.py's import chain was checked
  for real and never touches torch/transformers/opencv/PIL, so it needs
  none of `Dockerfile.jetson`'s CUDA stack; 256Mi memory limit (4× week
  11 day 4's real ~64 MB measurement).

Both Deployments use `strategy: Recreate`, not the k8s default
(`RollingUpdate`) — a single 8 GiB node has no spare capacity to run
old+new Pods simultaneously the way a multi-node cluster's surge
strategy assumes. `k3s/README.md` covers the full reasoning, the health
check design (`startupProbe`→`readinessProbe`/`livenessProbe` on each
service's own `/health`), and rollback (`kubectl rollout undo`, gated on
git-SHA-tagged images so it actually reverts to a different image).

One real cross-service wiring gap this day closed: `assistant/config/bridges_config.yaml`'s
`vision_bridge.base_url` (`http://localhost:8000`) is only correct when
`chat-backend` and `vlm-inference` share a network namespace
(docker-compose) — on k3s they're separate Pods, reachable only via the
Service's in-cluster DNS name. `load_bridges_config` (`assistant/utils/config.py`)
now resolves `$VLM_INFERENCE_URL` over the YAML default when set —
`k3s/chat-backend-deployment.yaml` sets it to
`http://vlm-inference.vlm-edge.svc.cluster.local:8000`.

`.github/workflows/jetson-cicd.yml` builds both images cross-compiled
(QEMU + buildx on GitHub-hosted runners — no Jetson needed to build,
just to run afterward) and deploys via a **self-hosted runner installed
on the device itself** (avoids exposing the single node's k3s API
server to the public internet just so a GitHub-hosted runner could
reach it). `scripts/validate_k3s_manifests.py` (`make
validate-k3s-manifests`) is the real structural check available without
`kubectl`/a cluster in this sandbox — see `k3s/README.md`'s honesty note
for exactly what it does and doesn't catch.

## Why `assistant/chat_app.py` didn't move into `src/api/`

Two real FastAPI processes exist in this repo, and they stay two
processes: `src/api/app.py` (week 08, CLIP embedding, `/v1/embed/*`) and
`assistant/chat_app.py` (week 11 day 4, chat + WebSocket,
`/ws/chat`). Physically relocating either one into the other's directory
would not change how many processes run or how much memory either one
uses — it would only rename import paths across two already-tested,
already-working weeks' worth of code for zero functional benefit, the
same "don't rename for symmetry alone" principle this repo has followed
since `assistant/`'s own ADR-042 explained why it deliberately sits
*outside* `src/` in the first place. `src/api/` stays week 08's package;
this document (not a code move) is what makes the two-API-process shape
of the whole system explicit.

## Why `src/config.py` became a package instead of staying a file

Unlike the five pure re-export layers, `config` needed one genuinely new
piece: a single dev/jetson environment switch every other layer can
share (`src/config/environment.py`'s own docstring explains why this
wasn't already covered by `TARGETS`/`apply_target_overrides`,
`CONFIG_PATH`, or `assistant/config/*.yaml` on their own — each of those
already answers a narrower, single-subsystem question). Since a Python
package and a same-named module can't coexist, the existing
`src/config.py` moved to `src/config/__init__.py` verbatim (same
classes, same `load_config`/`apply_target_overrides`, same public API) —
none of the 18 existing `from src.config import ...` call sites across
`src/`/`scripts/`/`pipeline/`/`quantization/`/`tests/` needed to change,
verified by running the full suite immediately after the move.

## Memory footprint — does the whole stack fit an 8 GB Jetson Orin Nano?

**Yes, with real numbers, not a guess: 6591.8 MB against 8192 MB,
leaving ~1600 MB of OS headroom.** See
[`outputs/capstone_demo/memory_report.md`](../outputs/capstone_demo/memory_report.md)
(and `scripts/generate_capstone_memory_report.py`, `make
capstone-memory-report`) for the full table, every number's real
source, and the open risks not yet folded in (real open-vocabulary
detection's memory cost is unmeasured; `scenarios/`'s own footprint
isn't known until day 3; k3s's control-plane overhead is day 4's
question entirely).

Nothing needed to move off-device beyond what week 10 already
externalized: Prometheus + Grafana already run on a separate host
(ADR-035/036); only their three thin on-device agents
(tegrastats-writer/jetson-exporter/node-exporter, 384 MB combined) are
counted in the on-device sum, and that sum still fits.
