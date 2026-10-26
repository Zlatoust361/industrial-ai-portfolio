# assistant/ — chat-driven VLM assistant (week 11)

Week 11 deliverable (day 4 of 5): `llm_parser.py` (day 2) — text →
intent + parameters, two-tier; `action_executor.py` +
`vision_bridge.py`/`ros_interface.py`/`opc_ua_bridge.py` (day 3) — dispatch
a parsed command to its domain; `chat_app.py` + `database.py`/
`database_models.py` (day 4) — FastAPI + WebSocket chat backend, SQLite
detection history, minimal HTML/JS frontend. A separate top-level
directory, same reasoning as `pipeline/`/`dataset_pipeline/`/
`quantization/`/`monitoring/`/`backup/`: week 11's own concern. Unlike
those, its modules use repo-root-qualified imports (`from
assistant.llm_parser import ...`), matching `src/`'s/`scripts/`'s
convention rather than a self-contained subsystem — day 3's bridges
reuse `src/` code directly (week 08's CLIP inference) and week 03's
`ros2_ws/src/vision_bridge` detector code via a `sys.path` addition, and
day 4's `chat_app.py` reuses `src/api/logging_config.py`, so `assistant/`
needs to sit at the same import level, not behind its own package
boundary.

```
assistant/
  llm_parser.py         # text -> intent + parameters, two-tier (day 2)
  schemas.py              # ParsedCommand + per-intent parameter schemas (pydantic)
  rate_limiter.py          # TokenBucketRateLimiter + ExternalCallQueue (day 2)
  action_executor.py        # dispatches a ParsedCommand to its domain (day 3)
  vision_bridge.py            # calls the deployed CLIP service + week 03's OWL-ViT detector (day 3)
  ros_interface.py             # ROS2 pub/sub (simulated bus, or real rclpy) (day 3)
  opc_ua_bridge.py              # OPC-UA/PLC stub -- read/write tags (day 3)
  database_models.py            # SQLite DDL for detection history (day 4)
  database.py                    # Database -- SQLite-backed DetectionLogProtocol impl (day 4)
  chat_app.py                     # FastAPI + WebSocket chat backend (day 4)
  static/
    index.html                    # minimal HTML/JS chat frontend (day 4)
  config/
    parser_config.yaml   # rate-limit + Claude API settings (day 2)
    bridges_config.yaml   # vision/ROS2/OPC-UA settings (day 3)
    chat_config.yaml       # SQLite path + host/port for chat_app.py (day 4)
  utils/
    config.py             # typed loaders for all three config files
  scripts/
    run_parser_demo.py    # `make demo` -- parse a batch, print/save the local-vs-API split
    run_action_demo.py      # `make action-demo` -- full parse -> execute pipeline
    measure_memory_footprint.py  # `make measure-memory` -- combined RSS vs week 01's budget (day 4)
  tests/                 # 151 pytest, PYTHONPATH=. from repo root
```

## Why two tiers

The Jetson Orin Nano this eventually runs on (same device as weeks 8-9's
detection service) has an 8 GB memory budget already mostly spent on the
CLIP/YOLO models (ADR-002) — there's no room for a second, general-purpose
LLM running locally just to parse chat commands. But most of what an
operator on the line actually types is the same handful of shapes over
and over: "show me the last N defects of type X above confidence Y",
"status", "restart training", "help". None of that needs an LLM at all.

- **Tier 1 — local rule-based (`match_locally`)**: plain `re` regexes,
  sub-millisecond, zero network, zero extra memory. Handles the frequent,
  well-shaped commands.
- **Tier 2 — external Claude API (`call_claude_api`)**: only reached when
  tier 1 finds nothing. Real tool-use (function calling) forcing a
  structured `{intent, parameters}` response, then validated against the
  exact same `assistant.schemas.INTENT_PARAM_SCHEMAS` registry tier 1
  uses — a hallucinated intent or malformed parameter fails identically
  regardless of which tier produced it, and is never executed
  unvalidated (`ParsedCommand.is_actionable` is `False` until it passes).

```
"покажи последние 50 дефектов crack с conf>0.9"
        │
        ▼
  match_locally()  ──── regex hit ────►  ParsedCommand(source="local")   [sub-ms, no network]
        │
     no match
        ▼
  rate_limiter.ExternalCallQueue.submit()
        │
        ▼
  call_claude_api()  ── tool_use ──►  validate_parameters()  ──►  ParsedCommand(source="api")
```

## Rate-limiting/queue on the external tier

This device sits on an industrial line with a potentially unstable or
metered internet uplink — a chat request that misses tier 1 must never
just hang on the network. `rate_limiter.py`:

- **`TokenBucketRateLimiter`**: caps external calls per time window,
  refilled continuously (not "reset every N seconds", which would let a
  burst through right at each boundary).
- **`ExternalCallQueue`**: a bounded backlog + one background worker
  thread draining it through the rate limiter. `submit()` never touches
  the network and never blocks on it — it either enqueues instantly
  (returning a `concurrent.futures.Future`) or raises `QueueFullError`
  immediately if the backlog is already full. `LLMParser.parse()` waits
  on that `Future` for at most `rate_limit.max_wait_seconds` (default 5s,
  `assistant/config/parser_config.yaml`) and returns a `"pending"` result
  instead of blocking further if the network/rate limit hasn't resolved
  it yet — the actual mechanism satisfying this day's "не должно
  блокироваться" requirement, not just a shorter HTTP timeout.

A real bug this design caught during testing: the first version marked a
queued item's `Future` `RUNNING` *before* waiting for a rate-limit token
— but a `Future` can only be cancelled while still `PENDING`
(`Future.cancel()` silently no-ops once `RUNNING`), so `shutdown()`
called while an item was stuck waiting for a token that wasn't refilling
in time left it `RUNNING` forever with no result ever set. Fixed by
waiting for the token first, only marking `RUNNING` right before actually
invoking the API call — caught by
`tests/test_rate_limiter.py::test_shutdown_cancels_items_still_waiting_for_a_rate_limit_token`,
not by inspection.

## Logging the local-vs-API split

`ParseStats` (in `llm_parser.py`) counts every outcome
(`local`/`api`/`pending`/`rejected`/`error`); `LLMParser.log_summary()`
emits one structured log line with the fractions (reusing week 08's
`src/api/logging_config.py` JSON formatter — see `run_parser_demo.py`),
and `save_stats()` writes the same summary as JSON. `make demo`'s own run
against 9 example commands in this sandbox (no `ANTHROPIC_API_KEY` — see
below): **78% local, 0% API, 2 clean errors** (the 2 free-form commands
correctly fall through to tier 2, correctly fail with a clear
"`$ANTHROPIC_API_KEY` is not set" message rather than hanging or
crashing) — see `outputs/week11/` once day 5 closes the week.

## Day 3: `action_executor.py` + three domain bridges

One handler per `assistant.schemas.KNOWN_INTENTS` entry
(`ActionExecutor.execute`), each calling into exactly the bridge that
owns its domain — never a monolithic dispatcher that knows every
system's details itself:

- **`vision_bridge.py`** — two genuinely different reuse shapes. Embedding
  + zero-shot ok/defect classification call the ALREADY-DEPLOYED FastAPI
  service (week 08's `src/api/app.py`, `http://localhost:8000`) over
  HTTP — this process never loads CLIP weights itself; an 8 GB Jetson has
  no room for a second copy of the model the deployed service already
  holds (ADR-002). Zero-shot classification math itself
  (`src.models.zero_shot.classify_by_cosine_similarity`, week 02) runs
  in-process — cheap, pure numpy, nothing to duplicate. Open-vocabulary
  object detection has no HTTP endpoint (the deployed service is
  embedding-only), so `detect_objects` reuses week 03's
  `ros2_ws/src/vision_bridge/vision_bridge/detector_factory.py`
  in-process instead, via a `sys.path` addition — its `Detector`/
  `Detection` types are deliberately rclpy-free, importable outside a
  ROS2 workspace by that module's own design.
- **`ros_interface.py`** — publishes text queries / training-restart
  commands, subscribes to detections, on the same topic names week 03's
  `vision_bridge` ROS2 node already uses by default
  (`/vision_bridge/text_query`, `/vision_bridge/detections`). Real
  `rclpy` isn't installed in the sandbox this was written in — `simulate:
  true` (the default, `assistant/config/bridges_config.yaml`) runs a
  real in-memory pub/sub bus instead (real multi-subscriber fan-out,
  real threading, no rclpy), so the dispatch logic is genuinely testable
  here; `simulate: false` is the real `rclpy` path, verified only to its
  exact `ModuleNotFoundError` boundary.
- **`opc_ua_bridge.py`** — deliberately a stub, not a real OT
  integration: `course/modules/ot-integration-for-cv-engineer/day04-plc-scada-integration-kks.md`
  explicitly frames week 11's OPC-UA bridge this way and defers real KKS
  tag naming / a real-or-simulated PLC to that later, dedicated module.
  Read-only by default, and `write_tag` needs TWO independent opt-ins
  (`config.read_only=false` AND a per-call `allow_write=True`) before it
  writes anything — writing to a PLC is safety-relevant, never implied by
  a single flag. Same `simulate`/real-`asyncua` split as `ros_interface.py`.

`list_defects` needed a real data source to filter/sort/limit over —
`DetectionLog` (in `action_executor.py`) was day 3's honest, explicitly-
labeled in-memory stand-in (`record()`/`query()`); day 4 (below) adds a
real SQLite-backed `assistant.database.Database` behind the exact same
`DetectionLogProtocol` shape, used by `chat_app.py` instead of
`DetectionLog`. `run_action_demo.py`'s own `ActionExecutor()` (below)
still uses the in-memory default deliberately — it's a stateless demo
script, not a long-running service.

```bash
make action-demo   # week 11's own two example commands, end to end
```

```
"покажи последние 50 дефектов crack с conf>0.9"
        │
        ▼ (LLMParser, day 2)
  ParsedCommand(intent="list_defects", parameters={defect_type: crack, min_confidence: 0.9, limit: 50})
        │
        ▼ (ActionExecutor.execute, day 3)
  re-validate against assistant.schemas.INTENT_PARAM_SCHEMAS
        │
        ▼
  DetectionLog.query(defect_type="crack", min_confidence=0.9, limit=50)
        │
        ▼
  ExecutionResult(success=True, data={"records": [...]})
```

## Day 4: `chat_app.py` + `database.py`/`database_models.py` + minimal frontend

- **`database_models.py`** — the SQLite DDL for one `detections` table
  (`defect_type`/`confidence`/`detected_at`/`source`, same fields day 3's
  `DetectionRecord` already has), plus an index on `detected_at DESC` --
  the exact column/direction `list_defects` always sorts by.
- **`database.py`** — `Database`, a real, file-backed, thread-safe
  (`sqlite3` + one `threading.Lock`) implementation of the exact same
  `record()`/`query()`/`__len__()` shape day 3's `DetectionLog` already
  has (`action_executor.DetectionLogProtocol`) — `ActionExecutor` needed
  zero changes to accept it, only what it's constructed with
  (`ActionExecutor(detection_log=Database(...))`). Real persistence
  across restarts, proven by a test that closes the connection, reopens
  the same file, and reads the row back.
- **`chat_app.py`** — a FastAPI app: `GET /health`, `GET /` (serves
  `static/index.html`), `WS /ws/chat`. `lifespan` opens one real
  `Database` + `ActionExecutor` + `LLMParser` for the process lifetime
  and tears them down cleanly on shutdown (stops the day 2 rate-limiter's
  worker thread, closes the ROS2/OPC-UA bridges, closes the SQLite
  connection) — skipping that would leak a real background thread per
  run. Each WebSocket message runs `LLMParser.parse()` →
  `ActionExecutor.execute()` via `starlette.concurrency.run_in_threadpool`,
  not inline in the `async def` handler — `parse()` can block up to
  `max_wait_seconds` (5s, day 2) waiting on a rate-limited external call,
  which would otherwise stall uvicorn's single event loop for every
  OTHER open chat connection for that long.
- **`static/index.html`** — plain HTML + inline JS, no framework/build
  step: connects to `/ws/chat`, sends the input box's text on submit,
  renders each JSON result as a message bubble. Verified for real in a
  headless Chromium browser (Playwright), not just via `curl`/pytest —
  screenshot and the exact commands used are in
  `course/DECISIONS.md`'s week 11 day 4 entry.

```bash
make chat-serve   # http://localhost:8100/ -- real chat UI, real WebSocket, real SQLite
```

## Combined memory footprint vs week 1's budget

`scripts/measure_memory_footprint.py` (`make measure-memory`) starts
`src/api/app.py` (the deployed detection service, weeks 08-09) and
`assistant/chat_app.py` (this day's chat backend) CONCURRENTLY as two
real `uvicorn` subprocesses, measures each one's real RSS via `psutil`,
and checks the combined number against
`config/model_config.jetson.yaml`'s `device_total_mb` (8192, week 01).

This sandbox has no `torch`, so it can't load real CLIP weights into the
detection service to measure its true post-load footprint — the number
that check actually needs for that service is `memory_budget_mb` (6144
MB), the configured ceiling ADR-002 (week 01) already defines for
"model weights + inference runtime working set" specifically so a real
Jetson deployment fails clean instead of getting OOM-killed. The chat
backend has no such dependency at all, so its measured RSS (against a
SQLite file pre-seeded with 2000 synthetic detections — an
operationally-realistic multi-day history, not an empty database) IS the
real number: **~64 MB** in this sandbox. Combined:
`6144 + 64 = 6208 MB`, leaving `8192 - 6208 ≈ 1984 MB` of OS headroom —
down from ADR-002's original ~2048 MB assumption, but still comfortably
within the 8 GB device budget. See `course/DECISIONS.md`'s week 11 day 4
entry and `outputs/assistant_demo/memory_footprint.json` for the full
real run.

## Running

```bash
make demo                          # from assistant/ -- runs against a built-in example set
make demo ARGS="--commands-file path/to/commands.txt"
make action-demo                   # full parse -> execute pipeline, week 11's own example commands
make chat-serve                    # FastAPI + WebSocket chat backend + frontend, http://localhost:8100/
make measure-memory                # combined RSS vs week 01's 8 GB budget, writes memory_footprint.json
cd assistant && make test          # 151 pytest passed
```

## Honesty note: no real Claude API call happens anywhere in this sandbox

This session's own infrastructure has no `ANTHROPIC_API_KEY` exposed to
it for `llm_parser.py`'s deployed-application use, and using this
session's own credentials to call the API on the built application's
behalf would conflate two separate identities and spend a budget this
session isn't authorized to spend. `call_claude_api` is real, correct
code — real tool-use schema, real request construction, real response
parsing and validation — verified against a real `httpx.MockTransport`-backed
client (`tests/test_llm_parser.py`), the same pattern week 05's
`StabilityAPIBackend` established for the Stability AI API. It has never
been run against the real `https://api.anthropic.com` endpoint. Tier 1
(`match_locally`), the schema validation, and the entire rate-limiting/
queue machinery (`rate_limiter.py`) need no external service at all and
are exercised for real throughout.

## Honesty note (day 3): what's real, what's a documented stub

`ActionExecutor`'s dispatch logic, `DetectionLog`'s filtering, and both
`ros_interface.py`/`opc_ua_bridge.py`'s `simulate=True` paths are real
code exercised for real (real threads, real pub/sub fan-out, real
read/write-tag semantics) — none of it is mocked. What genuinely isn't
runnable here: `vision_bridge.py`'s `detect_objects` against the real
`"pytorch"`/`"tensorrt"` backends (no `torch`, verified only to the exact
`ModuleNotFoundError` boundary, same as `llm_parser.py`'s Claude API
tier), and `ros_interface.py`/`opc_ua_bridge.py`'s `simulate=False` real
`rclpy`/`asyncua` paths (neither installed here, same boundary-only
verification). `vision_bridge.py`'s `health_check` IS tested against the
real `src/api/app.py` FastAPI application (via `TestClient`, not a mock)
— confirming the HTTP-calling code itself is correct, even though no
real CLIP inference ran anywhere in this session.

## Honesty note (day 4): what's real, what's modeled

`database.py`'s SQLite persistence, `chat_app.py`'s FastAPI routes and
`WS /ws/chat` handler, and `static/index.html`'s frontend are all real,
exercised for real: pytest drives real `TestClient` HTTP + WebSocket
round trips (`tests/test_chat_app.py`), and a headless-Chromium
(Playwright) browser session drove the actual served page against a
real running `uvicorn` process for the screenshot in
`course/DECISIONS.md`'s week 11 day 4 entry — not a mock of the page,
the actual page. `measure_memory_footprint.py`'s two `uvicorn`
subprocesses and their `psutil` RSS readings are real measurements on
this sandbox's CPU/x86_64 host, not a Jetson. What's explicitly modeled
rather than measured: the detection service's post-model-load memory —
this sandbox has no `torch` to actually load CLIP weights and reach
steady state, so the combined-budget check uses
`config/model_config.jetson.yaml`'s configured `memory_budget_mb`
ceiling (ADR-002, week 01) for that half of the sum instead of a guess.
