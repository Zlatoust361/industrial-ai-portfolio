# tracking/ — week 13: video and temporal defect tracking

Status: **first implementation pass**, part of
[`course/ROADMAP-WEEKS-13-24.md`](../course/ROADMAP-WEEKS-13-24.md)'s
volume 2 (not wired into `course.yaml`/`validate_course.py` — see that
roadmap's open question #1). Lives at the repo root, next to
`quantization/`/`peft/`/`fleet/`, not under `course/weeks/` (same split
every other week already has: `course/weeks/weekNN-*/` documents the
week, this directory IS the week's code).

## Idea

Every detector in this course (weeks 02-09) classifies ONE frame at a
time. Fed a real video stream, that produces a verdict per frame —
and a per-frame verdict flickers: the same physical crack gets called
`"ok"` on one frame and `"defect"` on the next from blur, glare, or a bad
angle (`course/FAILURE-TABLES.md#об-02`'s "разделимость низкая" entry
names exactly this). A human watching a flickering overlay on a
conveyor-line feed doesn't trust it, and a verdict that changes 30 times
a second isn't actionable.

This week's answer: give each detection a STABLE identity across frames
(tracking-by-detection), then smooth that identity's verdict over a
rolling window (majority vote) instead of trusting any single frame. Plus
the budget question the roadmap names explicitly: a Jetson can't
necessarily run the detector on every single frame of a real stream
alongside everything else already running — `frame_sampling.py` is the
"process every Nth frame" half of that tradeoff.

## What's new here vs. reused from other weeks

| Piece | Status |
|---|---|
| `sort.py` | **New** — `IoUTracker`, a SORT-style (tracking-by-detection) tracker: Hungarian-algorithm (`scipy.optimize.linear_sum_assignment`) IoU matching between each frame's detections and existing tracks' last-known boxes. No Kalman filter/motion model — deliberate first-pass simplification, see its own docstring. |
| `smoothing.py` | **New** — `VerdictSmoother`, a per-track-id rolling-window majority vote over raw labels, plus `count_flips()`, this week's headline "flicker" metric. |
| `frame_sampling.py` | **New** — `select_frame_indices`/`effective_detection_rate`/`max_stride_for_budget`: the stride-vs-Jetson-budget tradeoff named explicitly in the roadmap's week 13 entry. |
| `adapters.py` | **New**, tiny — converts a real week-03 `vision_bridge.detection.Detection` into this module's own `Detection` (duck-typed attribute copy; no import needed for the conversion itself, see below for why). |
| Detections themselves | **Not reused by import** — `tracking`'s `Detection`/`BBox` are its OWN dataclasses, same shape as `ros2_ws/src/vision_bridge/vision_bridge/detection.py`'s `Detection`/`BoundingBox`, deliberately NOT the same class (see `sort.py`'s docstring: a generic tracking library has no business depending on one ROS2 colcon package's nested file layout). `adapters.py` bridges the two, and `tests/test_adapters.py` proves the bridge against the REAL class. |

## Pipeline

```
scripts/run_tracking_demo.py
        │  generates a deterministic SYNTHETIC multi-object stream
        │  (two objects, straight-line motion, injected per-frame label
        │  noise) -- see its own docstring for exactly why synthetic
        ▼
    IoUTracker.update() per frame  ->  stable track_id per object
        ▼
    VerdictSmoother.update() per track  ->  majority-vote label
        ▼
    count_flips(raw) vs count_flips(smoothed) per track
        │  writes outputs/tracking_demo/report.json
        ▼
    NEXT (real hardware): adapters.from_vision_bridge_detection() +
    week 03's REAL vision_bridge detector output, frame-by-frame from
    a real camera/video file on the Jetson -- this module's actual
    target input, not yet run against it in this sandbox (see Honesty
    note below).
```

## Why IoU-only matching, not label-gated matching

The first version of `IoUTracker._match` refused to match a detection to
a track of a different label ("never match across defect types"). That
was a real design bug, caught and fixed **before any test was written**,
by reasoning through what this week's deliverable actually needs: a
label flip on the SAME physical object is exactly the noise
`VerdictSmoother` exists to clean up. If matching requires the label to
already agree, a single noisy flip breaks the track's identity and a new
track starts — there is nothing left for the smoother to smooth, and the
"before/after flip count" this week reports would be meaningless. Fixed
by making matching purely spatial (IoU), and updating `track.label` on
every match (not just `bbox`/`score`, which the original version also
missed). The accepted cost is documented in `sort.py`'s own docstring:
two genuinely different, spatially-overlapping objects could merge into
one track — a real tradeoff, not hidden, and a production tracker would
likely blend label similarity into the cost matrix instead of using it as
a hard gate.

A second real bug was caught only once the test suite ran (not by
reasoning alone): `IoUTracker.update()` aged newly-created tracks in the
same frame they were created, because the "age unmatched tracks" loop ran
over `self._tracks` AFTER new tracks had already been appended to it —
immediately setting a brand-new track's `time_since_update` to `1`, which
silently dropped it from that frame's returned detections even with
`min_hits=1`. Fixed by aging only the tracks that existed before this
frame's new ones are appended (see `sort.py::update`'s comment). Caught
by `test_first_frame_creates_one_track_per_detection` and six other
tests failing identically — exactly the kind of bug "actually run the
code" exists to catch, not just reading it.

## Honesty note — not yet run against a real video stream in this sandbox

What IS verified for real in this sandbox (49 tests, all passing):

- `sort.py`: `iou()` against hand-computed overlap fractions; `IoUTracker`
  construction validation; track creation/identity-preservation/aging/
  dropping across frames with real small-motion/far-away synthetic boxes;
  and — the test that matters most this week — a track's id surviving a
  raw label flip across consecutive frames, confirming the fix described
  above actually works, not just that it compiles.
- `smoothing.py`: majority vote, window sliding once full, per-track
  independence, `reset()`, tie-breaking, and `count_flips()` — including
  one test asserting smoothing never INCREASES the flip count on a
  genuinely noisy sequence.
- `frame_sampling.py`: full coverage of the stride/budget arithmetic,
  including the boundary (`headroom`, zero-latency, invalid inputs).
- `adapters.py`: against both a hand-built duck-typed object AND the
  REAL `ros2_ws/src/vision_bridge/vision_bridge/detection.py` classes
  (imported via the same lazy `sys.path` trick `assistant/vision_bridge.py`
  already established) — proven against the actual week-03 type, not
  just something that looks like it.
- `scripts/run_tracking_demo.py`: determinism for a fixed seed, seed
  divergence, the zero-noise boundary case, and the end-to-end
  smoothed-flips-≤-raw-flips claim this week's demo is actually about —
  run for real via `make demo`, producing `outputs/tracking_demo/report.json`
  (59 raw flips → 25 smoothed flips at `noise_rate=0.3`, 60 frames, two
  objects — the artifact itself was deleted after verifying it, not
  committed).

**Not yet done**: a real run against a real video stream on real
hardware, through week 03's actual `vision_bridge` detector output
frame-by-frame (`adapters.py` is the bridge for exactly that, untested
against a LIVE detector — only against the real `Detection` TYPE). That's
the natural next step once this reaches real hardware — see
`course/JETSON-COMMANDS.md`/`JETSON-GUIDE.md` for how this session's
other weeks' real-hardware runs were tracked, and add this week there
once it has one. Also not done: anything involving occlusion or
near-miss objects (`scripts/run_tracking_demo.py`'s two synthetic objects
stay spatially far apart throughout — see its own docstring) or a Kalman
filter/motion model (see `sort.py`'s docstring for why that was
deliberately left out of this first pass).

## Usage

```bash
cd tracking
make setup
make demo                                    # defaults: 60 frames, noise_rate=0.3
make demo NUM_FRAMES=120 NOISE_RATE=0.4 WINDOW_SIZE=7 SEED=7
```

See `tracking/Makefile` for each target's exact CLI, and
`course/weeks/week13-video-temporal-tracking/MANUAL-RUN.md` for the
code-level walkthrough and known-failure-boundary details.
