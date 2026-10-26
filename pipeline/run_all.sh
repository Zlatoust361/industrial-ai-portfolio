#!/usr/bin/env bash
# run_all.sh -- runs all 7 inspection-pipeline stages in order.
#
# Usage:
#   ./run_all.sh                                       # demo: config/demo_config.yaml
#   ./run_all.sh --config config/pipeline_config.yaml   # real run (needs torch/transformers
#                                                        # or ultralytics + real data/raw/)
#   ./run_all.sh --device jetson                        # forces fp16 + batch_size=1 in
#                                                        # 04_detect.py (see utils/config.py::
#                                                        # apply_device_target); meaningful only
#                                                        # with a real clip/yolo backend
#
# Default config is demo_config.yaml (detect.backend: stub) so a fresh
# checkout can run this end to end with zero setup, no network, no model
# weights -- see scripts/04_detect.py::StubDetector and
# scripts/generate_demo_dataset.py, which this script calls automatically
# if <data_dir>/raw/ is empty.
#
# Exits non-zero the moment any of stages 1-5 fails. Stage 6
# (06_quality_gate.py) is handled specially: its exit code (0 GREEN/YELLOW,
# 1 RED) becomes THIS SCRIPT's own exit code, but stage 7 (the report)
# still runs first regardless -- a RED verdict is exactly when you most
# want the report generated, not skipped. This is what CI's pipeline job
# runs to gate the build on outputs/metrics/quality_gate.json.
#
# Every stage's stdout+stderr is duplicated (via `tee`) into
# <logs_dir>/NN_stage_name.log as well as the terminal -- each stage script
# itself only prints, it doesn't know about logs_dir; capturing it here,
# once, in the one place that already sequences every stage, avoids adding
# the same boilerplate to all 7 scripts individually.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="config/demo_config.yaml"
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "== run_all.sh: config=$CONFIG device=${DEVICE:-<config default>} =="

DATA_DIR="$(python3 -c "
import sys
sys.path.insert(0, '.')
from utils.config import load_config
print(load_config('$CONFIG').paths.data_dir)
")"
LOGS_DIR="$(python3 -c "
import sys
sys.path.insert(0, '.')
from utils.config import load_config
print(load_config('$CONFIG').paths.logs_dir)
")"
mkdir -p "$LOGS_DIR"

if [ ! -d "$DATA_DIR/raw" ] || [ -z "$(ls -A "$DATA_DIR/raw" 2> /dev/null)" ]; then
  echo "== $DATA_DIR/raw is empty -- generating a synthetic demo dataset =="
  python3 scripts/generate_demo_dataset.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/00_generate_demo_dataset.log"
fi

DETECT_DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then
  DETECT_DEVICE_ARGS=(--device "$DEVICE")
fi

echo "== Stage 1/7: environment check =="
python3 scripts/01_environment_check.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/01_environment_check.log"

echo "== Stage 2/7: ingest =="
python3 scripts/02_ingest.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/02_ingest.log"

echo "== Stage 3/7: preprocess =="
python3 scripts/03_preprocess.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/03_preprocess.log"

echo "== Stage 4/7: detect =="
python3 scripts/04_detect.py --config "$CONFIG" "${DETECT_DEVICE_ARGS[@]}" 2>&1 | tee "$LOGS_DIR/04_detect.log"

echo "== Stage 5/7: interpret =="
python3 scripts/05_interpret.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/05_interpret.log"

echo "== Stage 6/7: quality gate =="
set +e
python3 scripts/06_quality_gate.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/06_quality_gate.log"
GATE_EXIT_CODE="${PIPESTATUS[0]}"
set -e

echo "== Stage 7/7: generate report =="
python3 scripts/07_generate_report.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/07_generate_report.log"

echo "== run_all.sh: all 7 stages complete (quality gate exit code: $GATE_EXIT_CODE) =="
echo "== Per-stage logs: $LOGS_DIR/ =="
exit "$GATE_EXIT_CODE"
