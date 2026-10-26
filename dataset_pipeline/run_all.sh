#!/usr/bin/env bash
# run_all.sh -- runs all 8 dataset-pipeline stages in order (week 06).
#
# Usage:
#   ./run_all.sh                                          # default config/dataset_config.yaml
#   ./run_all.sh --config config/dataset_config.yaml       # explicit (same as default)
#
# Default `generation.count` (60, see config/dataset_config.yaml) is sized
# for a fast, zero-setup demo run, not a real dataset -- stage 7 will
# clearly fail past that point (`jetson_subset.min_count` needs at least
# 200 val samples, which 60 total samples spread across dozens of strata
# can't produce) unless `generation.count` is raised into the thousands
# for a real run. Stage 8 (the dataset-integrity gate) still runs against
# whatever stages 1-6 produced either way, and THIS SCRIPT's own exit code
# is stage 8's -- that's the real "is the dataset OK" verdict, not stage
# 7's demo-scale limitation. See dataset_pipeline/README.md.
#
# Every stage's stdout+stderr is duplicated (via `tee`) into
# outputs/logs/NN_stage.log as well as the terminal, same reasoning as
# pipeline/run_all.sh (week 04): one place already sequences every stage,
# so capturing logs there avoids repeating the same boilerplate in all 8 scripts.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="config/dataset_config.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "== run_all.sh: config=$CONFIG =="

OUTPUTS_DIR="$(python3 -c "
import sys
sys.path.insert(0, '.')
from utils.config import load_config
print(load_config('$CONFIG').paths.outputs_dir)
")"
LOGS_DIR="$OUTPUTS_DIR/logs"
mkdir -p "$LOGS_DIR"

echo "== Stage 1/8: generate synthetic scenes =="
python3 scripts/01_generate_synthetic.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/01_generate_synthetic.log"

echo "== Stage 2/8: stratify =="
python3 scripts/02_stratify_dataset.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/02_stratify_dataset.log"

echo "== Stage 3/8: augment =="
python3 scripts/03_augment_dataset.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/03_augment_dataset.log"

echo "== Stage 4/8: convert to COCO =="
python3 scripts/04_convert_coco.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/04_convert_coco.log"

echo "== Stage 5/8: convert to YOLO =="
python3 scripts/05_convert_yolo.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/05_convert_yolo.log"

echo "== Stage 6/8: split =="
python3 scripts/06_split_dataset.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/06_split_dataset.log"

echo "== Stage 7/8: export Jetson val subset =="
set +e
python3 scripts/07_export_jetson_val_subset.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/07_export_jetson_val_subset.log"
STAGE_7_EXIT_CODE="${PIPESTATUS[0]}"
set -e

echo "== Stage 8/8: validate dataset integrity =="
set +e
python3 scripts/08_validate_dataset.py --config "$CONFIG" 2>&1 | tee "$LOGS_DIR/08_validate_dataset.log"
STAGE_8_EXIT_CODE="${PIPESTATUS[0]}"
set -e

echo "== run_all.sh: stages 1-6 complete; stage 7 exit code: $STAGE_7_EXIT_CODE; stage 8 (validation gate) exit code: $STAGE_8_EXIT_CODE =="
echo "== Per-stage logs: $LOGS_DIR/ =="
exit "$STAGE_8_EXIT_CODE"
