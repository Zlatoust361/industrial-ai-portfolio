# Makefile -- quantization/ (week 09: INT4 quantization + TensorRT)

.PHONY: setup setup-jetson test lint baseline export-onnx prepare-calibration quantize \
        build-engine build-engine-fp16 trt-inference evaluate-quality profile clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
## Steps 1-3/8 (baseline/export-onnx/prepare-calibration/profile) run here.
## Do NOT run this on the Jetson itself -- ../requirements.txt pulls torch
## from a CPU-only x86_64 wheel index that doesn't resolve on aarch64 at
## all, which aborts the whole `pip install` (pyyaml/numpy/pillow included)
## before it installs anything. Use `make setup-jetson` on-device instead.
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Install deps for steps 4-7 (quantize/build-engine/trt-inference/
## evaluate-quality) ON THE JETSON ITSELF. Those scripts + utils/
## tensorrt_builder.py/trt_inference.py only need pyyaml/numpy/pillow at
## import time -- tensorrt/pycuda are imported lazily and come from
## JetPack, never from pip (see README.md). Reuses ../requirements-jetson.txt
## (the Dockerfile.jetson deps list) precisely because it already excludes
## torch/the CPU wheel index for this same reason -- not because this
## needs a FastAPI service, just because it's the correct dependency set.
setup-jetson:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements-jetson.txt

## Run the unit test suite.
test:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/

## Lint this directory.
lint:
	$(PYTHON) -m ruff check .

## Step 1: FP16 baseline latency/throughput -> outputs/baseline/baseline_results.json.
baseline:
	PYTHONPATH=. $(PYTHON) scripts/01_baseline_inference.py $(if $(CONFIG),--config $(CONFIG),)

## Step 2: export the baseline's image encoder to ONNX + smoke-test it.
export-onnx:
	PYTHONPATH=. $(PYTHON) scripts/02_export_onnx.py $(if $(CONFIG),--config $(CONFIG),)

## Step 3: sample + preprocess the calibration image set from week 06's dataset.
prepare-calibration:
	PYTHONPATH=. $(PYTHON) scripts/03_prepare_calibration_data.py $(if $(CONFIG),--config $(CONFIG),)

## Step 4: TensorRT calibration -- ON THE JETSON ONLY (pass FORCE=1 to override
## the device check for a dev-machine smoke test; the result won't be meaningful).
quantize:
	PYTHONPATH=. $(PYTHON) scripts/04_quantize_model.py \
		$(if $(CONFIG),--config $(CONFIG),) $(if $(PROFILE),--profile $(PROFILE),) $(if $(FORCE),--force,)

## Step 5: serialize the final quantized .engine -- ON THE JETSON ONLY (same FORCE=1 escape hatch).
build-engine:
	PYTHONPATH=. $(PYTHON) scripts/05_build_engine.py \
		$(if $(CONFIG),--config $(CONFIG),) $(if $(PROFILE),--profile $(PROFILE),) $(if $(FORCE),--force,)

## Step 5, --fp16-only: the plain-FP16 reference engine 06/07 compare against.
build-engine-fp16:
	PYTHONPATH=. $(PYTHON) scripts/05_build_engine.py --fp16-only \
		$(if $(CONFIG),--config $(CONFIG),) $(if $(PROFILE),--profile $(PROFILE),) $(if $(FORCE),--force,)

## Step 6: latency for one engine (ENGINE=path LABEL=fp16|int4 required) -- ON THE
## JETSON ONLY. Pass MODE=0|1|2 to switch nvpmodel first (0=MAXN_SUPER, 1=15W,
## 2=7W); omit to benchmark under whatever mode the device is already in.
trt-inference:
	@if [ -z "$(ENGINE)" ] || [ -z "$(LABEL)" ]; then \
		echo "Usage: make trt-inference ENGINE=path/to.engine LABEL=fp16|int4 [MODE=0|1|2] [CONFIG=...]"; \
		exit 1; \
	fi
	PYTHONPATH=. $(PYTHON) scripts/06_trt_inference.py --engine $(ENGINE) --label $(LABEL) \
		$(if $(CONFIG),--config $(CONFIG),) $(if $(MODE),--nvpmodel-mode $(MODE),) $(if $(FORCE),--force,)

## Step 7: FP16-vs-INT4 embedding quality gate -- ON THE JETSON ONLY (both
## engines from build-engine/build-engine-fp16 must already exist). Same
## MODE=0|1|2 nvpmodel override as trt-inference.
evaluate-quality:
	PYTHONPATH=. $(PYTHON) scripts/07_evaluate_quality.py \
		$(if $(CONFIG),--config $(CONFIG),) $(if $(MODE),--nvpmodel-mode $(MODE),) $(if $(FORCE),--force,)

## Step 8: aggregate every outputs/trt_inference/*.json + the quality/baseline
## reports into outputs/summary.md. NOT Jetson-gated -- pure JSON aggregation,
## runs anywhere once 06 (and optionally 07/01) have produced their reports.
profile:
	PYTHONPATH=. $(PYTHON) scripts/08_profile.py $(if $(CONFIG),--config $(CONFIG),)

## Remove runtime artifacts (outputs/.cache) -- config/ and scripts/ untouched.
clean:
	rm -rf outputs .cache __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned quantization/ runtime artifacts."
