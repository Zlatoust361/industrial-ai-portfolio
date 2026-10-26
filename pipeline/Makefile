# Makefile -- pipeline/ (week 04 inspection pipeline)

.PHONY: setup test lint pipeline pipeline-jetson report clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite (pure utils/ logic, no heavy deps needed).
test:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/

## Lint this directory.
lint:
	$(PYTHON) -m ruff check .

## Run all 7 stages via run_all.sh. Defaults to the demo dataset + stub
## backend (zero setup, see run_all.sh) -- pass CONFIG=config/pipeline_config.yaml
## for a real clip/yolo run against your own data/raw/.
pipeline:
	./run_all.sh $(if $(CONFIG),--config $(CONFIG),)

## Same as `pipeline`, but forces fp16 + batch_size=1 in 04_detect.py's model
## inference (see utils/config.py::apply_device_target, course/DECISIONS.md
## ADR-002 for the 8 GB Orin Nano memory-budget default). Run this ON the
## Jetson Orin Nano itself, in 15W/MAXN_SUPER mode, with CONFIG pointing at a
## real clip/yolo config -- the stub backend ignores precision/batch_size.
pipeline-jetson:
	./run_all.sh --device jetson $(if $(CONFIG),--config $(CONFIG),)

## Regenerate just the final report from already-produced outputs/ (e.g.
## after editing quality_gate thresholds without re-running detection).
report:
	PYTHONPATH=. $(PYTHON) scripts/07_generate_report.py $(if $(CONFIG),--config $(CONFIG),)

## Remove runtime artifacts (data/outputs/logs/.cache) -- config/ and scripts/ untouched.
clean:
	rm -rf data outputs logs .cache __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned pipeline/ runtime artifacts."
