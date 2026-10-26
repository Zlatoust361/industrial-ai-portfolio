# Makefile -- dataset_pipeline/ (week 06 synthetic dataset pipeline)

.PHONY: setup test lint pipeline jetson-subset validate clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite (utils/ pure logic + script-level CLI wiring; no heavy deps needed).
test:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/

## Lint this directory.
lint:
	$(PYTHON) -m ruff check .

## Run all 8 stages via run_all.sh; exits with stage 8's (the dataset-
## integrity gate) code. Default config/dataset_config.yaml
## (generation.count: 60) is a fast, zero-setup demo -- stage 7 exits 1
## past that point (not enough val samples for jetson_subset.min_count),
## but stage 8 still validates whatever stages 1-6 produced; pass
## CONFIG=path/to/a/real/config.yaml with a much larger generation.count
## for an actual dataset.
pipeline:
	./run_all.sh $(if $(CONFIG),--config $(CONFIG),)

## Re-run just stage 7 (e.g. after tuning jetson_subset.* without
## regenerating the whole dataset).
jetson-subset:
	PYTHONPATH=. $(PYTHON) scripts/07_export_jetson_val_subset.py $(if $(CONFIG),--config $(CONFIG),)

## Re-run just stage 8 (the dataset-integrity gate) against whatever's
## already in data/outputs, without regenerating anything.
validate:
	PYTHONPATH=. $(PYTHON) scripts/08_validate_dataset.py $(if $(CONFIG),--config $(CONFIG),)

## Remove runtime artifacts (data/outputs) -- config/scripts/utils untouched.
clean:
	rm -rf data outputs
