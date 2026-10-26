# Makefile -- captioning/ (week 15: VLM captioning/VQA on edge)

.PHONY: setup test lint demo clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q captioning/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check captioning

## Week 15's whole non-interactive deliverable, one command: synthetic
## verdicts -> caption confirmed defects -> reference-free CLIP scoring.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) captioning/scripts/run_captioning_demo.py --config captioning/config/config.yaml $(ARGS)

## Remove runtime artifacts -- config/ and scripts/ untouched. `demo`
## above `cd ..` before running, so config.yaml's relative output paths
## (outputs/captioning_demo, .cache/) resolve against the REPO ROOT, not
## this directory -- clean must `cd ..` too (see fleet/Makefile's own
## clean target fix for the bug this avoids).
clean:
	cd .. && rm -rf outputs/captioning_demo .cache
	rm -rf __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned captioning/ runtime artifacts."
