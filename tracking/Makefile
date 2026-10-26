# Makefile -- tracking/ (week 13: video and temporal defect tracking)

.PHONY: setup test lint demo clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
## Nothing here needs a GPU, Jetson, or network access: numpy + scipy
## (Hungarian matching) only.
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/

## Lint this directory.
lint:
	$(PYTHON) -m ruff check .

## Run the synthetic-stream demo end-to-end, write outputs/tracking_demo/report.json.
demo:
	PYTHONPATH=. $(PYTHON) scripts/run_tracking_demo.py $(if $(NUM_FRAMES),--num-frames $(NUM_FRAMES),) $(if $(NOISE_RATE),--noise-rate $(NOISE_RATE),) $(if $(WINDOW_SIZE),--window-size $(WINDOW_SIZE),) $(if $(SEED),--seed $(SEED),)

## Remove runtime artifacts (outputs/.cache) -- scripts/ and tests/ untouched.
clean:
	rm -rf outputs .cache __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned tracking/ runtime artifacts."
