# Makefile -- classical_cv/ (week 25: classical CV как альтернатива VLM)

.PHONY: setup test lint demo clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q classical_cv/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check classical_cv

## Week 25's whole non-interactive deliverable, one command: generate a
## real synthetic defect dataset (week 05's mask generator), run the real
## OpenCV classical-CV detector, attempt the real CLIP zero-shot baseline
## (honestly reported as blocked if this sandbox's network is unreachable),
## write one combined comparison report.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) classical_cv/scripts/run_classical_cv_demo.py

## Same as `demo`, but skips the (network-dependent) CLIP baseline attempt
## entirely -- useful for a fast, fully offline classical-CV-only run.
demo-offline:
	cd .. && PYTHONPATH=. $(PYTHON) classical_cv/scripts/run_classical_cv_demo.py --skip-clip

clean:
	rm -rf ../outputs/classical_cv_demo
