# Makefile -- distillation/ (week 28: сжатие моделей с нуля)

.PHONY: setup test lint demo clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q distillation/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check distillation

## Week 28's whole non-interactive deliverable, one command: train a
## real teacher + student, distill, structurally prune, run a basic NAS
## search, statically quantize to real INT8, score everything the same
## way, write one combined comparison report.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) distillation/scripts/run_distillation_demo.py

clean:
	rm -rf ../outputs/distillation_demo
