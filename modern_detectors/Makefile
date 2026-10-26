# Makefile -- modern_detectors/ (week 26: DETR/RT-DETR vs YOLO vs OWL-ViT)

.PHONY: setup test lint demo demo-offline clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q modern_detectors/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check modern_detectors

## Week 26's whole non-interactive deliverable, one command: train a real
## tiny RT-DETR and a real from-scratch YOLOv8n on the same synthetic
## defect dataset (week 25's generator), score both the same way, attempt
## the real OWL-ViT zero-shot baseline (honestly reported as blocked if
## this sandbox's network is unreachable), write one combined report.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) modern_detectors/scripts/run_modern_detectors_demo.py

## Same as `demo`, but skips the (network-dependent) OWL-ViT attempt entirely.
demo-offline:
	cd .. && PYTHONPATH=. $(PYTHON) modern_detectors/scripts/run_modern_detectors_demo.py --skip-owlvit

clean:
	rm -rf ../outputs/modern_detectors_demo ../runs
