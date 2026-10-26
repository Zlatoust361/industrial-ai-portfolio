# Makefile -- compliance/ (week 21: security и лицензионный аудит)

.PHONY: setup test lint demo clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q compliance/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check compliance

## Week 21's whole non-interactive deliverable, one command: SBOM ->
## license audit (confirms ultralytics/AGPL-3.0 for real) -> ИБ
## checklist -> adversarial-patch demo with measured before/after.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) compliance/scripts/run_license_audit.py $(ARGS)

## Remove runtime artifacts -- config/ and scripts/ untouched. `demo`
## above `cd ..` before running, so its default relative output path
## (outputs/compliance_demo) resolves against the REPO ROOT, not this
## directory -- clean must `cd ..` too (see fleet/Makefile's own clean
## target fix for the bug this avoids).
clean:
	cd .. && rm -rf outputs/compliance_demo
	rm -rf __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned compliance/ runtime artifacts."
