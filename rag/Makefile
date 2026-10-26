# Makefile -- rag/ (week 19: RAG over factory regulations)

.PHONY: setup test lint demo clean

PYTHON ?= python3

## Install runtime + dev dependencies (shared with the repo root -- see ../requirements*.txt).
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt -r ../requirements-dev.txt

## Run the unit test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q rag/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check rag

## Week 19's whole non-interactive deliverable, one command: ingest the
## real docs/runbook.md + docs/architecture.md, answer the roadmap's own
## example questions plus one deliberately unrelated one.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) rag/scripts/run_rag_demo.py $(ARGS)

## Remove runtime artifacts -- `demo` above `cd ..` before running, so
## its default relative output path resolves against the REPO ROOT, not
## this directory (see fleet/Makefile's own clean target for the bug this avoids).
clean:
	cd .. && rm -rf outputs/rag_demo .cache
	rm -rf __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned rag/ runtime artifacts."
