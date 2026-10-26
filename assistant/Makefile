# Makefile -- assistant/ (week 11: chat-driven VLM assistant)
#
# Unlike quantization/monitoring/backup's Makefiles, these targets `cd ..`
# first: assistant/'s own modules use repo-root-qualified imports
# (`from assistant.llm_parser import ...`, matching src/'s and scripts/'s
# convention, not the "self-contained subsystem" convention those other
# weeks use) and its default config path (`assistant/config/parser_config.yaml`)
# is repo-root-relative -- both need PYTHONPATH/CWD to be the repo root,
# not this directory.

.PHONY: test lint demo action-demo chat-serve measure-memory week-demo clean

PYTHON ?= python3

## Run the unit + integration test suite.
test:
	cd .. && PYTHONPATH=. $(PYTHON) -m pytest -q assistant/tests/

## Lint this directory.
lint:
	cd .. && $(PYTHON) -m ruff check assistant

## Run llm_parser.py against a batch of example commands (real, not
## staged -- see run_parser_demo.py's own docstring for what "real" means
## here without a real ANTHROPIC_API_KEY). Prints + saves the local-vs-API
## split to outputs/assistant_demo/.
demo:
	cd .. && PYTHONPATH=. $(PYTHON) assistant/scripts/run_parser_demo.py $(ARGS)

## Run the full parse -> execute pipeline (day 2 + day 3 wired together)
## against week 11's own example commands ("покажи последние 50 дефектов
## crack с conf>0.9", "перезапусти обучение", ...), against seeded demo
## detections. Prints + saves results to outputs/assistant_demo/.
action-demo:
	cd .. && PYTHONPATH=. $(PYTHON) assistant/scripts/run_action_demo.py $(ARGS)

## Serve the FastAPI + WebSocket chat backend + minimal HTML/JS frontend
## (day 4) at http://<host>:8100/. Real, persistent SQLite detection
## history at outputs/assistant_demo/detections.db (config/chat_config.yaml).
## Pass ARGS="--reload" for dev auto-reload.
chat-serve:
	cd .. && PYTHONPATH=. $(PYTHON) -m uvicorn assistant.chat_app:app --host 0.0.0.0 --port 8100 $(ARGS)

## Start src/api/app.py (detection service) and assistant/chat_app.py
## (this day's chat backend) CONCURRENTLY as real uvicorn subprocesses,
## measure real RSS via psutil, combine with config/model_config.jetson.yaml's
## configured memory_budget_mb ceiling, and check the result against the
## Jetson Orin Nano's 8 GB (device_total_mb) from week 01. Writes
## outputs/assistant_demo/memory_footprint.json.
measure-memory:
	cd .. && PYTHONPATH=. $(PYTHON) assistant/scripts/measure_memory_footprint.py $(ARGS)

## One command for week 11's whole non-interactive deliverable (review,
## day 5): local-vs-API parse split (day 2), full parse -> execute
## pipeline against week 11's own example commands (day 3), combined
## memory footprint vs week 01's budget (day 4). `chat-serve` is
## deliberately NOT part of this -- it's a long-running server, not a
## one-shot artifact this chain can wait on.
week-demo: demo action-demo measure-memory
	@echo "week 11 deliverable reproduced -- see outputs/assistant_demo/ for all three runs' output."

## Remove test/lint caches.
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2> /dev/null || true
	@echo "Cleaned assistant/ caches."
