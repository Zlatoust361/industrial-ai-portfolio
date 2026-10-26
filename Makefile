.PHONY: venv setup download-models convert benchmark jetson-benchmark \
        classify optimize-prompts train-linear-head generate-masks inpaint \
        defect-report defect-pipeline serve build-jetson \
        scenario-color-sorting scenario-defect-detection validate-k3s-manifests \
        capstone-demo test lint

PYTHON ?= python

## Один venv на корень + pipeline/ + dataset_pipeline/ + quantization/ +
## distillation/ + captioning/ + rag/ + compliance/ + modern_detectors/ --
## у каждого из них своя make setup (безопасно, но избыточно запускать
## повторно после этой).
venv:
	python3 -m venv .venv
	@echo "Created .venv. Activate it, then run make setup:"
	@echo "  source .venv/bin/activate"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

## Трек B (edge-ai): CLIP на edge, ONNX-экспорт, Jetson-бенчмарк.
download-models:
	PYTHONPATH=. $(PYTHON) scripts/01_download_models.py

convert:
	PYTHONPATH=. $(PYTHON) scripts/02_convert_onnx.py

benchmark:
	PYTHONPATH=. $(PYTHON) scripts/03_benchmark.py $(ARGS)

jetson-benchmark:
	PYTHONPATH=. $(PYTHON) scripts/jetson_benchmark.py $(if $(HOST),--host $(HOST),)

## Трек A (cv-inspection): zero-shot дефектоскопия на CLIP.
classify:
	@if [ -z "$(IMAGES)" ]; then echo "Usage: make classify IMAGES=path/to/images [ARGS=...]"; exit 1; fi
	PYTHONPATH=. $(PYTHON) scripts/clip_classifier.py --input-dir $(IMAGES) $(ARGS)

optimize-prompts:
	@if [ -z "$(VAL)" ]; then echo "Usage: make optimize-prompts VAL=path/to/validation [ARGS=...]"; exit 1; fi
	PYTHONPATH=. $(PYTHON) scripts/optimize_prompts.py --validation-dir $(VAL) $(ARGS)

train-linear-head:
	@if [ -z "$(TRAIN)" ]; then echo "Usage: make train-linear-head TRAIN=path/to/train [ARGS=...]"; exit 1; fi
	PYTHONPATH=. $(PYTHON) scripts/train_linear_head.py --train-dir $(TRAIN) $(ARGS)

## Трек A: синтетические дефекты (inpainting) -- generate-masks -> inpaint -> defect-report.
generate-masks:
	PYTHONPATH=. $(PYTHON) scripts/generate_defect_masks.py \
		$(if $(IMAGES),--image-dir $(IMAGES),) $(ARGS)

inpaint:
	PYTHONPATH=. $(PYTHON) scripts/run_diffusion_pipeline.py \
		$(if $(IMAGES),--image-dir $(IMAGES),) $(ARGS)

defect-report:
	PYTHONPATH=. $(PYTHON) scripts/generate_defect_comparison_report.py $(ARGS)

defect-pipeline:
	PYTHONPATH=. $(PYTHON) scripts/generate_defect_masks.py \
		--image-size 512x512 --count 12 --seed 7 $(MASK_ARGS)
	PYTHONPATH=. $(PYTHON) scripts/run_diffusion_pipeline.py $(INPAINT_ARGS)
	PYTHONPATH=. $(PYTHON) scripts/generate_defect_comparison_report.py $(REPORT_ARGS)

## Бонус трека C (vlm-assistants): production VLM-сервис (FastAPI), src/api/.
## Нужны docker-compose.jetson.yml/Dockerfile.jetson (скопированы) -- build-jetson
## реально собирается только на arm64/Jetson или с buildx+QEMU.
serve:
	PYTHONPATH=. $(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 $(ARGS)

build-jetson:
	docker compose -f docker-compose.jetson.yml build

## Трек D (robotics-ot): капстоун -- CV+VLM+манипулятор+k3s.
## Оригинальный "day 2" агрегатор памяти (generate_capstone_memory_report.py)
## не перенесён в этот репозиторий -- он честно читает реальный
## monitoring/jetson/docker-compose.exporters.yml недели 10 (эксплуатация
## и надёжность), которая не входит ни в один из 4 треков этого
## портфолио; переносить его без monitoring/ значило бы либо тащить
## несвязанный пятый пакет, либо дать скрипту молча читать то, чего
## здесь нет.
scenario-color-sorting:
	PYTHONPATH=. $(PYTHON) scripts/run_scenario_demo.py color_sorting $(ARGS)

scenario-defect-detection:
	PYTHONPATH=. $(PYTHON) scripts/run_scenario_demo.py defect_detection $(ARGS)

validate-k3s-manifests:
	PYTHONPATH=. $(PYTHON) scripts/validate_k3s_manifests.py $(ARGS)

capstone-demo: validate-k3s-manifests
	PYTHONPATH=. $(PYTHON) scripts/run_scenario_demo.py color_sorting $(SCENARIO_ARGS)
	PYTHONPATH=. $(PYTHON) scripts/run_scenario_demo.py defect_detection $(SCENARIO_ARGS)
	@echo "капстоун воспроизведён -- см. outputs/scenarios_demo/"

## Юнит-тесты и линт -- только для файлов этого репозитория (src/, tests/, scripts/);
## у pipeline/, dataset_pipeline/, modern_detectors/, quantization/, distillation/,
## captioning/, rag/, ros2_ws/, compliance/ -- свои Makefile с `make test` внутри папки.
test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts
