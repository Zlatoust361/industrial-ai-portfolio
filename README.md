# Промышленное компьютерное зрение + VLM — портфолио по трекам

[![tests](https://github.com/Zlatoust361/industrial-ai-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/Zlatoust361/industrial-ai-portfolio/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Для нанимающего менеджера — 3 минуты.** Роль, на которую это
> портфолио: **архитектор решений / техлид направления «АСУ ТП + AI»** —
> человек, который отвечает за то, чтобы CV/VLM-система заработала на
> реальной линии, а не в ноутбуке. Ниже 4 трека по профилям вакансий;
> выберите свой. Что проверять на собеседовании — в конце каждого трека.
>
> *EN: Solutions architect / tech lead for industrial automation + AI.
> Four tracks (defect-detection CV, edge AI on Jetson, VLM/RAG assistants,
> robotics + PLC integration), each with runnable code, tests in CI and a
> log of real bugs found on real hardware (`HONESTY.md`).*

Этот репозиторий — не курс и не весь проект, а выборка рабочего кода из
более крупной практики по промышленному CV/VLM/робототехнике,
разложенная по профилям вакансий, а не по хронологии.

## За что я отвечаю в этом коде

За результат на железе. Моя работа здесь — архитектура задачи, карта
решений, критерии приёмки, ревью и диагностика на реальном окружении
(Jetson Orin Nano, dev-машина, реальная камера). Реализация собирается с
Claude Code и инженерами под моим контролем — в 2026 это норма для
сильных команд, и я не прячу инструмент. Что это значит на практике,
видно в [`HONESTY.md`](./HONESTY.md): каждая строка там — баг, который я
нашёл сам, прогоняя код на настоящей машине, с диагнозом и фиксом; и
каждое «не проверено на физическом Jetson» написано словами, а не
скрыто за зелёным демо.

Если вакансия — «писать production-код руками в одиночку», это не моё
портфолио. Если вакансия — «довести систему до работы на линии и
отвечать за это» — читайте дальше.

## Треки

### A. CV для дефектоскопии и контроля качества
Для вакансий: инженер компьютерного зрения для дефектоскопии/контроля
качества, ML-инженер промышленного зрения, НК-компании, QA в
металлургии/нефтегазе.

- [`src/models/zero_shot.py`](./src/models/zero_shot.py), [`scripts/clip_classifier.py`](./scripts/clip_classifier.py) — zero-shot классификация дефектов по тексту (CLIP), без обучения
- [`pipeline/`](./pipeline/) — промышленный pipeline визуального контроля, 7 стадий, с quality gate (`GREEN`/`YELLOW`/`RED`)
- [`src/synthetic/`](./src/synthetic/), [`scripts/generate_defect_masks.py`](./scripts/generate_defect_masks.py) — генерация синтетических дефектов (inpainting), реальные PSNR/SSIM-метрики качества
- [`dataset_pipeline/`](./dataset_pipeline/) — конвейер синтетических датасетов, 8 стадий с gate целостности
- [`modern_detectors/`](./modern_detectors/) — RT-DETR/YOLO обучены с нуля, честно сравнены одной методикой измерения

Запуск: `make classify IMAGES=...` / `cd pipeline && make demo` / `cd dataset_pipeline && make pipeline` / `cd modern_detectors && make demo`.

*Спросите меня:* почему `AP=0.0` у недообученного детектора оказался артефактом порога `conf=0.25`, а не провалом модели; почему наивный срез train/val положил весь класс в val; что значит quality gate `YELLOW` для оператора линии.

### B. Edge AI и встраиваемые системы
Для вакансий: Edge AI engineer, MLOps для встраиваемых систем, инженер
оптимизации моделей под слабое железо (Jetson и аналоги).

- [`src/models/clip.py`](./src/models/clip.py), [`scripts/01_download_models.py`](./scripts/01_download_models.py)–[`03_benchmark.py`](./scripts/03_benchmark.py) — CLIP на edge: ONNX-экспорт, бюджет памяти, Jetson-бенчмарк по power-режимам
- [`quantization/`](./quantization/) — INT4-квантование + TensorRT, 8-шаговый пайплайн
- [`distillation/`](./distillation/) — три способа сжать модель с нуля (дистилляция/прунинг/NAS/INT8), честно сравненные

Запуск: `make download-models && make convert && make benchmark` / `cd quantization && make demo` / `cd distillation && make demo`.

*Спросите меня:* почему `AdaptiveAvgPool2d` схлопывал модель к class prior на мелких дефектах и чем это лечится; почему `numpy<2` прибит гвоздями именно из-за Jetson; что из этого трека реально прогнано на физической плате, а что нет (ответ есть в `HONESTY.md`).

### C. VLM и мультимодальные ассистенты
Для вакансий: ML-инженер LLM/VLM, разработчик AI-ассистентов,
multimodal RAG.

- [`captioning/`](./captioning/) — captioning/VQA на edge-модели (SmolVLM/Idefics3)
- [`rag/`](./rag/) — RAG поверх регламентов, с реально найденным и исправленным багом токенизации длинного текста
- бонус: [`src/api/`](./src/api/) — production-обвязка (FastAPI, `make serve`)

Запуск: `cd captioning && make demo` / `cd rag && make demo`.

*Спросите меня:* почему RAG падал на длинной секции регламента и почему фикс `truncation=True` — компромисс, а не решение; какой `<image>`-плейсхолдер требует Idefics3 и как это нашлось.

### D. Робототехника и АСУ ТП с AI-слоем
Для вакансий: инженер робототехники, интегратор АСУ ТП + AI,
консультант по безопасности/аудиту для производства.

- [`ros2_ws/`](./ros2_ws/) — ROS2-узел детекции по открытому словарю (OWL-ViT)
- [`src/scenarios/`](./src/scenarios/), [`src/robot/manipulator.py`](./src/robot/manipulator.py), [`k3s/`](./k3s/) — промышленный капстоун: CV + VLM + манипулятор + k3s-деплой
- [`assistant/`](./assistant/) — ровно та точка стыка: чат-ассистент на VLM, который говорит и с роботом (`ros_interface.py`), и с ПЛК (`opc_ua_bridge.py`) через один `action_executor.py`, а не два несвязанных модуля
- [`compliance/`](./compliance/) — аудит безопасности/лицензий, реальный честный результат (не все пункты зелёные)

Запуск: `cd ros2_ws && colcon build` / `make scenario-color-sorting` / `cd assistant && make chat-serve` / `cd compliance && make demo`.

*Спросите меня:* почему ассистент говорит с роботом и с ПЛК через один `action_executor.py`, а не два модуля; почему чек-лист `compliance/` дал 1 closed / 9 open и почему это честнее, чем «всё зелёное»; что ROS2 выполняется под системным Python и чем это ломает venv.

## Как это читать

[`HONESTY.md`](./HONESTY.md) — реальные баги, найденные при прогоне на
настоящем окружении (sandbox + dev-машина), по каждому треку — не
гипотетический список рисков. Каждая папка трека (`pipeline/`,
`quantization/`, `captioning/`, `ros2_ws/` и т. д.) несёт собственный
README и Makefile с `make test`/`make demo` — тесты реально проходят,
это не витринный код.

## Структура и общий код

`src/`, `scripts/`, `tests/`, `config/` в корне — общий пакет, которым
пользуются несколько недель/треков сразу (CLIP-инференс, конфиг,
память, сценарии капстоуна, production API). `pipeline/`,
`dataset_pipeline/`, `modern_detectors/`, `quantization/`,
`distillation/`, `captioning/`, `rag/`, `ros2_ws/`, `compliance/`,
`assistant/` — самостоятельные пакеты со своими зависимостями и
тестами. `classical_cv/` и `tracking/` — две вспомогательные
библиотеки (синтетический генератор дефектов для демо-датасетов;
сглаживание детекций), от которых зависят `modern_detectors/`,
`distillation/` и `compliance/` — без них их тесты не соберутся.
`docs/` — тестовая фикстура для RAG (`rag/`), не документация этого
репозитория. `make setup` в корне ставит общие зависимости
(`requirements.txt`/`requirements-dev.txt`); у пакетов с собственным
`Makefile` — своя `make setup` при необходимости отдельных зависимостей.

**Как проверялось**: каждый пакет выше прогоняется в CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) на каждый push,
двумя джобами — и это не удобство, а следствие того, как написаны тесты:

- `core` — среда **без** torch (`requirements-ci-core.txt`): root,
  `pipeline/`, `dataset_pipeline/`, `tracking/`, `assistant/`,
  `quantization/`. Их тесты `*_reaches_torch_boundary` утверждают, что
  torch отсутствует, и в среде с torch падают — поэтому ML-стек сюда
  не ставится. Единственный тест, которому torch нужен
  (`quantization/…/test_run_smoke_test_accepts_fp16_dummy_input`), в этой
  среде пропускается, не падает.
- `ml` — полный стек из `requirements.txt` (torch CPU, transformers,
  ultralytics): `classical_cv/`, `modern_detectors/`, `distillation/`,
  `captioning/`, `rag/`, `compliance/`. Джоба идёт с `HF_HUB_OFFLINE=1`:
  тесты `*_reaches_real_network_boundary` писались в среде без сети до
  `huggingface.co` и утверждают, что загрузка модели не удаётся — с
  живой сетью GitHub Actions они скачали CLIP и упали на обратном
  утверждении (первый прогон CI это и показал).

ROS2 (`ros2_ws/`) в CI не прогоняется — нужен установленный ROS2 Humble;
проверен только синтаксически (`py_compile`). Реальный Jetson,
манипулятор и k3s-кластер в CI, разумеется, не участвуют — что из этого
прогнано на физическом железе, а что нет, по каждому треку сказано в
[`HONESTY.md`](./HONESTY.md).

## Как это обновляется

416 файлов зеркалятся из закрытой монорепы 1:1 скриптом [`.portfolio/sync.py`](.portfolio/sync.py) по манифесту; README, `HONESTY.md` и CI — собственные. Процедура и правила — в [`UPDATING.md`](UPDATING.md).

## Контакты и лицензия

Связаться по вакансии или проекту — через [профиль GitHub](https://github.com/Zlatoust361).
Код — под лицензией MIT, см. [`LICENSE`](LICENSE).
