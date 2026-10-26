"""Process-wide singletons for the FastAPI app (config + inference engine)."""

from __future__ import annotations

import os
from functools import lru_cache

from src.api.metrics import MetricsCollector, TegrastatsSampler
from src.config import AppConfig, load_config
from src.models.base import VLMInference
from src.models.factory import build_inference

DEFAULT_CONFIG_PATH = "config/model_config.yaml"
"""Fallback when neither an explicit path nor `$CONFIG_PATH` is given."""


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> AppConfig:
    """Load and cache the application config.

    Resolution order: `config_path` if given, else the `CONFIG_PATH`
    environment variable, else `DEFAULT_CONFIG_PATH`. The env var is what
    `docker-compose.jetson.yml` sets to point a built image at
    `config/model_config.jetson.yaml` (fp16/cuda) without baking a
    deployment-specific path into the image or the source tree -- the
    same image serves dev (`config/model_config.yaml`, fp32/cpu) and
    Jetson (fp16/cuda) depending only on which config file the
    environment hands it.

    Args:
        config_path: Path to the YAML config file; overrides `CONFIG_PATH`
            when given (mainly for tests).

    Returns:
        The parsed `AppConfig`, cached for subsequent calls.
    """
    return load_config(config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))


@lru_cache(maxsize=1)
def get_inference() -> VLMInference:
    """Build and cache the process-wide `VLMInference` singleton.

    Delegates backend selection to `src.models.factory.build_inference`
    (`config.model.backend`: `"pytorch"` default, `"tensorrt"`, `"auto"` --
    see `src.config.ModelSection`). For the default `"pytorch"` backend this
    is unchanged from before `build_inference` existed: a `CLIPInference`
    bound to a `CLIPLoader`, with the underlying model loaded lazily on
    first `encode_image`/`encode_text` call, not when this function runs.

    Returns:
        A `VLMInference` built from config.
    """
    config = get_config()
    return build_inference(config)


@lru_cache(maxsize=1)
def get_metrics_collector() -> MetricsCollector:
    """Build and cache the process-wide `MetricsCollector` singleton (week 08 day 4).

    Returns:
        A `MetricsCollector`, fed by `app.py`'s request-logging middleware
        and read by `GET /metrics`/`GET /metrics.json`.
    """
    return MetricsCollector()


@lru_cache(maxsize=1)
def get_tegrastats_sampler() -> TegrastatsSampler:
    """Build and cache the process-wide `TegrastatsSampler` singleton (week 08 day 4).

    Constructing it is cheap and side-effect-free (it only checks
    `shutil.which("tegrastats")`), so this is safe to call from
    `/metrics`/`/metrics.json` even if `app.py`'s `lifespan` never ran
    (e.g. under `TestClient` without a `with` block) -- only `.start()`ing
    it actually spawns the background process.

    Returns:
        A `TegrastatsSampler`, started/stopped by `app.py`'s `lifespan`.
    """
    return TegrastatsSampler()
