"""vision_bridge.py -- calls into weeks 1-9's ALREADY-DEPLOYED vision stack (week 11, day 3).

Two genuinely different reuse shapes, for two genuinely different reasons:

- **Embedding + zero-shot classification** (`embed_image_bytes`/`embed_text`/
  `classify_defect`): calls the already-running FastAPI service (week 08's
  `src/api/app.py`, `docker-compose.jetson.yml` maps `8000:8000`) over
  HTTP for the CLIP forward pass, then does the cheap cosine-similarity
  classification math in-process (`src.models.zero_shot`, pure numpy,
  reused directly -- week 02). This process never loads CLIP weights
  itself: an 8 GB Jetson Orin Nano has no room for a second copy of the
  model the deployed service already holds (ADR-002) -- see `README.md`'s
  "Week 11" section.
- **Open-vocabulary object detection** (`detect_objects`): no HTTP
  endpoint exists for this (the deployed service is embedding-only --
  see day 3's own research). Reused directly in-process instead, from
  week 03's `ros2_ws/src/vision_bridge/vision_bridge/detector_factory.py`
  (its `Detector`/`Detection` types are deliberately rclpy-free, importable
  outside a ROS2 workspace -- see that module's own docstring). This is a
  SEPARATE `vision_bridge` from this file: the ROS2 colcon package at
  `ros2_ws/src/vision_bridge/vision_bridge/`, reached via a `sys.path`
  addition, not a Python import of this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from assistant.utils.config import VisionBridgeConfig, load_bridges_config
from src.data.prompts import PromptManager
from src.models.zero_shot import (
    ClassificationResult,
    average_prompt_embedding,
    classify_by_cosine_similarity,
)

_ROS2_VISION_BRIDGE_ROOT = (
    Path(__file__).resolve().parent.parent / "ros2_ws" / "src" / "vision_bridge"
)
"""Week 03's colcon package root -- added to `sys.path` lazily (only when
`detect_objects` is first called), not at import time, so importing THIS
module never has a side effect on `sys.path` for callers who only need
embedding/classification."""


class VisionBridgeError(RuntimeError):
    """Raised when the deployed embedding service can't be reached or errors."""


@dataclass(frozen=True)
class EmbeddingResult:
    """One HTTP call's worth of embedding vectors, as returned by the deployed service."""

    vectors: list[list[float]]
    model_name: str
    dim: int


def _to_embedding_result(body: dict[str, object]) -> EmbeddingResult:
    return EmbeddingResult(vectors=body["vectors"], model_name=body["model_name"], dim=body["dim"])


def _ensure_ros2_vision_bridge_on_path() -> None:
    path_str = str(_ROS2_VISION_BRIDGE_ROOT)
    if path_str not in sys.path:
        if not _ROS2_VISION_BRIDGE_ROOT.is_dir():
            raise VisionBridgeError(
                f"ros2_ws/src/vision_bridge not found at {_ROS2_VISION_BRIDGE_ROOT} -- "
                "can't reuse week 03's detector code."
            )
        sys.path.insert(0, path_str)


class VisionBridge:
    """Calls into the deployed CLIP embedding service + reuses week 03's OWL-ViT detector."""

    def __init__(
        self, config: VisionBridgeConfig | None = None, client: object | None = None
    ) -> None:
        """
        Args:
            config: Defaults to `load_bridges_config().vision_bridge`.
            client: Optional `httpx.Client`-like object (`.post`/`.get`
                methods returning `httpx.Response`-like objects). Defaults
                to the real network via module-level `httpx.post`/`.get` --
                tests inject an `httpx.Client(transport=...)` instead,
                same pattern as `assistant.llm_parser.call_claude_api`
                (day 2) and `src/synthetic/diffusion_pipeline.py`'s
                `StabilityAPIBackend` (week 05).
        """
        self._config = config if config is not None else load_bridges_config().vision_bridge
        self._client = client
        self._prompt_manager: PromptManager | None = None
        self._detector: object | None = None

    def _get(self, path: str) -> object:
        import httpx

        getter = self._client.get if self._client is not None else httpx.get
        response = getter(f"{self._config.base_url}{path}", timeout=self._config.timeout_s)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, **kwargs: object) -> object:
        import httpx

        poster = self._client.post if self._client is not None else httpx.post
        response = poster(
            f"{self._config.base_url}{path}", timeout=self._config.timeout_s, **kwargs
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict[str, object]:
        """`GET /health` on the deployed service.

        Returns:
            `{"reachable": True, "status": "ok", ...}` on success, or
            `{"reachable": False, "error": "..."}` if the service can't
            be reached at all -- this is meant to be one input to
            `action_executor.py`'s `get_status` handler, not something
            that raises and aborts the whole status report.
        """
        try:
            body = self._get("/health")
        except Exception as exc:  # noqa: BLE001 -- ANY reachability failure becomes a
            return {"reachable": False, "error": str(exc)}  # status fact, not a crash
        return {"reachable": True, **body}

    def embed_image_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> EmbeddingResult:
        """`POST /v1/embed/image` on the deployed service.

        Args:
            image_bytes: Raw image file bytes.
            filename: Passed through as the multipart filename (the
                service infers content type from it).

        Returns:
            The real `EmbedResponse` body, typed.

        Raises:
            VisionBridgeError: If the service is unreachable or returns
                a non-2xx response (e.g. 503 -- `torch`/the CLIP weights
                aren't actually loadable on the machine running the
                deployed service either, in this sandbox).
        """
        try:
            body = self._post("/v1/embed/image", files={"file": (filename, image_bytes)})
        except Exception as exc:  # noqa: BLE001
            raise VisionBridgeError(f"embed_image_bytes failed: {exc}") from exc
        return _to_embedding_result(body)

    def embed_text(self, texts: list[str]) -> EmbeddingResult:
        """`POST /v1/embed/text` on the deployed service.

        Args:
            texts: One or more text prompts.

        Returns:
            The real `EmbedResponse` body, typed.

        Raises:
            VisionBridgeError: If the service is unreachable or errors.
        """
        try:
            body = self._post("/v1/embed/text", json={"texts": texts})
        except Exception as exc:  # noqa: BLE001
            raise VisionBridgeError(f"embed_text failed: {exc}") from exc
        return _to_embedding_result(body)

    def _get_prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager.from_yaml("config/prompts.yaml")
        return self._prompt_manager

    def classify_by_prompts(
        self,
        image_bytes: bytes,
        phrases_by_class: dict[str, list[str]],
        filename: str = "image.jpg",
    ) -> ClassificationResult:
        """Zero-shot classification against an arbitrary set of text-described classes.

        The general form `classify_defect` (below) is one fixed instance
        of: embed the image once, embed each class's reference phrases,
        average multi-phrase classes into one vector each (week 02's
        prompt-ensembling), then cosine-similarity-rank
        (`src.models.zero_shot`, pure numpy, not reimplemented here).
        Pulled out as its own method (week 12, day 3) because
        `scenarios/color_sorting.py`/`defect_detection.py` need this same
        shape for genuinely different class sets (colors; scratch/pit/
        crack) that don't come from `config/prompts.yaml`'s fixed
        ok-vs-defect styles.

        Args:
            image_bytes: Raw image file bytes.
            phrases_by_class: Class label -> one or more reference
                phrases for that class (more than one enables prompt
                ensembling; a single phrase is also valid).
            filename: Passed through to `embed_image_bytes`.

        Returns:
            A `ClassificationResult` over exactly `phrases_by_class`'s labels.

        Raises:
            VisionBridgeError: If either embedding call fails.
            ValueError: If `phrases_by_class` is empty (see
                `classify_by_cosine_similarity`).
        """
        image_result = self.embed_image_bytes(image_bytes, filename=filename)

        class_vectors: dict[str, np.ndarray] = {}
        for label, phrases in phrases_by_class.items():
            phrase_result = self.embed_text(phrases)
            class_vectors[label] = average_prompt_embedding(np.array(phrase_result.vectors))

        image_vector = np.array(image_result.vectors[0])
        return classify_by_cosine_similarity(image_vector, class_vectors)

    def classify_defect(
        self, image_bytes: bytes, filename: str = "image.jpg"
    ) -> ClassificationResult:
        """Zero-shot ok-vs-defect classification (week 02), image embedded via the deployed service.

        Args:
            image_bytes: Raw image file bytes.
            filename: Passed through to `embed_image_bytes`.

        Returns:
            A `ClassificationResult` (`predicted_class` is `"ok"` or
            `"defect"` for `config/prompts.yaml`'s default styles).

        Raises:
            VisionBridgeError: If either embedding call fails.
        """
        prompt_set = self._get_prompt_manager().get(self._config.prompt_style)
        return self.classify_by_prompts(image_bytes, prompt_set.phrases_by_class, filename=filename)

    def detect_objects(self, image_rgb: np.ndarray, text_query: str) -> list[object]:
        """Open-vocabulary detection, reusing week 03's OWL-ViT detector in-process.

        No HTTP endpoint exists for this (the deployed service is
        embedding-only) -- unlike the methods above, this one loads a
        model in THIS process if `config.detector_backend` isn't `"stub"`.
        On an 8 GB Jetson already running the CLIP embedding service,
        that's a real memory-budget tradeoff a caller needs to make
        deliberately, not something this bridge hides.

        Args:
            image_rgb: `(height, width, 3)` uint8 RGB image array.
            text_query: Open-vocabulary text query, e.g. `"a photo of a
                scratch on metal"`.

        Returns:
            Zero or more `vision_bridge.detection.Detection` (week 03's
            type -- `label`/`score`/`bbox`), from the ROS2 package, not a
            reimplementation.

        Raises:
            VisionBridgeError: If `ros2_ws/src/vision_bridge` isn't found
                on disk.
            ImportError: If `config.detector_backend` needs `torch`/
                `transformers`/`tensorrt` and they aren't installed (this
                sandbox has none -- see README.md).
        """
        _ensure_ros2_vision_bridge_on_path()
        from vision_bridge.detector_factory import build_detector

        if self._detector is None:
            self._detector = build_detector(
                backend=self._config.detector_backend,
                engine_path=None,
                model_name=self._config.detector_model_name,
                score_threshold=self._config.detector_score_threshold,
            )
        return self._detector.detect(image_rgb, text_query)
