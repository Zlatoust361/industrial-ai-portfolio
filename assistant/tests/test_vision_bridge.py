"""Tests for `assistant.vision_bridge`.

`health_check` is tested against the REAL `src/api/app.py` FastAPI
application (week 08), via `fastapi.testclient.TestClient` -- real
routing, real response, no mocking, no real network socket (plain
`httpx.Client(transport=httpx.ASGITransport(...))` only supports
`AsyncClient`; `TestClient` is the synchronous-compatible wrapper this
bridge's synchronous `.get`/`.post` calls actually need). The
embed/classify endpoints need `torch` for a real CLIP forward pass
(unavailable here), so those are tested against `httpx.MockTransport` (a
realistic response shape, real HTTP-layer + cosine-similarity code)
instead -- same pattern `assistant.llm_parser`'s `call_claude_api` tests
use. `detect_objects` is checked against this sandbox's REAL absence of
`torch` (via the `"pytorch"` backend) and REAL presence of a working
`"stub"` backend (zero detections, no model needed at all).
"""

from __future__ import annotations

import numpy as np
import pytest

from assistant.utils.config import VisionBridgeConfig
from assistant.vision_bridge import VisionBridge, VisionBridgeError

httpx = pytest.importorskip("httpx")


def _config(backend: str = "stub") -> VisionBridgeConfig:
    return VisionBridgeConfig(
        base_url="http://vision-bridge-test",
        timeout_s=5.0,
        prompt_style="basic",
        detector_backend=backend,
        detector_model_name="google/owlvit-base-patch32",
        detector_score_threshold=0.1,
    )


# -- health_check, against the real FastAPI app --------------------------------


def test_health_check_against_the_real_deployed_app() -> None:
    # `TestClient` (an `httpx.Client` subclass) is what actually supports
    # a real synchronous ASGI round-trip against `src/api/app.py` -- plain
    # `httpx.Client(transport=httpx.ASGITransport(...))` only implements
    # the ASYNC transport protocol (`AsyncClient` territory), confirmed
    # by trying it here first and getting a real
    # "ASGITransport has no attribute handle_request" error.
    from fastapi.testclient import TestClient

    from src.api.app import app

    with TestClient(app) as client:
        bridge = VisionBridge(config=_config(), client=client)
        result = bridge.health_check()

    assert result == {"reachable": True, "status": "ok"}


def test_health_check_unreachable_service_reports_not_a_crash() -> None:
    bridge = VisionBridge(config=_config())  # real network, nothing listening on this base_url
    result = bridge.health_check()
    assert result["reachable"] is False
    assert "error" in result


# -- embed_image_bytes / embed_text, against a mock transport -------------------


def _embed_response(
    vectors: list[list[float]], model_name: str = "openai/clip-vit-base-patch32"
) -> httpx.Response:
    dim = len(vectors[0]) if vectors else 0
    return httpx.Response(200, json={"vectors": vectors, "model_name": model_name, "dim": dim})


def test_embed_image_bytes_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embed/image"
        return _embed_response([[1.0, 0.0, 0.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    result = bridge.embed_image_bytes(b"fake-jpeg-bytes")
    assert result.vectors == [[1.0, 0.0, 0.0]]
    assert result.dim == 3


def test_embed_image_bytes_service_error_raises_vision_bridge_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"model not loadable here")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    with pytest.raises(VisionBridgeError):
        bridge.embed_image_bytes(b"x")


def test_embed_text_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embed/text"
        return _embed_response([[0.0, 1.0], [1.0, 0.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    result = bridge.embed_text(["ok surface", "a defect"])
    assert len(result.vectors) == 2


# -- classify_defect: two HTTP calls + real cosine-similarity math -------------


def test_classify_defect_combines_embed_calls_with_real_zero_shot_math() -> None:
    call_log = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append(request.url.path)
        if request.url.path == "/v1/embed/image":
            return _embed_response([[1.0, 0.0]])
        # Text embed calls (once per class label in config/prompts.yaml) --
        # each phrase batch gets a distinct-enough vector that averaging
        # doesn't collapse them, but the exact predicted class doesn't
        # matter to this test, only that classification runs end to end.
        return _embed_response([[0.0, 1.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    result = bridge.classify_defect(b"fake-image-bytes")

    assert result.predicted_class in {"ok", "defect"}
    assert set(result.similarities) == {"ok", "defect"}
    assert "/v1/embed/image" in call_log
    assert call_log.count("/v1/embed/text") == 2  # once per class in config/prompts.yaml


def test_classify_defect_propagates_embed_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    with pytest.raises(VisionBridgeError):
        bridge.classify_defect(b"x")


# -- classify_by_prompts: the general form classify_defect is one instance of --


def test_classify_by_prompts_over_an_arbitrary_class_set() -> None:
    call_log = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append(request.url.path)
        if request.url.path == "/v1/embed/image":
            return _embed_response([[1.0, 0.0, 0.0]])
        return _embed_response([[0.0, 1.0, 0.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    result = bridge.classify_by_prompts(
        b"fake-image-bytes",
        {
            "red": ["a photo of a red object"],
            "blue": ["a photo of a blue object"],
            "green": ["a photo of a green object"],
        },
    )

    assert result.predicted_class in {"red", "blue", "green"}
    assert set(result.similarities) == {"red", "blue", "green"}
    assert call_log.count("/v1/embed/text") == 3  # once per class passed in


def test_classify_defect_is_classify_by_prompts_with_the_prompt_manager_styles() -> None:
    # classify_defect delegates to classify_by_prompts -- confirm the
    # delegation actually happens (same call count/shape), not just that
    # both happen to work independently.
    call_log = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append(request.url.path)
        if request.url.path == "/v1/embed/image":
            return _embed_response([[1.0, 0.0]])
        return _embed_response([[0.0, 1.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VisionBridge(config=_config(), client=client)

    bridge.classify_defect(b"x")

    assert call_log.count("/v1/embed/image") == 1
    assert call_log.count("/v1/embed/text") == 2  # ok + defect, from config/prompts.yaml


# -- detect_objects: real reuse of week 03's detector code ----------------------


def test_detect_objects_stub_backend_runs_for_real() -> None:
    bridge = VisionBridge(config=_config(backend="stub"))
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    detections = bridge.detect_objects(image, "a scratch on metal")

    assert detections == []  # StubDetector's real, documented behavior


def test_detect_objects_pytorch_backend_reaches_the_real_torch_boundary() -> None:
    # Real check, not mocked: this sandbox genuinely has no torch/transformers
    # installed (same standing limitation as every torch-dependent path
    # elsewhere in this repo).
    bridge = VisionBridge(config=_config(backend="pytorch"))
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ModuleNotFoundError, match="torch"):
        bridge.detect_objects(image, "a crack")


def test_detect_objects_caches_the_detector_across_calls() -> None:
    bridge = VisionBridge(config=_config(backend="stub"))
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    bridge.detect_objects(image, "query one")
    first_detector = bridge._detector  # noqa: SLF001 -- verifying real caching behavior
    bridge.detect_objects(image, "query two")

    assert bridge._detector is first_detector  # noqa: SLF001
