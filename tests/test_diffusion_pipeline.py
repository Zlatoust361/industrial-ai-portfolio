"""Unit tests for `src.synthetic.diffusion_pipeline`.

`Cv2InpaintBackend` is exercised for real (no mocking) -- it needs
nothing this sandbox doesn't already have. `DiffusersInpaintBackend` is
checked against this sandbox's REAL absence of `torch`/`diffusers` (not
mocked -- the same "smoke test to the exact expected boundary" pattern
`quantization/`'s tests use for `tensorrt`/`pycuda`). `StabilityAPIBackend`
is exercised against a real `httpx.MockTransport`-backed client (real
HTTP-layer code path, fake network) since a real API key/network call
isn't available or authorized here.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.synthetic.diffusion_config import (
    Cv2BackendConfig,
    DiffusersBackendConfig,
    DiffusionConfig,
    PromptConfig,
    QualityGateConfig,
    StabilityAPIBackendConfig,
)
from src.synthetic.diffusion_pipeline import (
    Cv2InpaintBackend,
    StabilityAPIBackend,
    build_backend,
    synthesize_placeholder_surface,
)

httpx = pytest.importorskip("httpx")


def _base_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    return synthesize_placeholder_surface(size, seed=0)


def _mask_with_bbox(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 255
    return mask


def _config(backend: str = "cv2") -> DiffusionConfig:
    return DiffusionConfig(
        backend=backend,
        prompt=PromptConfig(positive="a defect", negative="bad quality"),
        cv2=Cv2BackendConfig(method="telea", radius=3),
        diffusers=DiffusersBackendConfig(
            model_id="some/model", device="cuda", num_inference_steps=1, guidance_scale=1.0, seed=0
        ),
        stability_api=StabilityAPIBackendConfig(
            endpoint="https://example.invalid/inpaint",
            api_key_env="TEST_STABILITY_API_KEY",
            output_format="png",
            timeout_s=5.0,
        ),
        quality_gate=QualityGateConfig(keep_fraction=0.75, weights={}, ssim_window=7),
    )


# -- Cv2InpaintBackend --------------------------------------------------------


def test_cv2_backend_fills_the_masked_region() -> None:
    backend = Cv2InpaintBackend(method="telea", radius=3)
    image = _base_image()
    mask = _mask_with_bbox((64, 64), (20, 20, 40, 40))

    result = backend.inpaint(image, mask, prompt="ignored", negative_prompt="ignored")

    assert result.size == image.size
    assert result.mode == "RGB"


def test_cv2_backend_leaves_pixels_outside_the_mask_untouched() -> None:
    backend = Cv2InpaintBackend()
    image = _base_image()
    mask = _mask_with_bbox((64, 64), (20, 20, 40, 40))

    result_rgb = np.array(backend.inpaint(image, mask, "", ""))
    original_rgb = np.array(image)
    outside = mask == 0
    assert np.array_equal(result_rgb[outside], original_rgb[outside])


def test_cv2_backend_actually_changes_pixels_inside_the_mask() -> None:
    backend = Cv2InpaintBackend()
    image = _base_image()
    mask = _mask_with_bbox((64, 64), (20, 20, 40, 40))

    result_rgb = np.array(backend.inpaint(image, mask, "", ""))
    original_rgb = np.array(image)
    inside = mask != 0
    # cv2.inpaint fills from surrounding texture -- won't be pixel-identical
    # to the original noise inside the mask (which had no correlation with
    # its neighbors), for a large enough masked region.
    assert not np.array_equal(result_rgb[inside], original_rgb[inside])


def test_cv2_backend_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        Cv2InpaintBackend(method="not-a-method")


# -- build_backend -------------------------------------------------------------


def test_build_backend_cv2_returns_working_backend() -> None:
    backend = build_backend(_config(backend="cv2"))
    assert isinstance(backend, Cv2InpaintBackend)
    assert backend.name == "cv2"


def test_build_backend_diffusers_raises_importerror_in_this_sandbox() -> None:
    # Real check, not mocked: this sandbox genuinely has no torch/diffusers
    # installed (same standing limitation as every torch-dependent path
    # elsewhere in this repo).
    with pytest.raises(ImportError, match="diffusers"):
        build_backend(_config(backend="diffusers"))


def test_build_backend_stability_api_construction_needs_no_network() -> None:
    backend = build_backend(_config(backend="stability_api"))
    assert isinstance(backend, StabilityAPIBackend)
    assert backend.name == "stability_api"


def test_build_backend_unknown_name_raises() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="Unknown backend"):
        build_backend(replace(_config(), backend="not-a-backend"))


# -- StabilityAPIBackend -------------------------------------------------------


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_stability_api_backend_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_STABILITY_API_KEY", raising=False)
    backend = StabilityAPIBackend(_config().stability_api)
    image = _base_image()
    mask = _mask_with_bbox((64, 64), (10, 10, 20, 20))

    with pytest.raises(RuntimeError, match="TEST_STABILITY_API_KEY"):
        backend.inpaint(image, mask, "prompt", "negative")


def test_stability_api_backend_real_http_layer_against_a_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_STABILITY_API_KEY", "fake-key-for-tests")
    expected_image = _base_image()
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        assert request.headers["authorization"] == "Bearer fake-key-for-tests"
        return httpx.Response(200, content=_png_bytes(expected_image))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = StabilityAPIBackend(_config().stability_api, client=client)
    mask = _mask_with_bbox((64, 64), (10, 10, 20, 20))

    result = backend.inpaint(_base_image(), mask, "a scratch", "cartoon")

    assert len(captured_requests) == 1
    assert result.size == expected_image.size
    assert np.array_equal(np.array(result), np.array(expected_image))


def test_stability_api_backend_raises_on_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_STABILITY_API_KEY", "fake-key-for-tests")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = StabilityAPIBackend(_config().stability_api, client=client)
    mask = _mask_with_bbox((64, 64), (10, 10, 20, 20))

    with pytest.raises(httpx.HTTPStatusError):
        backend.inpaint(_base_image(), mask, "a scratch", "cartoon")


# -- synthesize_placeholder_surface --------------------------------------------


def test_synthesize_placeholder_surface_is_deterministic() -> None:
    a = synthesize_placeholder_surface((32, 24), seed=7)
    b = synthesize_placeholder_surface((32, 24), seed=7)
    assert np.array_equal(np.array(a), np.array(b))


def test_synthesize_placeholder_surface_different_seed_differs() -> None:
    a = synthesize_placeholder_surface((32, 24), seed=7)
    b = synthesize_placeholder_surface((32, 24), seed=8)
    assert not np.array_equal(np.array(a), np.array(b))


def test_synthesize_placeholder_surface_has_correct_size_and_mode() -> None:
    image = synthesize_placeholder_surface((40, 30), seed=1)
    assert image.size == (40, 30)
    assert image.mode == "RGB"
