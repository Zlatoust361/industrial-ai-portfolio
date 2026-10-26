"""Inpainting backends (week 05, day 3): mask -> defect image.

Three backends behind one `InpaintingBackend` protocol, selected by
`src.synthetic.diffusion_config.DiffusionConfig.backend` via
`build_backend()`:

- `Cv2InpaintBackend` -- OpenCV's classical (non-diffusion) inpainting.
  Real, working code, needs nothing this sandbox doesn't already have
  (`opencv-python-headless`, already in `requirements.txt`). This is
  what actually runs here and in CI -- see this module's own docstring
  section in `README.md`'s "Week 05" for why that's an honest
  substitution, not day 3's literal ask.
- `DiffusersInpaintBackend` -- real Stable Diffusion inpainting via
  HuggingFace `diffusers`, on a local CUDA dev GPU. Lazy-imports
  `torch`/`diffusers` (NOT in `requirements.txt` -- same reasoning as
  `tensorrt`/`pycuda` in `quantization/`: a large, environment-specific
  dependency nobody needs just to run this repo's test suite). Refuses to
  run without a real CUDA device -- see its own docstring for why.
- `StabilityAPIBackend` -- real Stable Diffusion inpainting via Stability
  AI's REST API, no local GPU needed. Lazy-imports `httpx` (also not in
  `requirements.txt`). Needs a real API key this session doesn't have and
  isn't authorized to spend -- written correctly, exercised only against
  a mocked HTTP transport in tests.

None of the three is Jetson-specific code (nothing here has a Jetson code
path at all -- day 3's device requirement is "dev-GPU, not Jetson" in the
sense that generation should simply never be attempted there, not that
there's Jetson-specific logic to gate). See `README.md`'s "Week 05"
section for the full "why not Jetson" reasoning.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.synthetic.diffusion_config import (
    Cv2BackendConfig,
    DiffusersBackendConfig,
    DiffusionConfig,
    StabilityAPIBackendConfig,
)


class InpaintingBackend(Protocol):
    """One inpainting backend -- mask a region of `image`, fill it in."""

    name: str

    def inpaint(
        self, image: Image.Image, mask: np.ndarray, prompt: str, negative_prompt: str
    ) -> Image.Image:
        """Return a new RGB image with `mask`'s non-zero region filled in.

        Args:
            image: The base image, any PIL mode (converted to RGB internally).
            mask: `(height, width)` `uint8` array, non-zero = region to fill
                (see `src.synthetic.mask_generation.generate_defect_mask`).
            prompt: Positive text prompt (ignored by backends with no text
                conditioning, e.g. `Cv2InpaintBackend`).
            negative_prompt: Negative text prompt (same caveat).

        Returns:
            A new RGB `PIL.Image`, same size as `image`.
        """
        ...


@dataclass
class Cv2InpaintBackend:
    """OpenCV's classical inpainting (Telea or Navier-Stokes) -- no diffusion, no GPU.

    `cv2.inpaint` fills the masked region by propagating nearby texture
    inward -- fast, deterministic, no text conditioning (`prompt`/
    `negative_prompt` are accepted for interface compatibility and
    ignored). Real algorithm, real output -- not a mock -- just not the
    diffusion model day 3 actually asks for; see this module's docstring.
    """

    name = "cv2"
    method: str = "telea"
    radius: int = 3

    def __post_init__(self) -> None:
        if self.method not in ("telea", "ns"):
            raise ValueError(f"method must be 'telea' or 'ns', got {self.method!r}.")

    def inpaint(
        self, image: Image.Image, mask: np.ndarray, prompt: str = "", negative_prompt: str = ""
    ) -> Image.Image:
        flag = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(bgr, mask, self.radius, flag)
        return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))


class DiffusersInpaintBackend:
    """Real Stable Diffusion inpainting via HuggingFace `diffusers`, on a CUDA dev GPU.

    Never usable in this sandbox (no `torch`/`diffusers`, no GPU) and
    never usable on a Jetson Orin Nano -- not because of any Jetson-specific
    code here, but because a single 512x512 SD inpainting call takes
    multiple seconds to tens of seconds even on a real dev GPU (dozens of
    diffusion steps, each a full U-Net forward pass) and would be far
    worse on the Orin Nano's shared 8 GB memory with no dedicated VRAM --
    unacceptable for the iterative "generate a batch, look at it, adjust
    defects.yaml/prompt, regenerate" workflow this pipeline exists for.
    See `README.md`'s "Week 05" section.
    """

    name = "diffusers"

    def __init__(self, config: DiffusersBackendConfig) -> None:
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:
            raise ImportError(
                "DiffusersInpaintBackend needs `pip install torch diffusers accelerate` "
                "on a CUDA-capable dev machine -- deliberately NOT in requirements.txt "
                "(same reasoning as tensorrt/pycuda in quantization/: a large, "
                "environment-specific dependency nobody needs just to run this repo's "
                "test suite). See README.md's 'Week 05' section."
            ) from exc

        if config.device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError(
                "DiffusersInpaintBackend requires a real CUDA dev GPU -- refusing to run "
                "on CPU (config.device='cpu' or torch.cuda.is_available() is False). A "
                "single Stable Diffusion inpaint on CPU can take minutes; running this "
                "pipeline anywhere without real GPU acceleration defeats the point of "
                "iterative synthetic-defect generation. See README.md's 'Week 05' section "
                "for the same reasoning applied to why this never runs on a Jetson Orin Nano."
            )

        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_id, torch_dtype=torch.float16
        ).to(config.device)
        self._generator = torch.Generator(device=config.device).manual_seed(config.seed)
        self._steps = config.num_inference_steps
        self._guidance = config.guidance_scale

    def inpaint(
        self, image: Image.Image, mask: np.ndarray, prompt: str, negative_prompt: str
    ) -> Image.Image:
        mask_image = Image.fromarray(mask, mode="L")
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image.convert("RGB"),
            mask_image=mask_image,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
            generator=self._generator,
        )
        return result.images[0]


class StabilityAPIBackend:
    """Real Stable Diffusion inpainting via Stability AI's REST API -- no local GPU needed.

    Lazy-imports `httpx` (not in `requirements.txt` -- see this module's
    docstring). Needs `config.api_key_env` set to a real API key; this
    session has neither one nor standing authorization to spend a real
    API budget on the user's behalf, so this backend is exercised only
    against a mocked HTTP transport in tests, never against the real
    endpoint. Correct usage, per Stability AI's v2beta inpaint API:
    multipart-form POST with the base image + mask + prompt, an image
    response back.
    """

    name = "stability_api"

    def __init__(self, config: StabilityAPIBackendConfig, client: object | None = None) -> None:
        """
        Args:
            config: `stability_api:` section of `diffusion.yaml`.
            client: Optional `httpx.Client`-like object (anything with a
                `.post(url, headers=, files=, data=, timeout=)` method
                returning an `httpx.Response`-like object). Defaults to
                the real network via `httpx.post` -- tests inject an
                `httpx.Client(transport=httpx.MockTransport(...))` here
                instead, so this backend is exercised for real against a
                fake HTTP layer rather than mocked at the method level.
        """
        self._config = config
        self._client = client

    def inpaint(
        self, image: Image.Image, mask: np.ndarray, prompt: str, negative_prompt: str
    ) -> Image.Image:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "StabilityAPIBackend needs `pip install httpx` -- deliberately NOT in "
                "requirements.txt (optional dependency, only needed for this one backend). "
                "See README.md's 'Week 05' section."
            ) from exc

        api_key = os.environ.get(self._config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"${self._config.api_key_env} is not set -- required for "
                "StabilityAPIBackend (config/diffusion.yaml's stability_api.api_key_env)."
            )

        poster = self._client.post if self._client is not None else httpx.post
        response = poster(
            self._config.endpoint,
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files={
                "image": ("image.png", _to_png_bytes(image.convert("RGB")), "image/png"),
                "mask": ("mask.png", _to_png_bytes(Image.fromarray(mask, mode="L")), "image/png"),
            },
            data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "output_format": self._config.output_format,
            },
            timeout=self._config.timeout_s,
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")


def _to_png_bytes(image: Image.Image) -> bytes:
    """Serialize `image` to in-memory PNG bytes (for `StabilityAPIBackend`'s multipart upload)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_backend(config: DiffusionConfig) -> InpaintingBackend:
    """Instantiate the backend named by `config.backend`.

    Args:
        config: Parsed `diffusion.yaml`.

    Returns:
        A ready-to-use `InpaintingBackend`.

    Raises:
        ValueError: If `config.backend` isn't a known name (shouldn't
            happen for a config that passed `load_diffusion_config`,
            which already validates this).
        ImportError: From `DiffusersInpaintBackend.__init__` if
            `torch`/`diffusers` aren't installed. `StabilityAPIBackend`
            defers its `httpx` import to `inpaint()` instead, since
            constructing it needs no network/GPU check up front.
        RuntimeError: From `DiffusersInpaintBackend.__init__` if no real
            CUDA device is available.
    """
    if config.backend == "cv2":
        cv2_config: Cv2BackendConfig = config.cv2
        return Cv2InpaintBackend(method=cv2_config.method, radius=cv2_config.radius)
    if config.backend == "diffusers":
        return DiffusersInpaintBackend(config.diffusers)
    if config.backend == "stability_api":
        return StabilityAPIBackend(config.stability_api)
    raise ValueError(f"Unknown backend {config.backend!r}.")


def synthesize_placeholder_surface(image_size: tuple[int, int], seed: int) -> Image.Image:
    """Deterministic textured placeholder base image -- no client data.

    `scripts/run_diffusion_pipeline.py`'s fallback when no `--image-dir`
    of real base images is given: `scripts/generate_defect_masks.py`'s own
    equivalent fallback (blank canvases, week 05 day 2) has no texture at
    all, which would make `quality_assessment`'s PSNR/SSIM/blur/contrast
    numbers on it borderline meaningless (a blank canvas trivially
    "inpaints" to another blank canvas). This generates smoothed Gaussian
    noise instead -- grayscale, brushed-metal-ish, with real local texture
    for the quality metrics to actually measure, while still being
    synthetic (no real facility/product imagery of any kind).

    Args:
        image_size: `(width, height)`.
        seed: Seeds a fresh `np.random.default_rng` -- same `(image_size, seed)`
            always produces the same image.

    Returns:
        An RGB `PIL.Image` of the given size.
    """
    width, height = image_size
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=160.0, scale=25.0, size=(height, width))
    smoothed = np.clip(gaussian_filter(noise, sigma=2.0), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([smoothed] * 3, axis=-1), mode="RGB")
