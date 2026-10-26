"""Parametric 2D scene simulation (week 07): camera/light/material/noise.

`render_scene(object_rgba, config, rng=None)` places an object cutout onto
a background and puts the composited frame through camera (scale/rotation/
position), light (brightness/contrast/white-balance), material (hue/
saturation, object only) and noise (blur/gaussian) transforms.

`rng=None` is the `no_dr` branch: every parameter is held at its neutral/
identity value (native scale, centered, no lighting/material change, no
noise) -- a single fixed, deterministic scene, not "no randomization
applied to otherwise-random values". `rng=<Generator>` (`with_dr`) samples
every range in `config` uniformly. Same renderer, same `parameters.yaml`,
two branches -- see `scripts/render_dr_dataset.py --branch {no_dr,with_dr}`.

Pure `numpy` + `PIL` -- no `torch`, cheap enough to run anywhere (including
the Jetson, unlike the YOLOv8 training week 07 day 3 does on this data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.domain_randomization.config import (
    BackgroundConfig,
    CameraConfig,
    DRParameters,
    LightConfig,
    MaterialConfig,
    NoiseConfig,
)

Bbox = tuple[int, int, int, int]
"""`(x_min, y_min, x_max, y_max)` pixel bounding box -- half-open, i.e.
`image[y_min:y_max, x_min:x_max]` covers it exactly (numpy-slice convention)."""

# Neutral/identity values for the `no_dr` branch -- NOT range midpoints.
# For multiplicative factors (scale/brightness/contrast/saturation) the
# range midpoint happens to coincide with "no change" (1.0) only because
# parameters.yaml's ranges are centered on it; for noise/blur, whose ranges
# start at 0.0 ("no effect") rather than being centered on it, the
# identity value is the range's lower bound, not its midpoint. Using
# explicit constants here avoids that trap entirely.
_IDENTITY_SCALE = 1.0
_IDENTITY_ROTATION_DEG = 0.0
_IDENTITY_OFFSET = 0.0
_IDENTITY_BRIGHTNESS = 1.0
_IDENTITY_CONTRAST = 1.0
_IDENTITY_COLOR_TEMPERATURE = 0.0
_IDENTITY_HUE_SHIFT_DEG = 0.0
_IDENTITY_SATURATION = 1.0
_IDENTITY_GAUSSIAN_STD = 0.0
_IDENTITY_BLUR_RADIUS = 0.0


@dataclass(frozen=True)
class CameraTransform:
    """Sampled/identity camera parameters for one render.

    Attributes:
        scale: Multiplier applied to the object's native pixel size.
        rotation_deg: In-plane rotation, in degrees.
        offset_x: Horizontal offset from canvas center, in pixels.
        offset_y: Vertical offset from canvas center, in pixels.
    """

    scale: float
    rotation_deg: float
    offset_x: float
    offset_y: float


@dataclass(frozen=True)
class LightParams:
    """Sampled/identity lighting parameters for one render."""

    brightness: float
    contrast: float
    color_temperature: float


@dataclass(frozen=True)
class MaterialParams:
    """Sampled/identity material parameters for one render."""

    hue_shift_deg: float
    saturation: float


@dataclass(frozen=True)
class NoiseParams:
    """Sampled/identity noise parameters for one render."""

    gaussian_std: float
    blur_radius: float


@dataclass(frozen=True)
class RenderParams:
    """Every parameter actually used for one render -- for manifests/ablation analysis.

    Attributes:
        camera: See `CameraTransform`.
        light: See `LightParams`.
        material: See `MaterialParams`.
        noise: See `NoiseParams`.
        background_color: `(r, g, b)` fill color used behind the object.
    """

    camera: CameraTransform
    light: LightParams
    material: MaterialParams
    noise: NoiseParams
    background_color: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        """This record as a plain JSON-serializable dict."""
        return {
            "camera": vars(self.camera),
            "light": vars(self.light),
            "material": vars(self.material),
            "noise": vars(self.noise),
            "background_color": list(self.background_color),
        }


@dataclass(frozen=True)
class RenderResult:
    """One rendered scene.

    Attributes:
        image: The composited `RGB` frame.
        bbox: The placed object's ground-truth bounding box.
        params: Every parameter actually used to produce `image`.
    """

    image: Image.Image
    bbox: Bbox
    params: RenderParams


def sample_camera_transform(
    config: CameraConfig, canvas_size: tuple[int, int], rng: np.random.Generator | None
) -> CameraTransform:
    """Sample (or return the identity) camera transform.

    Args:
        config: `parameters.yaml`'s `camera:` section.
        canvas_size: `(width, height)` the offset is measured against.
        rng: Source of randomness; `None` returns the identity transform
            (native scale, no rotation, centered).

    Returns:
        A `CameraTransform`.
    """
    if rng is None:
        return CameraTransform(
            scale=_IDENTITY_SCALE,
            rotation_deg=_IDENTITY_ROTATION_DEG,
            offset_x=_IDENTITY_OFFSET,
            offset_y=_IDENTITY_OFFSET,
        )
    max_offset_x = config.translation_fraction * canvas_size[0]
    max_offset_y = config.translation_fraction * canvas_size[1]
    return CameraTransform(
        scale=float(rng.uniform(*config.scale_range)),
        rotation_deg=float(rng.uniform(*config.rotation_range_deg)),
        offset_x=float(rng.uniform(-max_offset_x, max_offset_x)),
        offset_y=float(rng.uniform(-max_offset_y, max_offset_y)),
    )


def sample_light_params(config: LightConfig, rng: np.random.Generator | None) -> LightParams:
    """Sample (or return the identity) lighting parameters. `rng=None` -> no change."""
    if rng is None:
        return LightParams(
            brightness=_IDENTITY_BRIGHTNESS,
            contrast=_IDENTITY_CONTRAST,
            color_temperature=_IDENTITY_COLOR_TEMPERATURE,
        )
    return LightParams(
        brightness=float(rng.uniform(*config.brightness_range)),
        contrast=float(rng.uniform(*config.contrast_range)),
        color_temperature=float(rng.uniform(*config.color_temperature_range)),
    )


def sample_material_params(
    config: MaterialConfig, rng: np.random.Generator | None
) -> MaterialParams:
    """Sample (or return the identity) material parameters. `rng=None` -> no change."""
    if rng is None:
        return MaterialParams(
            hue_shift_deg=_IDENTITY_HUE_SHIFT_DEG, saturation=_IDENTITY_SATURATION
        )
    return MaterialParams(
        hue_shift_deg=float(rng.uniform(*config.hue_shift_range_deg)),
        saturation=float(rng.uniform(*config.saturation_range)),
    )


def sample_noise_params(config: NoiseConfig, rng: np.random.Generator | None) -> NoiseParams:
    """Sample (or return the identity) noise parameters. `rng=None` -> no noise/blur."""
    if rng is None:
        return NoiseParams(
            gaussian_std=_IDENTITY_GAUSSIAN_STD, blur_radius=_IDENTITY_BLUR_RADIUS
        )
    return NoiseParams(
        gaussian_std=float(rng.uniform(*config.gaussian_std_range)),
        blur_radius=float(rng.uniform(*config.blur_radius_range)),
    )


def sample_background_color(
    config: BackgroundConfig, rng: np.random.Generator | None
) -> tuple[int, int, int]:
    """Sample (or return the midpoint) background fill color.

    Unlike the other `sample_*` functions, `rng=None` here uses the
    midpoint of `(color_low, color_high)`, not a hardcoded identity value
    -- a solid background color has no "neutral" meaning the way a
    multiplicative light/material factor does, so a fixed representative
    color is used instead.
    """
    if rng is None:
        return tuple(
            round((lo + hi) / 2) for lo, hi in zip(config.color_low, config.color_high, strict=True)
        )
    return tuple(
        int(rng.integers(lo, hi + 1))
        for lo, hi in zip(config.color_low, config.color_high, strict=True)
    )


def synthesize_placeholder_object(size: tuple[int, int] = (128, 128)) -> Image.Image:
    """A simple synthetic RGBA object cutout (a filled rounded rectangle).

    Stands in for a real object cutout image in demos/tests -- no client
    data or real product photography involved.

    Args:
        size: `(width, height)` of the returned cutout.

    Returns:
        An `RGBA` image: opaque rounded rectangle on a transparent background.
    """
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    margin = int(min(size) * 0.15)
    ImageDraw.Draw(image).rounded_rectangle(
        [margin, margin, size[0] - margin, size[1] - margin],
        radius=int(min(size) * 0.15),
        fill=(140, 140, 150, 255),
    )
    return image


def apply_material(object_rgba: Image.Image, material: MaterialParams) -> Image.Image:
    """Shift an object cutout's hue/saturation, preserving its alpha silhouette.

    Args:
        object_rgba: The object cutout, `RGBA`.
        material: Hue/saturation adjustment to apply.

    Returns:
        A new `RGBA` image with the same alpha channel as `object_rgba`.
    """
    alpha = object_rgba.split()[-1]
    hsv = np.array(object_rgba.convert("RGB").convert("HSV"), dtype=np.int16)
    hue_shift = round(material.hue_shift_deg / 360.0 * 255)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * material.saturation, 0, 255)
    shifted = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGBA")
    shifted.putalpha(alpha)
    return shifted


def place_object(
    object_rgba: Image.Image, canvas_size: tuple[int, int], transform: CameraTransform
) -> tuple[Image.Image, Bbox]:
    """Scale/rotate/position an object cutout onto a transparent canvas.

    Args:
        object_rgba: The object cutout, `RGBA`.
        canvas_size: `(width, height)` of the output canvas.
        transform: Where/how large/how rotated to place it.

    Returns:
        `(canvas, bbox)`: `canvas` is `RGBA`, `canvas_size`-sized, with the
        transformed object pasted onto transparent background. `bbox` is
        the object's actual visible pixel extent (from its alpha channel),
        the ground-truth box for detector training.

    Raises:
        ValueError: If the transform places the object entirely outside `canvas_size`.
    """
    new_size = (
        max(1, round(object_rgba.width * transform.scale)),
        max(1, round(object_rgba.height * transform.scale)),
    )
    obj = object_rgba.resize(new_size, Image.BILINEAR)
    if transform.rotation_deg != 0:
        obj = obj.rotate(transform.rotation_deg, expand=True, resample=Image.BILINEAR)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    center_x = canvas_size[0] / 2 + transform.offset_x
    center_y = canvas_size[1] / 2 + transform.offset_y
    top_left = (round(center_x - obj.width / 2), round(center_y - obj.height / 2))
    canvas.paste(obj, top_left, obj)

    alpha = np.array(canvas.split()[-1])
    ys, xs = np.nonzero(alpha)
    if xs.size == 0:
        raise ValueError(
            f"Camera transform {transform} placed the object entirely outside the "
            f"{canvas_size} canvas -- no visible pixels."
        )
    bbox: Bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return canvas, bbox


def apply_light(image_rgb: Image.Image, light: LightParams) -> Image.Image:
    """Apply brightness/contrast/white-balance to a composited `RGB` frame.

    Args:
        image_rgb: The frame, `RGB`.
        light: Brightness/contrast/color-temperature to apply.

    Returns:
        A new `RGB` image.
    """
    image = ImageEnhance.Brightness(image_rgb).enhance(light.brightness)
    image = ImageEnhance.Contrast(image).enhance(light.contrast)
    if light.color_temperature != 0:
        array = np.array(image, dtype=np.float32)
        # positive = cooler (boost blue, cut red); negative = warmer (boost red, cut blue).
        array[..., 0] = np.clip(array[..., 0] * (1 - light.color_temperature), 0, 255)
        array[..., 2] = np.clip(array[..., 2] * (1 + light.color_temperature), 0, 255)
        image = Image.fromarray(array.astype(np.uint8))
    return image


def apply_noise(
    image_rgb: Image.Image, noise: NoiseParams, rng: np.random.Generator
) -> Image.Image:
    """Apply gaussian blur then additive gaussian sensor noise to a frame.

    Args:
        image_rgb: The frame, `RGB`.
        noise: Blur radius / noise std to apply. Both `0` is a no-op.
        rng: Source of randomness for the additive noise.

    Returns:
        A new `RGB` image.
    """
    image = image_rgb
    if noise.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=noise.blur_radius))
    if noise.gaussian_std > 0:
        array = np.array(image, dtype=np.float32)
        array += rng.normal(loc=0.0, scale=noise.gaussian_std * 255.0, size=array.shape)
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    return image


def render_scene(
    object_rgba: Image.Image,
    config: DRParameters,
    rng: np.random.Generator | None = None,
) -> RenderResult:
    """Render one scene: place `object_rgba` on a background, apply camera/light/material/noise.

    Args:
        object_rgba: The object cutout to place, `RGBA` (alpha marks its silhouette).
        config: Parsed `parameters.yaml`.
        rng: Source of randomness for the `with_dr` branch; `None` for the
            `no_dr` branch (every parameter held at its neutral/identity
            value, see this module's docstring).

    Returns:
        A `RenderResult` with the composited image, the object's ground-truth
        bounding box, and every parameter actually used.

    Raises:
        ValueError: If the sampled camera transform places the object
            entirely outside the canvas (from `place_object`).
    """
    canvas_size = config.canvas.size

    camera = sample_camera_transform(config.camera, canvas_size, rng)
    light = sample_light_params(config.light, rng)
    material = sample_material_params(config.material, rng)
    noise = sample_noise_params(config.noise, rng)
    background_color = sample_background_color(config.background, rng)

    placed, bbox = place_object(apply_material(object_rgba, material), canvas_size, camera)

    background = Image.new("RGBA", canvas_size, (*background_color, 255))
    composited = Image.alpha_composite(background, placed).convert("RGB")

    lit = apply_light(composited, light)
    noisy_rng = rng if rng is not None else np.random.default_rng(config.seed)
    frame = apply_noise(lit, noise, noisy_rng)

    params = RenderParams(
        camera=camera,
        light=light,
        material=material,
        noise=noise,
        background_color=background_color,
    )
    return RenderResult(image=frame, bbox=bbox, params=params)
