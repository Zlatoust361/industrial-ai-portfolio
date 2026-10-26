"""Inpainting quality assessment + top-fraction selection (week 05, day 3).

Four metrics, none needing `torch`/a GPU -- `numpy` + `opencv-python-headless`
+ `scipy` only, so this module (unlike the diffusion backends themselves)
runs and is genuinely testable anywhere, including this sandbox:

- `psnr_outside_mask`/`ssim_outside_mask`: fidelity checks on the region
  OUTSIDE the inpainting mask. A correct inpainting backend only touches
  pixels the mask marks -- everything else should come back (near-)identical
  to the original. Low scores here catch a backend that leaked changes
  outside the mask (a real failure mode, not a hypothetical one).
- `blur_inside_mask`/`contrast_inside_mask`: realism proxies for the
  generated defect content INSIDE the mask. A degenerate inpaint (flat
  smear, oversmoothed patch) scores low on both -- a real, common failure
  mode of inpainting models, and exactly the kind of thing week 05 day 1's
  research prompt asked about ("типичные артефакты, метрики quality-gate").

`ssim_outside_mask` uses a *simplified*, uniform-window SSIM (via
`scipy.ndimage.uniform_filter`), not the reference Gaussian-window
implementation from Wang et al. 2004 -- close enough for a quality-gate
ranking signal, and avoids adding `scikit-image` as a dependency for one
function. Say so explicitly rather than implying this is the canonical
SSIM some other tool might report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

PSNR_CAP_DB = 100.0
"""Returned in place of `+inf` when outside-mask pixels are bit-identical
(MSE == 0) -- a real, not-uncommon case for `Cv2InpaintBackend` on a
noiseless synthetic base image. 100 dB is already far beyond any
meaningfully distinguishable fidelity difference for 8-bit images."""


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Half-open pixel bounding box `(x_min, y_min, x_max, y_max)` of `mask`'s non-zero pixels.

    Args:
        mask: `(height, width)` array, non-zero = defect area (see
            `src.synthetic.mask_generation.generate_defect_mask`).

    Returns:
        `(x_min, y_min, x_max, y_max)`, half-open (matches
        `src.synthetic.mask_generation.Bbox`'s convention).

    Raises:
        ValueError: If `mask` has no non-zero pixels.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Mask has no non-zero pixels -- nothing to assess inside it.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """`(H, W, 3)` uint8 RGB -> `(H, W)` float64 grayscale, via OpenCV's standard weights."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def psnr_outside_mask(
    original_rgb: np.ndarray, inpainted_rgb: np.ndarray, mask: np.ndarray
) -> float:
    """Peak signal-to-noise ratio between `original_rgb` and `inpainted_rgb`, outside `mask`.

    Args:
        original_rgb: `(H, W, 3)` uint8, the pre-inpaint base image.
        inpainted_rgb: `(H, W, 3)` uint8, same shape, the backend's output.
        mask: `(H, W)` uint8, non-zero = defect area (excluded from this comparison).

    Returns:
        PSNR in dB (higher = more faithful outside the mask). Capped at
        `PSNR_CAP_DB` when the outside-mask region is bit-identical.

    Raises:
        ValueError: If shapes don't match, or `mask` covers the entire image.
    """
    _check_same_shape(original_rgb, inpainted_rgb, mask)
    outside = mask == 0
    if not outside.any():
        raise ValueError("Mask covers the entire image -- no outside-mask region to compare.")

    diff = original_rgb[outside].astype(np.float64) - inpainted_rgb[outside].astype(np.float64)
    mse = float(np.mean(diff**2))
    if mse == 0.0:
        return PSNR_CAP_DB
    return min(PSNR_CAP_DB, 10.0 * math.log10((255.0**2) / mse))


def ssim_outside_mask(
    original_rgb: np.ndarray, inpainted_rgb: np.ndarray, mask: np.ndarray, window: int = 7
) -> float:
    """Simplified (uniform-window) structural similarity, outside `mask`, on grayscale.

    Args:
        original_rgb: `(H, W, 3)` uint8, the pre-inpaint base image.
        inpainted_rgb: `(H, W, 3)` uint8, same shape, the backend's output.
        mask: `(H, W)` uint8, non-zero = defect area (excluded from this comparison).
        window: Uniform-filter window size, in pixels (odd, `>= 3` recommended).

    Returns:
        Mean SSIM over outside-mask pixels, in `[-1, 1]` (in practice
        close to `[0, 1]` for natural images; `1.0` = identical). Note:
        even when outside-mask pixels are bit-identical, this can be
        slightly below `1.0` -- `window`'s local statistics near the mask
        boundary straddle a few inside-mask pixels too, which did change.
        Only a thin boundary band is affected; the score still stays high.

    Raises:
        ValueError: If shapes don't match, `mask` covers the entire image,
            or `window` is smaller than 2.
    """
    _check_same_shape(original_rgb, inpainted_rgb, mask)
    outside = mask == 0
    if not outside.any():
        raise ValueError("Mask covers the entire image -- no outside-mask region to compare.")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")

    img1 = _to_grayscale(original_rgb)
    img2 = _to_grayscale(inpainted_rgb)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = uniform_filter(img1, size=window)
    mu2 = uniform_filter(img2, size=window)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = uniform_filter(img1 * img1, size=window) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, size=window) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=window) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    return float(ssim_map[outside].mean())


def blur_inside_mask(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian, cropped to `mask`'s bounding box -- a standard sharpness proxy.

    Higher = sharper/more detail; a near-flat, oversmoothed inpaint scores
    low. Computed over the mask's bounding-box crop (not pixel-exact
    masking) -- simpler, and detail *near* the defect matters too, not
    just literally inside its silhouette.

    Args:
        image_rgb: `(H, W, 3)` uint8, the inpainted image.
        mask: `(H, W)` uint8, non-zero = defect area.

    Returns:
        Non-negative float; unbounded above.

    Raises:
        ValueError: If `mask` has no non-zero pixels (from `mask_bbox`).
    """
    x0, y0, x1, y1 = mask_bbox(mask)
    crop_gray = _to_grayscale(image_rgb[y0:y1, x0:x1])
    return float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())


def contrast_inside_mask(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Standard-deviation contrast, cropped to `mask`'s bounding box.

    Args:
        image_rgb: `(H, W, 3)` uint8, the inpainted image.
        mask: `(H, W)` uint8, non-zero = defect area.

    Returns:
        Non-negative float, in `[0, 127.5]` for 8-bit grayscale.

    Raises:
        ValueError: If `mask` has no non-zero pixels (from `mask_bbox`).
    """
    x0, y0, x1, y1 = mask_bbox(mask)
    crop_gray = _to_grayscale(image_rgb[y0:y1, x0:x1])
    return float(crop_gray.std())


def _check_same_shape(
    original_rgb: np.ndarray, inpainted_rgb: np.ndarray, mask: np.ndarray
) -> None:
    if original_rgb.shape != inpainted_rgb.shape:
        raise ValueError(
            f"original_rgb {original_rgb.shape} and inpainted_rgb {inpainted_rgb.shape} "
            "must have the same shape."
        )
    if mask.shape != original_rgb.shape[:2]:
        raise ValueError(
            f"mask {mask.shape} must match image height/width {original_rgb.shape[:2]}."
        )


METRIC_NAMES: tuple[str, ...] = (
    "psnr_outside_mask",
    "ssim_outside_mask",
    "blur_inside_mask",
    "contrast_inside_mask",
)
"""`QualityMetrics.to_dict()`'s keys, in a fixed order -- shared by
`composite_scores` and week 05 day 4's comparison report so both iterate
metrics in the same, single place of truth."""


@dataclass(frozen=True)
class QualityMetrics:
    """One inpainting result's four raw metric values (see this module's docstring)."""

    psnr_outside_mask: float
    ssim_outside_mask: float
    blur_inside_mask: float
    contrast_inside_mask: float

    def to_dict(self) -> dict[str, float]:
        """This instance as a plain JSON-serializable dict."""
        return {
            "psnr_outside_mask": self.psnr_outside_mask,
            "ssim_outside_mask": self.ssim_outside_mask,
            "blur_inside_mask": self.blur_inside_mask,
            "contrast_inside_mask": self.contrast_inside_mask,
        }


def assess(
    original_rgb: np.ndarray, inpainted_rgb: np.ndarray, mask: np.ndarray, ssim_window: int = 7
) -> QualityMetrics:
    """Compute all four quality metrics for one inpainting result.

    Args:
        original_rgb: `(H, W, 3)` uint8, the pre-inpaint base image.
        inpainted_rgb: `(H, W, 3)` uint8, same shape, the backend's output.
        mask: `(H, W)` uint8, non-zero = defect area.
        ssim_window: See `ssim_outside_mask`.

    Returns:
        A `QualityMetrics` with all four values.
    """
    return QualityMetrics(
        psnr_outside_mask=psnr_outside_mask(original_rgb, inpainted_rgb, mask),
        ssim_outside_mask=ssim_outside_mask(original_rgb, inpainted_rgb, mask, window=ssim_window),
        blur_inside_mask=blur_inside_mask(inpainted_rgb, mask),
        contrast_inside_mask=contrast_inside_mask(inpainted_rgb, mask),
    )


def normalize_min_max(values: list[float]) -> list[float]:
    """Min-max normalize `values` to `[0, 1]`.

    Args:
        values: Raw metric values across one batch.

    Returns:
        Normalized values, same order. If every value is equal (including
        a single-element list), returns all `1.0` -- no discriminating
        signal in that metric for this batch, so it shouldn't penalize
        anything relative to the others.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


@dataclass(frozen=True)
class ScoredResult:
    """One inpainting result plus its metrics and composite ranking score.

    Attributes:
        identifier: Caller-supplied label (e.g. base filename) -- carried
            through purely for manifest/report traceability, not used by
            the scoring/selection logic itself.
        metrics: The raw `QualityMetrics`.
        composite_score: Weighted sum of `metrics`' min-max-normalized
            values across the batch (see `composite_scores`). Higher = better.
        payload: Caller-supplied extra data (e.g. the `PIL.Image`, backend
            name) carried through unchanged -- this module never inspects it.
    """

    identifier: str
    metrics: QualityMetrics
    composite_score: float
    payload: object = field(default=None)


def composite_scores(
    identifiers: list[str],
    metrics: list[QualityMetrics],
    weights: dict[str, float],
    payloads: list[object] | None = None,
) -> list[ScoredResult]:
    """Combine a batch of `QualityMetrics` into per-item composite scores.

    Each metric is min-max normalized ACROSS THIS BATCH (see
    `normalize_min_max`) before weighting -- so `weights` are relative
    importances, not absolute thresholds, and the same weights are
    meaningful regardless of what raw scale a given metric happens to
    produce (PSNR in dB, blur variance unbounded, etc.).

    Args:
        identifiers: One label per item, same length/order as `metrics`.
        metrics: One `QualityMetrics` per item.
        weights: Per-metric weight, keyed by `QualityMetrics.to_dict()`'s
            field names. Missing keys default to `0.0` (that metric is
            ignored); unknown extra keys are ignored.
        payloads: Optional, one per item -- carried through unchanged on
            each `ScoredResult`.

    Returns:
        One `ScoredResult` per item, in the SAME order as the input (not
        sorted -- see `select_top_fraction` for ranking/selection).

    Raises:
        ValueError: If `identifiers`/`metrics`/`payloads` lengths don't match.
    """
    if payloads is None:
        payloads = [None] * len(metrics)
    if not (len(identifiers) == len(metrics) == len(payloads)):
        raise ValueError(
            f"identifiers ({len(identifiers)}), metrics ({len(metrics)}), and payloads "
            f"({len(payloads)}) must be the same length."
        )
    if not metrics:
        return []

    field_names = METRIC_NAMES
    normalized_by_field = {
        name: normalize_min_max([getattr(m, name) for m in metrics]) for name in field_names
    }

    results = []
    for i, (identifier, m, payload) in enumerate(zip(identifiers, metrics, payloads, strict=True)):
        score = sum(weights.get(name, 0.0) * normalized_by_field[name][i] for name in field_names)
        results.append(
            ScoredResult(identifier=identifier, metrics=m, composite_score=score, payload=payload)
        )
    return results


def select_top_fraction(
    results: list[ScoredResult], fraction: float
) -> tuple[list[ScoredResult], list[ScoredResult]]:
    """Split `results` into (kept, rejected) by composite score, highest first.

    Args:
        results: Scored batch, any order (see `composite_scores`).
        fraction: Fraction to keep, in `[0, 1]`. The keep-count is
            `floor(len(results) * fraction)` -- rounds DOWN, so this never
            keeps more than the requested fraction; ties at the cutoff are
            broken by whichever sorts first under a stable sort on the
            original list order (deterministic given the same input order).

    Returns:
        `(kept, rejected)`, each sorted by composite score descending.

    Raises:
        ValueError: If `fraction` isn't in `[0, 1]`.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    ranked = sorted(results, key=lambda r: r.composite_score, reverse=True)
    keep_count = math.floor(len(ranked) * fraction)
    return ranked[:keep_count], ranked[keep_count:]
