"""Unit tests for `src.synthetic.quality_assessment`."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic.quality_assessment import (
    PSNR_CAP_DB,
    QualityMetrics,
    ScoredResult,
    assess,
    blur_inside_mask,
    composite_scores,
    contrast_inside_mask,
    mask_bbox,
    normalize_min_max,
    psnr_outside_mask,
    select_top_fraction,
    ssim_outside_mask,
)


def _textured_image(size: tuple[int, int] = (64, 64), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width, height = size
    gray = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return np.stack([gray] * 3, axis=-1)


def _mask_with_bbox(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 255
    return mask


# -- mask_bbox -------------------------------------------------------------


def test_mask_bbox_matches_the_drawn_rectangle() -> None:
    mask = _mask_with_bbox((64, 64), (10, 20, 30, 40))
    assert mask_bbox(mask) == (10, 20, 30, 40)


def test_mask_bbox_empty_mask_raises() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="no non-zero pixels"):
        mask_bbox(mask)


# -- psnr_outside_mask -------------------------------------------------------


def test_psnr_outside_mask_identical_images_is_capped() -> None:
    image = _textured_image()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    assert psnr_outside_mask(image, image, mask) == PSNR_CAP_DB


def test_psnr_outside_mask_ignores_differences_strictly_inside_the_mask() -> None:
    original = _textured_image()
    inpainted = original.copy()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    inpainted[10:30, 10:30] = 0  # change only inside the mask
    assert psnr_outside_mask(original, inpainted, mask) == PSNR_CAP_DB


def test_psnr_outside_mask_penalizes_changes_outside_the_mask() -> None:
    original = _textured_image()
    inpainted = original.copy()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    inpainted[40:50, 40:50] = 255 - original[40:50, 40:50]  # big change outside the mask
    assert psnr_outside_mask(original, inpainted, mask) < PSNR_CAP_DB


def test_psnr_outside_mask_full_image_mask_raises() -> None:
    image = _textured_image()
    mask = np.full((64, 64), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="entire image"):
        psnr_outside_mask(image, image, mask)


def test_psnr_outside_mask_shape_mismatch_raises() -> None:
    original = _textured_image((64, 64))
    inpainted = _textured_image((32, 32))
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    with pytest.raises(ValueError, match="same shape"):
        psnr_outside_mask(original, inpainted, mask)


# -- ssim_outside_mask -------------------------------------------------------


def test_ssim_outside_mask_identical_images_is_near_one() -> None:
    image = _textured_image()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    assert ssim_outside_mask(image, image, mask) == pytest.approx(1.0, abs=1e-9)


def test_ssim_outside_mask_lower_for_noisier_outside_region() -> None:
    original = _textured_image()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))

    slightly_noisy = original.copy()
    rng = np.random.default_rng(1)
    slightly_noisy[..., :] = np.clip(
        original.astype(int) + rng.integers(-5, 6, size=original.shape), 0, 255
    ).astype(np.uint8)

    very_noisy = original.copy()
    very_noisy[..., :] = 255 - original  # maximally different outside the mask too

    ssim_slight = ssim_outside_mask(original, slightly_noisy, mask)
    ssim_very = ssim_outside_mask(original, very_noisy, mask)
    assert 1.0 >= ssim_slight > ssim_very


def test_ssim_outside_mask_rejects_window_below_two() -> None:
    image = _textured_image()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    with pytest.raises(ValueError, match="window"):
        ssim_outside_mask(image, image, mask, window=1)


# -- blur_inside_mask / contrast_inside_mask --------------------------------


def test_blur_inside_mask_higher_for_textured_than_flat_region() -> None:
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))

    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    textured = flat.copy()
    textured[10:30, 10:30] = _textured_image((20, 20), seed=2)

    assert blur_inside_mask(flat, mask) == pytest.approx(0.0, abs=1e-6)
    assert blur_inside_mask(textured, mask) > blur_inside_mask(flat, mask)


def test_contrast_inside_mask_higher_for_textured_than_flat_region() -> None:
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))

    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    textured = flat.copy()
    textured[10:30, 10:30] = _textured_image((20, 20), seed=3)

    assert contrast_inside_mask(flat, mask) == pytest.approx(0.0, abs=1e-6)
    assert contrast_inside_mask(textured, mask) > 0.0


def test_blur_and_contrast_inside_mask_empty_mask_raises() -> None:
    image = _textured_image()
    mask = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="no non-zero pixels"):
        blur_inside_mask(image, mask)
    with pytest.raises(ValueError, match="no non-zero pixels"):
        contrast_inside_mask(image, mask)


# -- assess ------------------------------------------------------------------


def test_assess_returns_all_four_metrics() -> None:
    original = _textured_image()
    mask = _mask_with_bbox((64, 64), (10, 10, 30, 30))
    inpainted = original.copy()
    inpainted[10:30, 10:30] = _textured_image((20, 20), seed=5)

    metrics = assess(original, inpainted, mask)
    assert metrics.psnr_outside_mask == PSNR_CAP_DB  # untouched outside the mask
    # Not exactly 1.0: the windowed SSIM's local statistics near the
    # mask boundary straddle a few changed inside-mask pixels too (see
    # ssim_outside_mask's own boundary-effect test below) -- still high,
    # since only a thin boundary band is affected.
    assert metrics.ssim_outside_mask > 0.95
    assert metrics.blur_inside_mask > 0.0
    assert metrics.contrast_inside_mask > 0.0
    assert set(metrics.to_dict()) == {
        "psnr_outside_mask",
        "ssim_outside_mask",
        "blur_inside_mask",
        "contrast_inside_mask",
    }


# -- normalize_min_max --------------------------------------------------------


def test_normalize_min_max_basic_range() -> None:
    assert normalize_min_max([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalize_min_max_all_equal_returns_all_ones() -> None:
    assert normalize_min_max([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]


def test_normalize_min_max_empty_returns_empty() -> None:
    assert normalize_min_max([]) == []


def test_normalize_min_max_single_value_returns_one() -> None:
    assert normalize_min_max([42.0]) == [1.0]


# -- composite_scores ----------------------------------------------------------


def _metrics(psnr: float, ssim: float, blur: float, contrast: float) -> QualityMetrics:
    return QualityMetrics(
        psnr_outside_mask=psnr,
        ssim_outside_mask=ssim,
        blur_inside_mask=blur,
        contrast_inside_mask=contrast,
    )


def test_composite_scores_ranks_the_better_item_higher() -> None:
    metrics = [_metrics(20, 0.5, 100, 20), _metrics(40, 0.9, 500, 50)]
    weights = {
        "psnr_outside_mask": 1.0,
        "ssim_outside_mask": 1.0,
        "blur_inside_mask": 1.0,
        "contrast_inside_mask": 1.0,
    }
    results = composite_scores(["worse", "better"], metrics, weights)
    by_id = {r.identifier: r.composite_score for r in results}
    assert by_id["better"] > by_id["worse"]


def test_composite_scores_missing_weight_key_defaults_to_zero() -> None:
    metrics = [_metrics(20, 0.1, 999, 999), _metrics(40, 0.9, 1, 1)]
    results = composite_scores(["a", "b"], metrics, {"ssim_outside_mask": 1.0})
    # Only ssim_outside_mask counts -- "b" has the higher ssim, so it wins
    # despite far worse blur/contrast, which have no weight entry.
    by_id = {r.identifier: r.composite_score for r in results}
    assert by_id["b"] > by_id["a"]


def test_composite_scores_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        composite_scores(["a", "b"], [_metrics(1, 1, 1, 1)], {})


def test_composite_scores_empty_input_returns_empty() -> None:
    assert composite_scores([], [], {}) == []


def test_composite_scores_preserves_input_order_not_sorted() -> None:
    metrics = [_metrics(40, 0.9, 500, 50), _metrics(20, 0.5, 100, 20)]
    results = composite_scores(["better", "worse"], metrics, {"psnr_outside_mask": 1.0})
    assert [r.identifier for r in results] == ["better", "worse"]


def test_composite_scores_carries_payload_through_unchanged() -> None:
    results = composite_scores(["a"], [_metrics(1, 1, 1, 1)], {}, payloads=[{"marker": 123}])
    assert results[0].payload == {"marker": 123}


# -- select_top_fraction -------------------------------------------------------


def _scored(identifier: str, score: float) -> ScoredResult:
    return ScoredResult(identifier=identifier, metrics=_metrics(0, 0, 0, 0), composite_score=score)


def test_select_top_fraction_keeps_the_highest_scores() -> None:
    results = [_scored("a", 0.1), _scored("b", 0.9), _scored("c", 0.5), _scored("d", 0.3)]
    kept, rejected = select_top_fraction(results, 0.5)

    assert [r.identifier for r in kept] == ["b", "c"]
    assert [r.identifier for r in rejected] == ["d", "a"]


def test_select_top_fraction_rounds_down() -> None:
    results = [_scored(str(i), float(i)) for i in range(3)]  # 3 * 0.75 = 2.25 -> floor to 2
    kept, _ = select_top_fraction(results, 0.75)
    assert len(kept) == 2


def test_select_top_fraction_zero_keeps_nothing() -> None:
    results = [_scored("a", 1.0)]
    kept, rejected = select_top_fraction(results, 0.0)
    assert kept == []
    assert len(rejected) == 1


def test_select_top_fraction_one_keeps_everything() -> None:
    results = [_scored("a", 1.0), _scored("b", 2.0)]
    kept, rejected = select_top_fraction(results, 1.0)
    assert len(kept) == 2
    assert rejected == []


def test_select_top_fraction_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="fraction"):
        select_top_fraction([], 1.5)


def test_select_top_fraction_stable_on_ties() -> None:
    results = [_scored("first", 1.0), _scored("second", 1.0), _scored("third", 1.0)]
    kept, _ = select_top_fraction(results, 2 / 3)
    assert [r.identifier for r in kept] == ["first", "second"]
