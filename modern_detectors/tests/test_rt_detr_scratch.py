"""Unit tests for `modern_detectors.rt_detr_scratch` -- real training steps, no mocking.

Scaled down (few steps, tiny batches) so the suite stays fast -- the
demo script (`make demo`) is where the real, illustrative-scale
training run happens (see `modern_detectors/README.md`'s real numbers).
"""

from __future__ import annotations

from classical_cv.defect_images import build_demo_dataset
from modern_detectors.dataset import resize_dataset
from modern_detectors.rt_detr_scratch import (
    build_tiny_rt_detr_config,
    build_tiny_rt_detr_model,
    predict_rt_detr,
    train_rt_detr,
)
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")
SOURCE_SIZE = (512, 512)


def _tiny_dataset(num_positive: int = 4, num_negative: int = 2, seed: int = 42):
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive, num_negative, seed)
    return resize_dataset(dataset, SOURCE_SIZE)


def test_build_tiny_rt_detr_config_has_no_timm_backbone() -> None:
    config = build_tiny_rt_detr_config()
    assert config.backbone_config is not None
    assert config.backbone_config.model_type == "rt_detr_resnet"


def test_build_tiny_rt_detr_model_is_reproducible_for_the_same_seed() -> None:
    first = build_tiny_rt_detr_model(seed=7)
    second = build_tiny_rt_detr_model(seed=7)
    first_param = next(first.parameters())
    second_param = next(second.parameters())
    assert (first_param == second_param).all()


def test_train_rt_detr_loss_decreases_over_a_short_real_run() -> None:
    """30 steps / batch_size=4 (not fewer) -- empirically the minimum that
    reliably shows a real downward trend on this tiny dataset; shorter
    runs are noisy enough to occasionally go the other way (this is a
    few-dozen-parameter-update toy run, not a converged training curve)."""
    images = _tiny_dataset()
    model = build_tiny_rt_detr_model(seed=0)
    run = train_rt_detr(model, images, num_steps=30, batch_size=4, learning_rate=1e-3, seed=1)

    assert len(run.step_losses) == 30
    assert all(loss > 0.0 for loss in run.step_losses)
    first_five_mean = sum(run.step_losses[:5]) / 5
    last_five_mean = sum(run.step_losses[-5:]) / 5
    assert last_five_mean < first_five_mean


def test_train_rt_detr_handles_negative_images_with_zero_boxes() -> None:
    """A batch containing an all-negative image (zero ground truth boxes)
    must not crash the Hungarian matcher -- real bug class this guards
    against, see `rt_detr_scratch._to_rt_detr_labels`'s docstring."""
    images = _tiny_dataset(num_positive=2, num_negative=4)
    model = build_tiny_rt_detr_model(seed=0)
    run = train_rt_detr(model, images, num_steps=3, batch_size=4, learning_rate=1e-3, seed=2)
    assert len(run.step_losses) == 3


def test_predict_rt_detr_returns_one_entry_per_image() -> None:
    images = _tiny_dataset()
    model = build_tiny_rt_detr_model(seed=0)
    predictions = predict_rt_detr(model, images, score_threshold=0.0)

    assert set(predictions.keys()) == {image.image_id for image in images}
    for preds in predictions.values():
        assert len(preds) <= 10  # TINY_NUM_QUERIES
        for prediction in preds:
            assert prediction.class_name == "defect"
            assert 0.0 <= prediction.score <= 1.0
