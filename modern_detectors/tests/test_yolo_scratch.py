"""Unit tests for `modern_detectors.yolo_scratch` -- real `ultralytics` training, no mocking.

Scaled down (2 epochs, tiny dataset) so the suite stays fast -- see
`modern_detectors/README.md`'s real numbers for the demo-scale run
(`make demo`, 60 epochs).
"""

from __future__ import annotations

from classical_cv.defect_images import build_demo_dataset
from modern_detectors.dataset import resize_dataset
from modern_detectors.yolo_scratch import predict_yolo, train_yolo_from_scratch
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")
SOURCE_SIZE = (512, 512)


def _tiny_dataset(num_positive: int = 6, num_negative: int = 2, seed: int = 42):
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive, num_negative, seed)
    return resize_dataset(dataset, SOURCE_SIZE)


def test_train_yolo_from_scratch_builds_no_pretrained_weights(tmp_path) -> None:
    """`YOLO("yolov8n.yaml")`'s random init -- `train_yolo_from_scratch` must never
    download or load a `.pt` checkpoint (the whole point of the from-scratch comparison,
    see the module docstring)."""
    images = _tiny_dataset()
    run = train_yolo_from_scratch(images, tmp_path / "ds", epochs=1, batch_size=2, seed=0)
    assert run.train_seconds > 0.0
    assert (tmp_path / "ds" / "images").is_dir()


def test_predict_yolo_returns_one_entry_per_image(tmp_path) -> None:
    images = _tiny_dataset()
    run = train_yolo_from_scratch(images, tmp_path / "ds", epochs=1, batch_size=2, seed=0)
    predictions = predict_yolo(run.model, images[:3])

    assert set(predictions.keys()) == {image.image_id for image in images[:3]}
    for preds in predictions.values():
        for prediction in preds:
            assert prediction.class_name == "defect"
            assert 0.0 <= prediction.score <= 1.0
            x0, y0, x1, y1 = prediction.box
            assert x0 <= x1
            assert y0 <= y1
