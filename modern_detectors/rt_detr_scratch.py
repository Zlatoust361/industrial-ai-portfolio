"""A real, tiny, from-scratch RT-DETR, trained on week 25's real synthetic defects.

Uses the real `transformers.RTDetrForObjectDetection` architecture (Hungarian
matching + set-prediction loss, anchor-free NMS-free head -- not a
reimplementation), but with a deliberately tiny custom backbone/encoder/
decoder (`build_tiny_rt_detr_config`) so a training step runs in well
under a second on this sandbox's CPU (measured ~0.19s/step at batch=4,
224x224 -- see `modern_detectors/README.md`). This is a toy-scale model,
not `PekingU/rtdetr_r50vd` or similar production checkpoint: the point of
week 26 is the real anchor-free/set-prediction MECHANISM and a real,
reproducible training curve, not a claim of production accuracy. No
network call anywhere in this module -- `RTDetrConfig`/
`RTDetrResNetConfig` are plain Python objects, not `from_pretrained` calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    RTDetrConfig,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrResNetConfig,
)

from classical_cv.defect_images import Box
from modern_detectors.dataset import DEFECT_CLASS_INDEX, TRAIN_IMAGE_SIZE, ResizedImage
from src.evaluation.detection_metrics import Prediction
from src.models.seed import set_deterministic_seed

TINY_BACKBONE_HIDDEN_SIZES = (8, 16, 32, 64)
TINY_BACKBONE_DEPTHS = (1, 1, 1, 1)
TINY_D_MODEL = 32
TINY_NUM_QUERIES = 10
"""`num_queries` caps how many objects RT-DETR can report per image at
once -- 10 is comfortably above `config/defects.yaml`'s `count_range`
(1-2 defects/image), no real-world limitation for this dataset."""


@dataclass(frozen=True)
class RtDetrTrainingRun:
    """Outcome of `train_rt_detr`.

    Attributes:
        model: The trained model, ready for `predict_rt_detr`.
        step_losses: Real training loss at every step, in order --
            `benchmark.py`/the demo script report whether it actually
            decreased, not just a final number.
    """

    model: RTDetrForObjectDetection
    step_losses: tuple[float, ...]


def build_tiny_rt_detr_config() -> RTDetrConfig:
    """Build a small `RTDetrConfig` with a custom (non-`timm`) tiny backbone.

    Returns:
        A config whose `RTDetrForObjectDetection(config)` builds a
        ~280K-parameter model, no network/`timm` dependency (the default
        RT-DETR backbone is a `timm` ResNet, not installed in this
        environment -- this config avoids it entirely via an explicit
        small `RTDetrResNetConfig`).
    """
    backbone_config = RTDetrResNetConfig(
        num_channels=3,
        embedding_size=TINY_BACKBONE_HIDDEN_SIZES[0],
        hidden_sizes=list(TINY_BACKBONE_HIDDEN_SIZES),
        depths=list(TINY_BACKBONE_DEPTHS),
        layer_type="basic",
        out_features=["stage2", "stage3", "stage4"],
    )
    return RTDetrConfig(
        backbone_config=backbone_config,
        num_labels=1,
        num_queries=TINY_NUM_QUERIES,
        d_model=TINY_D_MODEL,
        encoder_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        decoder_attention_heads=2,
        encoder_hidden_dim=TINY_D_MODEL,
        encoder_in_channels=list(TINY_BACKBONE_HIDDEN_SIZES[1:]),
        decoder_in_channels=[TINY_D_MODEL, TINY_D_MODEL, TINY_D_MODEL],
    )


def build_tiny_rt_detr_model(seed: int) -> RTDetrForObjectDetection:
    """Build a randomly-initialized tiny RT-DETR, seeded for reproducibility.

    Args:
        seed: RNG seed for weight initialization.

    Returns:
        An untrained `RTDetrForObjectDetection` on CPU.
    """
    set_deterministic_seed(seed)
    model = RTDetrForObjectDetection(build_tiny_rt_detr_config())
    model.train()
    return model


def _to_pixel_values(images: list[ResizedImage]) -> torch.Tensor:
    """`(N, H, W, 3)` uint8 RGB arrays -> `(N, 3, H, W)` float tensor in `[0, 1]`."""
    stacked = np.stack([image.rgb_pixels for image in images]).astype(np.float32) / 255.0
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


def _to_rt_detr_labels(images: list[ResizedImage]) -> list[dict[str, torch.Tensor]]:
    """Ground truth boxes -> RT-DETR's expected `labels` format.

    `transformers`' DETR-family loss expects normalized `(cx, cy, w, h)`
    in `[0, 1]`, one dict per image with `"class_labels"`/`"boxes"` --
    an image with zero ground truth boxes still needs an entry (empty
    tensors), not an omitted one, or the Hungarian matcher's batch
    indexing breaks.
    """
    labels = []
    for image in images:
        boxes = []
        for x0, y0, x1, y1 in image.ground_truth_boxes:
            cx = (x0 + x1) / 2 / TRAIN_IMAGE_SIZE
            cy = (y0 + y1) / 2 / TRAIN_IMAGE_SIZE
            box_w = (x1 - x0) / TRAIN_IMAGE_SIZE
            box_h = (y1 - y0) / TRAIN_IMAGE_SIZE
            boxes.append([cx, cy, box_w, box_h])
        labels.append(
            {
                "class_labels": torch.full((len(boxes),), DEFECT_CLASS_INDEX, dtype=torch.long),
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            }
        )
    return labels


def train_rt_detr(
    model: RTDetrForObjectDetection,
    images: list[ResizedImage],
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> RtDetrTrainingRun:
    """Train `model` for real, a fixed number of steps, on `images`.

    Args:
        model: An untrained (or already-trained) model from `build_tiny_rt_detr_model`.
        images: Training images -- resampled with replacement into
            batches (small dataset, short run; this is the same
            trade-off `federated/local_client.py`'s few-step real
            training makes).
        num_steps: Number of optimizer steps.
        batch_size: Images per step.
        learning_rate: AdamW learning rate.
        seed: RNG seed for batch sampling.

    Returns:
        An `RtDetrTrainingRun` with the trained `model` and every step's
        real loss value.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    step_losses = []
    for _ in range(num_steps):
        batch_indices = rng.integers(0, len(images), size=batch_size)
        batch = [images[i] for i in batch_indices]

        pixel_values = _to_pixel_values(batch)
        labels = _to_rt_detr_labels(batch)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        step_losses.append(float(outputs.loss.item()))

    return RtDetrTrainingRun(model=model, step_losses=tuple(step_losses))


def predict_rt_detr(
    model: RTDetrForObjectDetection, images: list[ResizedImage], score_threshold: float = 0.0
) -> dict[str, list[Prediction]]:
    """Run inference, decoding raw outputs into pixel-space `Prediction`s.

    Args:
        model: A trained (or untrained, for a sanity check) model.
        images: Images to run inference on (`TRAIN_IMAGE_SIZE` each).
        score_threshold: Forwarded to `RTDetrImageProcessor.post_process_object_detection`
            -- `0.0` keeps every query's prediction (scoring code picks
            its own operating point afterward), a higher value mimics an
            actual deployed confidence cutoff.

    Returns:
        `image_id -> list[Prediction]`, boxes in `TRAIN_IMAGE_SIZE` pixel
        space (same space `modern_detectors.dataset.ResizedImage`'s
        ground truth boxes use).
    """
    model.eval()
    processor = RTDetrImageProcessor()
    pixel_values = _to_pixel_values(images)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    target_sizes = [(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)] * len(images)
    decoded = processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )

    predictions: dict[str, list[Prediction]] = {}
    for image, result in zip(images, decoded, strict=True):
        boxes: list[Prediction] = []
        for score, box in zip(result["scores"], result["boxes"], strict=True):
            x0, y0, x1, y1 = (float(v) for v in box)
            box_t: Box = (x0, y0, x1, y1)
            boxes.append(
                Prediction(
                    image_id=image.image_id, class_name="defect", score=float(score), box=box_t
                )
            )
        predictions[image.image_id] = boxes
    return predictions
