"""Multi-class OWL-ViT wrapper, for compare_owlvit_yolov8.py only.

Different from `vision_bridge.owlvit_pytorch`/`owlvit_tensorrt` (the ROS2
node's `Detector` backends, which handle one text query at a time for live
streaming): this evaluates against every test-set class name at once, in a
single forward pass per image -- matching how YOLOv8 also predicts all of
its classes at once, needed for a fair mAP comparison. Uses OWL-ViT's own
multi-query support (`OwlViTProcessor(text=[[q1, q2, ...]], ...)`) rather
than looping one query at a time per image.

PyTorch-only: TensorRT vision-tower acceleration (see
`vision_bridge.owlvit_tensorrt`) is a possible follow-up, not implemented
here -- this script's job is comparing OWL-ViT vs YOLOv8 as detectors, not
re-deriving the (already unverified, see its module docstring) TensorRT
multi-query pipeline on top of that.

Not runnable in this sandbox: no network access to download OWL-ViT
weights (see course/DECISIONS.md ADR-004). Written from documented
`transformers` OWL-ViT usage, not executed here.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


class OwlViTMultiClassDetector:
    """Runs OWL-ViT against a fixed list of class names, one forward pass per image."""

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        score_threshold: float = 0.1,
        device: str = "cpu",
    ) -> None:
        """Args:
            model_name: HuggingFace Hub id of the OWL-ViT checkpoint.
            score_threshold: Minimum detection confidence to report.
            device: `"cpu"` or `"cuda"`.
        """
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self._torch = torch
        self._score_threshold = score_threshold
        self._device = device

        self._processor = OwlViTProcessor.from_pretrained(model_name)
        self._model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()

    def detect(self, image_rgb: np.ndarray, class_names: list[str]) -> list[tuple[str, float, Box]]:
        """Detect every class in `class_names` present in `image_rgb`.

        Args:
            image_rgb: An `(height, width, 3)` uint8 RGB image array.
            class_names: Every class to query for, in one forward pass.

        Returns:
            `(class_name, score, box_xyxy)` tuples with `score >= score_threshold`.
        """
        torch = self._torch
        inputs = self._processor(text=[class_names], images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([image_rgb.shape[:2]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=self._score_threshold, target_sizes=target_sizes
        )[0]

        scores = results["scores"].detach().cpu().numpy()
        boxes = results["boxes"].detach().cpu().numpy()
        label_indices = results["labels"].detach().cpu().numpy()

        predictions: list[tuple[str, float, Box]] = []
        for score, box, label_index in zip(scores, boxes, label_indices, strict=True):
            if score < self._score_threshold:
                continue
            x_min, y_min, x_max, y_max = (float(v) for v in box)
            predictions.append(
                (class_names[int(label_index)], float(score), (x_min, y_min, x_max, y_max))
            )
        return predictions
