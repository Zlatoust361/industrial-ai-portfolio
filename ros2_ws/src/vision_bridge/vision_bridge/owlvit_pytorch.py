"""OWL-ViT PyTorch backend: eager `transformers.OwlViTForObjectDetection`.

The fallback path when no usable TensorRT engine is available (see
`detector_factory.build_detector`), and the only viable backend on a
dev machine, where a Jetson-built `.engine` wouldn't load anyway (see
`tensorrt_engine.TensorRTEngineError`).
"""

from __future__ import annotations

import numpy as np

from vision_bridge.detection import Detection
from vision_bridge.owlvit_postprocess import build_detections


class PyTorchOwlViTDetector:
    """`Detector` backend running OWL-ViT via HuggingFace `transformers`, in eager PyTorch."""

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

    def detect(self, image_rgb: np.ndarray, text_query: str) -> list[Detection]:
        """Run OWL-ViT end to end (vision + text + postprocess) in PyTorch."""
        torch = self._torch
        inputs = self._processor(text=[text_query], images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([image_rgb.shape[:2]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=self._score_threshold, target_sizes=target_sizes
        )[0]

        scores = results["scores"].detach().cpu().numpy()
        boxes = results["boxes"].detach().cpu().numpy()
        return build_detections(
            scores, boxes, label=text_query, score_threshold=self._score_threshold
        )
