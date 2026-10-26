"""OWL-ViT TensorRT backend: vision tower runs as a TensorRT engine.

Only the vision forward pass -- the part that runs on every video frame --
is TensorRT-accelerated. Text encoding and the detection heads (box/class
prediction, post-processing) stay in HuggingFace `transformers`/PyTorch:
they're comparatively cheap, only rerun when the query changes, and this
way the model's own real box/class-head logic runs unmodified instead of
this module reimplementing it from a guess at OWL-ViT's internals.

How the pieces are combined: `OwlViTForObjectDetection.owlvit.vision_model`
is a CLIP-style vision transformer (the same architecture week 01's CLIP
image encoder targets) whose public contract is `forward(pixel_values) ->
object with .last_hidden_state`. This backend monkeypatches just that one
method on the loaded model to return the TensorRT engine's output instead
of running the real ViT, then calls the full, unmodified
`OwlViTForObjectDetection.forward()` -- so every downstream computation
(box regression, class prediction, thresholding) is the model's own real,
correct, currently-installed implementation, not a hand reimplementation
of it. This is deliberately the smallest, most stable interface to depend
on; it has not been run against a real checkpoint in this sandbox (no
network access to download OWL-ViT weights -- see `course/DECISIONS.md`),
so verify `vision_model`'s exact output contract against the installed
`transformers` version on-device before relying on this.
"""

from __future__ import annotations

import numpy as np

from vision_bridge.detection import Detection
from vision_bridge.owlvit_postprocess import build_detections
from vision_bridge.tensorrt_engine import TensorRTEngine


class TensorRTOwlViTDetector:
    """`Detector` backend: TensorRT vision tower + PyTorch text tower/heads/postprocess."""

    def __init__(
        self,
        engine_path: str,
        model_name: str = "google/owlvit-base-patch32",
        score_threshold: float = 0.1,
    ) -> None:
        """Args:
            engine_path: Path to the vision-tower `.engine` (see
                `ros2_ws/scripts/convert_owlvit_tensorrt.py`). Checked for
                existence, and the engine deserialized, before any
                `torch`/`transformers` import -- so a missing/mismatched
                engine fails fast without needing them installed at all.
            model_name: HuggingFace Hub id of the *matching* OWL-ViT
                checkpoint the engine was exported from (for the processor,
                text tower, and detection heads).
            score_threshold: Minimum detection confidence to report.

        Raises:
            FileNotFoundError: If `engine_path` doesn't exist.
            vision_bridge.tensorrt_engine.TensorRTEngineError: If the engine
                fails to load -- most commonly a GPU-architecture/TensorRT-
                version mismatch. Callers (see `detector_factory.
                build_detector`) should catch this and fall back to
                `owlvit_pytorch.PyTorchOwlViTDetector`.
        """
        self._engine = TensorRTEngine(engine_path)  # fails fast, before the imports below

        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self._torch = torch
        self._score_threshold = score_threshold

        self._processor = OwlViTProcessor.from_pretrained(model_name)
        # The full model is kept only for its text tower, detection heads,
        # and post_process_object_detection -- its own vision_model.forward
        # is replaced per-call below and never actually runs its ViT.
        self._model = OwlViTForObjectDetection.from_pretrained(model_name).eval()

    def detect(self, image_rgb: np.ndarray, text_query: str) -> list[Detection]:
        """Run the vision tower via TensorRT, everything else via PyTorch/transformers."""
        torch = self._torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        inputs = self._processor(text=[text_query], images=image_rgb, return_tensors="pt")

        engine_outputs = self._engine.infer(inputs["pixel_values"].numpy())
        last_hidden_state = torch.from_numpy(engine_outputs["last_hidden_state"])

        vision_model = self._model.owlvit.vision_model
        original_forward = vision_model.forward

        def _tensorrt_vision_forward(
            *_args: object, **_kwargs: object
        ) -> BaseModelOutputWithPooling:
            # pooler_output is left None: OWL-ViT's detection heads consume
            # the patch-level last_hidden_state, not a pooled embedding --
            # verify against the installed transformers version if this
            # assumption turns out wrong on-device (see module docstring).
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state, pooler_output=None
            )

        vision_model.forward = _tensorrt_vision_forward
        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
        finally:
            vision_model.forward = original_forward

        target_sizes = torch.tensor([image_rgb.shape[:2]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=self._score_threshold, target_sizes=target_sizes
        )[0]

        scores = results["scores"].detach().cpu().numpy()
        boxes = results["boxes"].detach().cpu().numpy()
        return build_detections(
            scores, boxes, label=text_query, score_threshold=self._score_threshold
        )
