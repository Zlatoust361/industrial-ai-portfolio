"""Reference-free caption-quality check: CLIP text-image similarity, not human review.

Reuses `src.models.clip`'s `CLIPInference` (weeks 01/02's text branch) --
NOT a new embedding model -- to score how well a generated caption
matches the image it describes: the same cosine-similarity primitive
week 02's zero-shot classifier already uses, applied to a generated
caption instead of a fixed prompt. This is explicitly NOT a substitute
for human review (see `captioning/README.md`'s Honesty note) -- a
caption can be fluent, on-topic, and still factually wrong about the
specific defect; CLIPScore-style similarity only checks "does this text
broadly match this image", not correctness of the description.

The already-loaded `inference` object is passed in, not built inside
this module -- `clip_caption_similarity` never imports `torch`/
`transformers` itself and is fully testable with a duck-typed fake.
"""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class _ImageTextEncoder(Protocol):
    """The slice of `src.models.clip.CLIPInference`'s interface this module needs."""

    def encode_image(self, images: list[Image.Image]) -> object: ...

    def encode_text(self, texts: list[str]) -> object: ...


def clip_caption_similarity(
    inference: _ImageTextEncoder, image: Image.Image, caption_text: str
) -> float:
    """Cosine similarity between one image and its generated caption, via CLIP.

    Args:
        inference: An already-loaded `CLIPInference`-shaped object (or a
            test fake with the same two methods) -- loading/unloading is
            the caller's responsibility, same separation of concerns as
            every other `CLIPInference` call site in this repository.
        image: The defect image the caption describes.
        caption_text: The generated caption.

    Returns:
        Cosine similarity in roughly `[-1, 1]` between the image and text
        embeddings (both L2-normalized by `CLIPInference`, so the plain
        dot product IS the cosine similarity -- no extra normalization
        needed here).
    """
    import numpy as np

    image_result = inference.encode_image([image])
    text_result = inference.encode_text([caption_text])
    return float(np.dot(image_result.vectors[0], text_result.vectors[0]))
