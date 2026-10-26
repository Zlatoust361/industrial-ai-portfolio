"""Vision layer -- the capstone's single entry point for "what does the camera see" (week 12).

Composition, not new code: everything here is a re-export of an
already-built, already-tested implementation from its own week, so
`scenarios/` (day 3) has one place to import vision from instead of
reaching into `assistant.vision_bridge`/`src.models.clip`/
`src.models.zero_shot` directly and needing to know which week built
which piece.

- `VisionBridge` (week 11 day 3) is the primary interface: embedding +
  zero-shot ok/defect classification over HTTP against the already-
  deployed `src/api/app.py` service (week 08) -- never a second
  in-process CLIP copy, the same reasoning ADR-002/ADR-043 already
  established -- plus in-process open-vocabulary object detection
  reusing week 03's OWL-ViT detector code.
- `CLIPLoader`/`CLIPInference` (weeks 01/08) and
  `classify_by_cosine_similarity` (week 02) are the lower-level
  primitives `VisionBridge` itself is built from, re-exported too for
  callers that need direct in-process access (e.g. `src/api/app.py`
  itself, or an offline script) rather than the HTTP-calling wrapper.
"""

from __future__ import annotations

from assistant.vision_bridge import EmbeddingResult, VisionBridge, VisionBridgeError
from src.models.clip import CLIPInference, CLIPLoader
from src.models.zero_shot import ClassificationResult, classify_by_cosine_similarity

__all__ = [
    "VisionBridge",
    "VisionBridgeError",
    "EmbeddingResult",
    "CLIPLoader",
    "CLIPInference",
    "ClassificationResult",
    "classify_by_cosine_similarity",
]
