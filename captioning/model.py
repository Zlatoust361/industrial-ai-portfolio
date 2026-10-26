"""Instruction-tuned VLM captioning/VQA loader for edge defect description.

Same `VLMLoader` contract as `src.models.clip.CLIPLoader` (`load()`/
`unload()`, memory budget checked before and after loading weights via
`src.models.memory`) -- this week's "second model next to the detector"
is still just one more edge-memory-constrained model load, no new loader
abstraction needed. But captioning is GENERATIVE (image -> text), not
embedding, so there is no matching `VLMInference` subclass here:
forcing a caption call onto `encode_image`/`encode_text` would be the
wrong abstraction fit -- see `tracking/adapters.py`'s docstring for the
same kind of "don't force-fit an interface that doesn't match" judgment
call made elsewhere in this repo. `generate_caption_text` below is the
captioning-shaped equivalent instead.

No top-level `import torch`/`transformers` -- lazy, same convention as
every other module in this repository (see `src/models/clip.py`).
"""

from __future__ import annotations

from PIL import Image

from src.models.base import VLMLoader
from src.models.memory import (
    available_system_memory_mb,
    check_memory_budget,
    check_process_memory,
    process_resident_memory_mb,
)
from src.models.seed import set_deterministic_seed


class CaptioningLoader(VLMLoader):
    """Loads a small instruction-tuned image-captioning/VQA checkpoint."""

    def load(self) -> None:
        """Load the captioning model/processor, applying seed/precision/device from config.

        Raises:
            src.models.memory.MemoryBudgetExceededError: If available
                system memory can't cover `self.memory.memory_budget_mb`
                before loading, or the loaded model already exceeds it
                afterwards. This budget is ADDITIVE on top of whatever
                the detector pipeline (weeks 03/04/09) is already using
                on the same device -- see `captioning/README.md`'s
                Jetson-reality note for why these two budgets are checked
                independently, not summed automatically here.
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        set_deterministic_seed(self.config.seed)
        check_memory_budget(self.memory, available_mb=available_system_memory_mb())

        dtype = torch.float16 if self.config.precision == "fp16" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir,
            torch_dtype=dtype,
        )
        model.to(self.config.device)
        model.eval()

        self._model = model
        self._processor = AutoProcessor.from_pretrained(
            self.config.name, cache_dir=self.config.cache_dir
        )

        check_process_memory(
            self.memory,
            process_rss_mb=process_resident_memory_mb(),
            stage="after model load",
        )

    def unload(self) -> None:
        """Release model resources so they can be garbage collected."""
        self._model = None
        self._processor = None


def generate_caption_text(
    loader: CaptioningLoader, image: Image.Image, prompt: str, max_new_tokens: int = 40
) -> str:
    """Run one real captioning/VQA generation call, loading the model lazily on first use.

    This is the REAL `CaptionFn` (see `captioning/prompting.py`'s
    `CaptionFn` protocol) -- bound via `functools.partial` at the one real
    call site (`captioning/scripts/run_captioning_demo.py`'s default,
    not its tests, which inject a fake `CaptionFn` instead).

    Args:
        loader: A `CaptioningLoader` (loaded lazily if not already).
        image: The defect image to describe.
        prompt: The constrained instruction prompt (see
            `captioning/prompting.py::build_prompt`).
        max_new_tokens: Generation length cap -- short by design, this
            week's deliverable is a one-line description, not a paragraph.

    Returns:
        The generated text, stripped of leading/trailing whitespace.
    """
    if not loader.is_loaded:
        loader.load()
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    chat_prompt = loader.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = loader.processor(images=image, text=chat_prompt, return_tensors="pt")
    inputs = inputs.to(loader.config.device)

    with torch.no_grad():
        output_ids = loader.model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = loader.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()
