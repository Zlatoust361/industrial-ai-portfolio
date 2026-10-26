"""CLIP implementation of the VLMLoader / VLMInference pattern.

Heavy dependencies (`torch`, `transformers`) are imported lazily inside the
methods below, so importing this module -- and importing `src.models.base`
in particular -- never requires them to be installed.
"""

from __future__ import annotations

from PIL import Image

from src.models.base import EmbeddingResult, VLMInference, VLMLoader
from src.models.memory import (
    available_system_memory_mb,
    check_memory_budget,
    check_process_memory,
    process_resident_memory_mb,
)
from src.models.seed import set_deterministic_seed


def extract_embedding_tensor(output: object) -> object:
    """Unwrap `CLIPModel.get_image_features`/`get_text_features`'s return value.

    Historically (`transformers<5`) both methods return a bare projected
    embedding tensor. `transformers>=5` (confirmed empirically against
    5.17.0 -- real installed weights on real hardware, not assumed) wraps
    the same already-projected embedding in a
    `transformers.modeling_outputs.BaseModelOutputWithPooling`, with the
    embedding itself under `.pooler_output` (its shape matches
    `projection_dim`, not the vision/text tower's raw `hidden_size` --
    this is the final embedding, not an earlier, unprojected stage).
    Handling both shapes here, once, keeps every caller of
    `get_image_features`/`get_text_features` (this module's
    `encode_image`/`encode_text`, and the ONNX-export wrappers in
    `scripts/02_convert_onnx.py` and `quantization/scripts/02_export_onnx.py`,
    and `scripts/optimize_prompts.py`) correct across the transformers
    major-version boundary instead of silently breaking on whichever
    version happens to be installed. Public (no leading underscore)
    precisely because those other modules need to import it too.

    Args:
        output: Whatever `get_image_features`/`get_text_features` returned.

    Returns:
        The embedding tensor itself.

    Raises:
        TypeError: If `output` is neither a bare tensor nor an object
            exposing a `.pooler_output` tensor -- a future transformers
            release changing shape again should fail loudly here, not
            produce a silently wrong embedding.
    """
    import torch

    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError(
        f"Unexpected return type from get_image_features()/get_text_features(): "
        f"{type(output)!r} -- expected a torch.Tensor (transformers<5) or an "
        f"object with a .pooler_output tensor (transformers>=5)."
    )


class CLIPLoader(VLMLoader):
    """Loads a HuggingFace CLIP checkpoint (model + processor) for edge inference."""

    def load(self) -> None:
        """Load the CLIP model/processor, applying seed, precision and device from config.

        Raises:
            src.models.memory.MemoryBudgetExceededError: If available system
                memory can't cover `self.memory.memory_budget_mb` before
                loading, or the loaded model already exceeds it afterwards.
        """
        import torch
        from transformers import CLIPModel, CLIPProcessor

        set_deterministic_seed(self.config.seed)
        check_memory_budget(self.memory, available_mb=available_system_memory_mb())

        dtype = torch.float16 if self.config.precision == "fp16" else torch.float32
        model = CLIPModel.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir,
            torch_dtype=dtype,
        )
        model.to(self.config.device)
        model.eval()

        self._model = model
        self._processor = CLIPProcessor.from_pretrained(
            self.config.name, cache_dir=self.config.cache_dir
        )

        check_process_memory(
            self.memory,
            process_rss_mb=process_resident_memory_mb(),
            stage="after model load",
        )

    def unload(self) -> None:
        """Drop references to the model/processor so they can be garbage collected."""
        self._model = None
        self._processor = None


class CLIPInference(VLMInference):
    """Unified image/text embedding interface backed by a `CLIPLoader`."""

    def encode_image(self, images: list[Image.Image]) -> EmbeddingResult:
        """Encode a batch of PIL images into L2-normalized CLIP image embeddings."""
        self._ensure_loaded()
        import torch

        inputs = self.loader.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.loader.config.device)

        with torch.no_grad():
            features = extract_embedding_tensor(self.loader.model.get_image_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)

        return EmbeddingResult(
            vectors=features.float().cpu().numpy(),
            model_name=self.loader.config.name,
        )

    def encode_text(self, texts: list[str]) -> EmbeddingResult:
        """Encode a batch of text prompts into L2-normalized CLIP text embeddings."""
        self._ensure_loaded()
        import torch

        inputs = self.loader.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = inputs.to(self.loader.config.device)

        with torch.no_grad():
            features = extract_embedding_tensor(self.loader.model.get_text_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)

        return EmbeddingResult(
            vectors=features.float().cpu().numpy(),
            model_name=self.loader.config.name,
        )
