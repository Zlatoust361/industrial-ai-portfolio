#!/usr/bin/env python3
"""Step 2/8: export the FP16 baseline's image encoder to ONNX + smoke-test it.

The ONNX graph `04_quantize_model.py`/`05_build_engine.py` build a TensorRT
engine from. Exported at batch=1 (the Orin Nano's single-video-stream
scenario, same reasoning as week 01's `scripts/02_convert_onnx.py`), with
the batch axis marked dynamic when `onnx.dynamic_batch` is set so the
graph still *accepts* other batch sizes even though the trace itself
doesn't exercise them.

Not gated to the Jetson: ONNX export/ONNXRuntime are portable across
machines -- only the TensorRT-specific steps from `04_quantize_model.py`
onward require the target device (see `../utils/environment.py`).

Input: none (loads the checkpoint itself, same as `01_baseline_inference.py`).
Output: `<paths.onnx_path>`.

Usage:
    python3 scripts/02_export_onnx.py [--config config/config.yaml] [--skip-smoke-test]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))
sys.path.insert(0, str(_QUANTIZATION_ROOT.parent))

from utils.config import QuantizationConfig, load_config  # noqa: E402

EXPORT_BATCH_SIZE = 1
"""Trace/smoke-test batch size: one frame at a time, the on-device scenario."""


def build_image_encoder(clip_model: Any) -> Any:
    """Wrap `CLIPModel.get_image_features` as a single-input/output `nn.Module`.

    Same wrapper as week 01's `scripts/02_convert_onnx.py::build_image_encoder`
    -- duplicated rather than imported (root `scripts/` isn't a shared
    package; only `src/` is, see `../README.md`), not re-derived logic.
    `extract_embedding_tensor` itself lives in `src/` and is imported
    (not duplicated): `transformers>=5`'s `get_image_features` return
    shape (see that function's docstring) would otherwise break this
    wrapper's single-tensor contract the same way it broke week 01's.

    Args:
        clip_model: A loaded, eval-mode `transformers.CLIPModel`.

    Returns:
        An `nn.Module` mapping `pixel_values -> image_embeds`.
    """
    from torch import nn

    from src.models.clip import extract_embedding_tensor

    class ImageEncoder(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.clip_model = model

        def forward(self, pixel_values: Any) -> Any:
            return extract_embedding_tensor(
                self.clip_model.get_image_features(pixel_values=pixel_values)
            )

    return ImageEncoder(clip_model).eval()


def export_to_onnx(
    encoder: Any, dummy_input: Any, output_path: Path, config: QuantizationConfig
) -> None:
    """Export `encoder` to ONNX at `output_path`.

    Args:
        encoder: The `nn.Module` to export (see `build_image_encoder`).
        dummy_input: Representative input tensor used to trace the graph.
        output_path: Destination `.onnx` file path; parent dirs are created.
        config: This week's `QuantizationConfig` (reads `onnx.opset`/`onnx.dynamic_batch`).
    """
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}}
        if config.onnx.dynamic_batch
        else None
    )
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes=dynamic_axes,
        opset_version=config.onnx.opset,
    )


def run_smoke_test(
    encoder: Any, onnx_path: Path, dummy_input: Any, config: QuantizationConfig
) -> None:
    """Run `dummy_input` through both PyTorch and ONNX Runtime and compare outputs.

    Args:
        encoder: The same `nn.Module` instance that was exported (still in eval mode).
        onnx_path: Path to the exported `.onnx` file.
        dummy_input: Input tensor to run through both backends (the same one used for export).
        config: This week's `QuantizationConfig` (reads `onnx.atol`/`onnx.rtol`).

    Raises:
        src.evaluation.onnx_parity.OnnxParityError: If outputs disagree beyond tolerance.
    """
    import onnxruntime as ort
    import torch

    from src.evaluation.onnx_parity import check_output_parity

    with torch.no_grad():
        pytorch_output = encoder(dummy_input).float().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # NOT .float(): the exported graph's `pixel_values` input dtype is whatever
    # dummy_input's dtype was at export_to_onnx() time above (fp16 when
    # model.precision: "fp16", which CLIPLoader honors regardless of device --
    # see src/models/clip.py). Casting the ONNX RUNTIME INPUT to fp32 here
    # while the graph itself was traced/exported with fp16 input is a dtype
    # mismatch ONNX Runtime rejects outright: confirmed on a real Jetson Orin
    # Nano as `InvalidArgument: Unexpected input data type. Actual:
    # (tensor(float)), expected: (tensor(float16))`. pytorch_output's `.float()`
    # above is unrelated -- it only casts the comparison OUTPUT, not the input
    # fed to either backend.
    onnx_inputs = {"pixel_values": dummy_input.cpu().numpy()}
    onnxruntime_output = session.run(["image_embeds"], onnx_inputs)[0]

    check_output_parity(
        pytorch_output, onnxruntime_output, atol=config.onnx.atol, rtol=config.onnx.rtol
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `02_export_onnx.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Export only; skip the PyTorch-vs-ONNXRuntime comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configured CLIP checkpoint, export its image encoder, and smoke-test it."""
    args = parse_args()
    config: QuantizationConfig = load_config(args.config)

    import torch

    from src.config import MemorySection, ModelSection
    from src.models.clip import CLIPLoader

    model_section = ModelSection(
        name=config.model.name,
        type=config.model.type,
        precision=config.model.precision,
        device=config.model.device,
        seed=config.model.seed,
        cache_dir=config.model.cache_dir,
    )
    memory_section = MemorySection(
        memory_budget_mb=config.memory.memory_budget_mb,
        device_total_mb=config.memory.device_total_mb,
    )
    loader = CLIPLoader(model_section, memory_section)
    loader.load()

    encoder = build_image_encoder(loader.model)
    dummy_input = torch.zeros(
        EXPORT_BATCH_SIZE,
        3,
        config.baseline.image_size,
        config.baseline.image_size,
        dtype=next(loader.model.parameters()).dtype,
    )

    output_path = config.paths.onnx_path
    export_to_onnx(encoder, dummy_input, output_path, config)
    print(
        f"Exported ONNX image encoder to '{output_path}' "
        f"(trace batch={EXPORT_BATCH_SIZE}, dynamic_batch={config.onnx.dynamic_batch})."
    )

    if args.skip_smoke_test:
        return

    run_smoke_test(encoder, output_path, dummy_input, config)
    print(
        f"Smoke test passed: PyTorch vs ONNXRuntime match within "
        f"atol={config.onnx.atol}, rtol={config.onnx.rtol}."
    )


if __name__ == "__main__":
    main()
