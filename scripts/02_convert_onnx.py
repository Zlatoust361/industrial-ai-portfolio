"""Export the CLIP image encoder to ONNX and smoke-test it against PyTorch.

The exported graph marks the batch axis dynamic (so it *accepts* any batch
size at inference time), but both the export trace and the smoke test are
run at batch=1 by default -- the primary edge scenario on the Jetson Orin
Nano, which reads a single video stream and has no real batching to do.

Usage:
    PYTHONPATH=. python scripts/02_convert_onnx.py [--config config/model_config.yaml] \\
        [--batch-size 1] [--atol 0.001] [--rtol 0.001] [--skip-smoke-test]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import AppConfig, load_config
from src.evaluation.onnx_parity import check_output_parity
from src.models.clip import CLIPLoader, extract_embedding_tensor

EDGE_BATCH_SIZE = 1
"""Default export/smoke-test batch size: one frame at a time, the on-device scenario."""


def build_image_encoder(clip_model: Any) -> Any:
    """Wrap `CLIPModel.get_image_features` as a single-input/output `nn.Module`.

    `torch.onnx.export` traces a plain forward pass, so the multi-argument,
    dict-returning `CLIPModel.forward` needs this thin wrapper to expose a
    single `pixel_values -> image_embeds` signature.

    Args:
        clip_model: A loaded, eval-mode `transformers.CLIPModel`.

    Returns:
        An `nn.Module` mapping `pixel_values -> image_embeds` (unnormalized,
        matching `CLIPModel.get_image_features`), ready for `torch.onnx.export`.
    """
    from torch import nn

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
    encoder: Any,
    dummy_input: Any,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    """Export `encoder` to ONNX at `output_path`.

    Args:
        encoder: The `nn.Module` to export (see `build_image_encoder`).
        dummy_input: Representative input tensor used to trace the graph
            (its batch dimension is just the trace batch size, not a cap).
        output_path: Destination `.onnx` file path; parent dirs are created.
        opset: ONNX opset version.
        dynamic_batch: If `True`, mark the batch axis (dim 0) dynamic on
            both input and output, so the exported graph accepts any batch
            size, not just `dummy_input`'s.
    """
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}} if dynamic_batch else None
    )
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )


def run_smoke_test(
    encoder: Any, onnx_path: Path, dummy_input: Any, atol: float, rtol: float
) -> None:
    """Run `dummy_input` through both PyTorch and ONNX Runtime and compare outputs.

    Args:
        encoder: The same `nn.Module` instance that was exported (still in eval mode).
        onnx_path: Path to the exported `.onnx` file.
        dummy_input: Input tensor to run through both backends (the same one used for export).
        atol: Absolute tolerance for the comparison.
        rtol: Relative tolerance for the comparison.

    Raises:
        src.evaluation.onnx_parity.OnnxParityError: If outputs disagree beyond tolerance.
    """
    import onnxruntime as ort
    import torch

    with torch.no_grad():
        pytorch_output = encoder(dummy_input).float().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs = {"pixel_values": dummy_input.float().cpu().numpy()}
    onnxruntime_output = session.run(["image_embeds"], onnx_inputs)[0]

    check_output_parity(pytorch_output, onnxruntime_output, atol=atol, rtol=rtol)


def build_dummy_input(config: AppConfig, batch_size: int, dtype: Any) -> Any:
    """Build the all-zero dummy input tensor used to trace/smoke-test the export.

    Args:
        config: Loaded application config (for `inference.image_size`).
        batch_size: Batch dimension of the dummy tensor.
        dtype: Torch dtype to match the loaded model's weights.

    Returns:
        A `(batch_size, 3, image_size, image_size)` zero tensor.
    """
    import torch

    return torch.zeros(
        batch_size, 3, config.inference.image_size, config.inference.image_size, dtype=dtype
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `02_convert_onnx.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EDGE_BATCH_SIZE,
        help="Batch size for the export trace and smoke test (default: 1, the on-device scenario).",
    )
    parser.add_argument(
        "--atol", type=float, default=None, help="Overrides export.onnx_atol from config."
    )
    parser.add_argument(
        "--rtol", type=float, default=None, help="Overrides export.onnx_rtol from config."
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Export only; skip the PyTorch-vs-ONNXRuntime comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configured CLIP checkpoint, export its image encoder, and smoke-test it."""
    args = parse_args()

    config = load_config(args.config)
    loader = CLIPLoader(config.model, config.memory)
    loader.load()

    encoder = build_image_encoder(loader.model)
    dummy_input = build_dummy_input(
        config, batch_size=args.batch_size, dtype=next(loader.model.parameters()).dtype
    )

    output_path = config.export.onnx_output_path
    export_to_onnx(
        encoder,
        dummy_input,
        output_path,
        opset=config.export.onnx_opset,
        dynamic_batch=config.export.dynamic_batch,
    )
    print(
        f"Exported ONNX image encoder to '{output_path}' "
        f"(trace batch={args.batch_size}, dynamic_batch={config.export.dynamic_batch})."
    )

    if args.skip_smoke_test:
        return

    atol = args.atol if args.atol is not None else config.export.onnx_atol
    rtol = args.rtol if args.rtol is not None else config.export.onnx_rtol
    run_smoke_test(encoder, output_path, dummy_input, atol=atol, rtol=rtol)
    print(f"Smoke test passed: PyTorch vs ONNXRuntime match within atol={atol}, rtol={rtol}.")


if __name__ == "__main__":
    main()
