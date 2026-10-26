"""On-device OWL-ViT -> TensorRT engine conversion for the vision_bridge node.

MUST run on the actual Jetson Orin Nano the engine will be deployed on:
`.engine` files are NOT portable across GPU architectures or TensorRT
versions -- a build produced on a dev GPU will fail to load on the Orin
(see `vision_bridge.tensorrt_engine.TensorRTEngineError`). This script
refuses to proceed unless it detects it's actually running on a Jetson
(checks `/etc/nv_tegra_release`, the standard JetPack marker file), unless
`--force` is passed.

Two steps:
  1. Export `OwlViTForObjectDetection.owlvit.vision_model` -- the CLIP-style
     ViT backbone, same architecture week 01's CLIP image encoder targets
     (see `scripts/02_convert_onnx.py` at the repo root) -- to ONNX, at
     batch=1 (one camera frame at a time; see `course/DECISIONS.md` ADR-003
     for the same edge-batching reasoning applied there).
  2. Build a TensorRT engine from that ONNX via the TensorRT Python builder
     API (fp16 by default).

Usage (run on the Jetson):
    PYTHONPATH=../src/vision_bridge python3 convert_owlvit_tensorrt.py \\
        --model-name google/owlvit-base-patch32 \\
        --onnx-path models/owlvit_vision.onnx \\
        --engine-path models/owlvit_vision.engine
    # then point vision_bridge at it:
    #   ros2 launch vision_bridge vision_bridge.launch.py \\
    #       engine_path:=$(pwd)/models/owlvit_vision.engine

Not runnable in this repo's dev sandbox (no GPU/CUDA/TensorRT, and
huggingface.co is network-blocked -- see `course/DECISIONS.md` ADR-004);
reviewed carefully against documented TensorRT/transformers API usage but
not executed. Verify on the actual Jetson before relying on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

JETSON_RELEASE_FILE = Path("/etc/nv_tegra_release")


def is_running_on_jetson() -> bool:
    """Best-effort check for actually running on a Jetson device.

    Returns:
        `True` if `/etc/nv_tegra_release` exists (the standard JetPack
        marker file), `False` otherwise.
    """
    return JETSON_RELEASE_FILE.exists()


def export_vision_tower_to_onnx(model_name: str, onnx_path: Path, image_size: int) -> None:
    """Export `OwlViTForObjectDetection.owlvit.vision_model` to ONNX at batch=1.

    Exports `last_hidden_state` (the patch-level feature map the detection
    heads need), not a pooled embedding -- see `vision_bridge.
    owlvit_tensorrt` for how the runtime side consumes this.

    Args:
        model_name: HuggingFace Hub id of the OWL-ViT checkpoint.
        onnx_path: Destination `.onnx` path; parent dirs are created.
        image_size: Square input resolution (OWL-ViT's default is 768).
    """
    import torch
    from torch import nn
    from transformers import OwlViTForObjectDetection

    model = OwlViTForObjectDetection.from_pretrained(model_name).eval()
    vision_model = model.owlvit.vision_model

    class VisionTowerWrapper(nn.Module):
        def __init__(self, vision_model: nn.Module) -> None:
            super().__init__()
            self.vision_model = vision_model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.vision_model(pixel_values=pixel_values).last_hidden_state

    wrapper = VisionTowerWrapper(vision_model).eval()
    dummy_input = torch.zeros(1, 3, image_size, image_size)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        opset_version=17,
    )
    print(f"Exported vision tower ONNX to '{onnx_path}'.")


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, fp16: bool) -> None:
    """Build a TensorRT engine from `onnx_path`, saving to `engine_path`.

    Args:
        onnx_path: Path to the exported vision-tower ONNX graph.
        engine_path: Destination `.engine` path; parent dirs are created.
        fp16: Build with the FP16 flag if the platform supports it (matches
            the Jetson Orin Nano's typical on-device precision).

    Raises:
        RuntimeError: If parsing the ONNX graph or building the engine fails.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as onnx_file:
        if not parser.parse(onnx_file.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX '{onnx_path}': {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("WARNING: this device reports no fast-fp16 support; building fp32 instead.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT engine build failed for '{onnx_path}'.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as engine_file:
        engine_file.write(serialized_engine)
    size_mb = engine_path.stat().st_size / 1024**2
    print(f"Built TensorRT engine at '{engine_path}' ({size_mb:.1f} MB).")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `convert_owlvit_tensorrt.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-name", default="google/owlvit-base-patch32")
    parser.add_argument("--onnx-path", default="models/owlvit_vision.onnx")
    parser.add_argument("--engine-path", default="models/owlvit_vision.engine")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument(
        "--fp32", action="store_true", help="Build fp32 instead of the fp16 default."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the on-Jetson check (the resulting engine is useless off-device; "
        "only useful to test the ONNX export step).",
    )
    return parser.parse_args()


def main() -> None:
    """Export OWL-ViT's vision tower to ONNX, then build a TensorRT engine from it."""
    args = parse_args()

    if not args.force and not is_running_on_jetson():
        print(
            f"Error: this does not look like a Jetson device ({JETSON_RELEASE_FILE} "
            "not found). TensorRT engines are NOT portable across GPU architectures/"
            "TensorRT versions -- building here would produce a .engine that fails "
            "to load on the Orin Nano. Run this on the Orin Nano itself, or pass "
            "--force to override (e.g. to test just the ONNX export step).",
            file=sys.stderr,
        )
        sys.exit(1)

    onnx_path = Path(args.onnx_path)
    engine_path = Path(args.engine_path)

    export_vision_tower_to_onnx(args.model_name, onnx_path, args.image_size)
    build_tensorrt_engine(onnx_path, engine_path, fp16=not args.fp32)


if __name__ == "__main__":
    main()
