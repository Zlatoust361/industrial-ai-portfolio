"""Real static INT8 quantization for `TinyCNN`, via `torch.ao.quantization` (week 28).

The roadmap for this week asks for a comparison against week 09's INT4/
TensorRT quantization -- but week 09's own numbers in THIS sandbox are
synthetic end-to-end (no `tensorrt`/`pycuda`, no physical Jetson Orin
Nano, see `course/weeks/week09-quantization-tensorrt/README.md`'s own
status), so citing them here would compare a real number against a
fabricated one. Real `torch.ao.quantization` static INT8 (CPU-only,
`fbgemm` backend, no Jetson/TensorRT needed) is the closest available
REAL substitute: a different quantization backend/bit-width than week
09's INT4-on-TensorRT, but a genuinely measured one, not a guessed one.

`torch.ao.quantization`'s eager-mode API is marked deprecated as of
this environment's `torch==2.14.1` (in favor of `torchao`, not
installed -- not in `requirements.txt`, and adding a new heavy
dependency for one week's quantization experiment was judged not
worth it). Used anyway: deprecated-but-functional, and the deprecation
warnings are suppressed here explicitly (not silently) so they don't
drown out this week's own real output.

Real, measured limitation found while building this week:
`AdaptiveMaxPool2d` has no `QuantizedCPU` kernel in this `torch`
version (`NotImplementedError: Could not run 'aten::adaptive_max_pool2d
.out' with arguments from the 'QuantizedCPU' backend`) -- `Quantizable
TinyCNN` dequantizes right after the conv backbone and runs pooling/the
linear head in float, quantizing only the convolutions. A real, common
partial-quantization pattern, not a workaround that defeats the point.
"""

from __future__ import annotations

import warnings

import torch
import torch.ao.quantization as tq
import torch.nn as nn

from distillation.models import TinyCNN


class QuantizableTinyCNN(nn.Module):
    """A `TinyCNN`-equivalent wrapped with `QuantStub`/`DeQuantStub` for the static-quant flow.

    `QuantStub`/`DeQuantStub` are identity ops until `torch.ao.quantization
    .prepare`/`.convert` swap them for real quantize/dequantize
    operators -- kept in a separate subclass (not added to `TinyCNN`
    itself) so the rest of this week's modules (`distillation.py`,
    `pruning.py`, `nas.py`) never need to import `torch.ao.quantization`
    at all.
    """

    def __init__(self, widths: tuple[int, ...], input_channels: int = 1) -> None:
        super().__init__()
        self.widths = widths
        self.quant = tq.QuantStub()

        layers: list[nn.Module] = []
        in_channels = input_channels
        for width in widths:
            layers.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(widths[-1], 1)
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantizes only the conv backbone -- `AdaptiveMaxPool2d` has no
        `QuantizedCPU` kernel in this `torch` version (confirmed empirically
        while building this week, see the module docstring), so `dequant`
        runs right after `features`, and pooling/the linear head stay in
        float. A real, common quantization pattern (not every op needs to
        be quantized), not a workaround that defeats the point."""
        x = self.quant(x)
        x = self.features(x)
        x = self.dequant(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def _copy_weights_from(source: TinyCNN, target: QuantizableTinyCNN) -> None:
    """Copy every real trained weight from `source` into the quant-ready `target`."""
    target.features.load_state_dict(source.features.state_dict())
    target.classifier.load_state_dict(source.classifier.state_dict())


def quantize_int8(model: TinyCNN, calibration_images: torch.Tensor) -> QuantizableTinyCNN:
    """Statically quantize a trained `TinyCNN` to real INT8 weights/activations.

    Args:
        model: An already-trained `TinyCNN`.
        calibration_images: `(N, 1, H, W)` real images (ideally a held-
            out calibration set, not the test set) -- the observers
            inserted by `tq.prepare` need real forward passes through
            real data to estimate activation ranges; quantizing without
            calibration would silently use degenerate (all-zero) ranges.

    Returns:
        A real `QuantizableTinyCNN` with INT8 weights/activations --
        `model_size_kb` on it reports the real, smaller serialized size.
    """
    quantizable = QuantizableTinyCNN(model.widths, input_channels=model.features[0].in_channels)
    _copy_weights_from(model, quantizable)
    quantizable.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        quantizable.qconfig = tq.get_default_qconfig("fbgemm")
        # `forward` dequantizes right after `features` and runs pooling/the
        # linear head in float (see `forward`'s docstring) -- but `qconfig`
        # propagates to every submodule by default, so without this,
        # `tq.convert` below would still swap `classifier` for a quantized
        # Linear that then receives a float (not quantized) input and
        # raises `NotImplementedError: Could not run 'quantized::linear'
        # ... from the 'CPU' backend` (confirmed empirically). Explicitly
        # opting the un-quantized submodules out keeps them real `nn`
        # float modules after `convert`.
        quantizable.pool.qconfig = None
        quantizable.classifier.qconfig = None
        tq.prepare(quantizable, inplace=True)
        with torch.no_grad():
            quantizable(calibration_images)
        tq.convert(quantizable, inplace=True)

    return quantizable
