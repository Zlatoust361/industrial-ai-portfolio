"""Numeric parity check between a PyTorch module and its exported ONNX graph.

Pure comparison logic only -- no `torch`/`onnxruntime` import, so it's
unit-testable without either installed. Running the two backends to produce
the arrays being compared is `scripts/02_convert_onnx.py`'s job.
"""

from __future__ import annotations

import numpy as np


class OnnxParityError(AssertionError):
    """Raised when ONNX Runtime output doesn't match PyTorch output within tolerance."""


def check_output_parity(
    pytorch_output: np.ndarray,
    onnxruntime_output: np.ndarray,
    atol: float,
    rtol: float,
) -> None:
    """Assert two output arrays match within tolerance, with a diagnostic error otherwise.

    Args:
        pytorch_output: Reference output from the original (pre-export) PyTorch module.
        onnxruntime_output: Output from running the exported ONNX graph via ONNX Runtime,
            on the same input.
        atol: Absolute tolerance, forwarded to `numpy.allclose`.
        rtol: Relative tolerance, forwarded to `numpy.allclose`.

    Raises:
        OnnxParityError: If the shapes differ, or values disagree beyond
            `(atol, rtol)` -- most likely an operator (e.g. LayerNorm/GELU)
            that the export/opset combination handles differently than eager
            PyTorch.
    """
    if pytorch_output.shape != onnxruntime_output.shape:
        raise OnnxParityError(
            f"Shape mismatch: PyTorch output {pytorch_output.shape} vs "
            f"ONNXRuntime output {onnxruntime_output.shape}."
        )

    if not np.allclose(pytorch_output, onnxruntime_output, atol=atol, rtol=rtol):
        max_abs_diff = float(np.max(np.abs(pytorch_output - onnxruntime_output)))
        raise OnnxParityError(
            "PyTorch vs ONNXRuntime outputs diverge beyond "
            f"atol={atol}, rtol={rtol} (max abs diff={max_abs_diff:.6g}). "
            "The ONNX export is likely lossy for this model/opset combination "
            "(common culprits: LayerNorm, GELU) -- try a newer opset, or "
            "widen --atol/--rtol if the divergence is expected numerical noise."
        )
