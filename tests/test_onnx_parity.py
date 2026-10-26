"""Unit tests for `src.evaluation.onnx_parity.check_output_parity`."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.onnx_parity import OnnxParityError, check_output_parity


def test_check_output_parity_passes_for_identical_arrays() -> None:
    output = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    check_output_parity(output, output.copy(), atol=1e-6, rtol=1e-6)


def test_check_output_parity_passes_within_tolerance() -> None:
    pytorch_output = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    onnxruntime_output = pytorch_output + 1e-5
    check_output_parity(pytorch_output, onnxruntime_output, atol=1e-3, rtol=1e-3)


def test_check_output_parity_raises_when_values_diverge() -> None:
    pytorch_output = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    onnxruntime_output = np.array([[1.0, 2.0, 3.5]], dtype=np.float32)

    with pytest.raises(OnnxParityError, match="diverge"):
        check_output_parity(pytorch_output, onnxruntime_output, atol=1e-3, rtol=1e-3)


def test_check_output_parity_raises_on_shape_mismatch() -> None:
    pytorch_output = np.zeros((1, 512), dtype=np.float32)
    onnxruntime_output = np.zeros((2, 512), dtype=np.float32)

    with pytest.raises(OnnxParityError, match="Shape mismatch"):
        check_output_parity(pytorch_output, onnxruntime_output, atol=1e-3, rtol=1e-3)
