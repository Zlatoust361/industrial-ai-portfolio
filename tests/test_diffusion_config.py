"""Unit tests for `src.synthetic.diffusion_config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.synthetic.diffusion_config import load_diffusion_config


def test_load_diffusion_config_parses_the_real_config_file() -> None:
    config = load_diffusion_config("config/diffusion.yaml")

    assert config.backend == "cv2"
    assert config.prompt.positive
    assert config.prompt.negative
    assert config.cv2.method in ("telea", "ns")
    assert config.diffusers.device == "cuda"
    assert config.stability_api.api_key_env == "STABILITY_API_KEY"
    assert 0.0 <= config.quality_gate.keep_fraction <= 1.0
    assert set(config.quality_gate.weights) == {
        "psnr_outside_mask",
        "ssim_outside_mask",
        "blur_inside_mask",
        "contrast_inside_mask",
    }


def test_load_diffusion_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_diffusion_config("config/does_not_exist.yaml")


_MINIMAL_YAML = """
backend: cv2
prompt:
  positive: a defect
  negative: bad quality
cv2:
  method: telea
  radius: 3
diffusers:
  model_id: some/model
  device: cuda
  num_inference_steps: 30
  guidance_scale: 7.5
  seed: 42
stability_api:
  endpoint: https://example.invalid/inpaint
  api_key_env: SOME_KEY
  output_format: png
  timeout_s: 60.0
quality_gate:
  keep_fraction: {keep_fraction}
  weights:
    psnr_outside_mask: 0.2
    ssim_outside_mask: 0.3
    blur_inside_mask: 0.3
    contrast_inside_mask: 0.2
  ssim_window: 7
"""


def _write_config(tmp_path: Path, backend: str = "cv2", keep_fraction: float = 0.75) -> Path:
    path = tmp_path / "diffusion.yaml"
    text = _MINIMAL_YAML.format(keep_fraction=keep_fraction)
    path.write_text(text.replace("backend: cv2", f"backend: {backend}"))
    return path


def test_load_diffusion_config_rejects_unknown_backend(tmp_path: Path) -> None:
    path = _write_config(tmp_path, backend="magic")
    with pytest.raises(ValueError, match="magic"):
        load_diffusion_config(path)


@pytest.mark.parametrize("keep_fraction", [-0.1, 1.1])
def test_load_diffusion_config_rejects_out_of_range_keep_fraction(
    tmp_path: Path, keep_fraction: float
) -> None:
    path = _write_config(tmp_path, keep_fraction=keep_fraction)
    with pytest.raises(ValueError, match="keep_fraction"):
        load_diffusion_config(path)


def test_load_diffusion_config_roundtrips_all_backend_sections(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    config = load_diffusion_config(path)

    assert config.cv2.radius == 3
    assert config.diffusers.num_inference_steps == 30
    assert config.stability_api.timeout_s == 60.0
