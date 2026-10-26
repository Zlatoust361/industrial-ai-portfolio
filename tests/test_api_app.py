"""Integration tests for `src.api.app`'s FastAPI routes.

Uses a real `TestClient` over the real app -- validation and video/RTSP
frame sampling (real OpenCV, no mocking of our own logic) are exercised
for real. Every success path that would reach `CLIPInference.encode_image`/
`encode_text` is smoke-tested only up to `torch` not being installed in
this sandbox (see README.md's week 01 section) -- the same boundary the
rest of this repo hits, not a gap specific to this module.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api.app as api_app_module
from src.api.app import app
from src.data.video import VideoSourceError

client = TestClient(app)


def _make_png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 60, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_mp4_bytes(tmp_path: Path, num_frames: int = 3, size: tuple[int, int] = (16, 12)) -> bytes:
    import cv2

    video_path = tmp_path / "clip.mp4"
    width, height = size
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=(i * 30) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return video_path.read_bytes()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_sets_request_id_header() -> None:
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_metrics_prometheus_reflects_recorded_requests() -> None:
    client.get("/health")
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# HELP vlm_api_requests_total" in body
    assert 'vlm_api_requests_total{endpoint="/health"}' in body
    assert "vlm_jetson_tegrastats_available 0" in body  # no tegrastats binary in this sandbox


def test_metrics_json_reflects_recorded_requests() -> None:
    client.get("/health")

    response = client.get("/metrics.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["endpoints"]["/health"]["count"] >= 1
    assert payload["gpu_memory"] is None  # no CUDA in this sandbox
    assert payload["tegrastats"]["available"] is False  # no tegrastats binary in this sandbox


def test_embed_image_reaches_torch_boundary() -> None:
    files = {"file": ("test.png", _make_png_bytes(), "image/png")}
    with pytest.raises(ModuleNotFoundError, match="torch"):
        client.post("/v1/embed/image", files=files)


def test_embed_image_rejects_empty_upload() -> None:
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/v1/embed/image", files=files)
    assert response.status_code == 400
    assert "Empty" in response.json()["detail"]


def test_embed_image_rejects_corrupt_bytes() -> None:
    files = {"file": ("bad.png", b"not an image", "image/png")}
    response = client.post("/v1/embed/image", files=files)
    assert response.status_code == 400
    assert "Not a valid image" in response.json()["detail"]


def test_embed_image_rejects_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_app_module, "MAX_IMAGE_BYTES", 10)
    files = {"file": ("test.png", _make_png_bytes(), "image/png")}
    response = client.post("/v1/embed/image", files=files)
    assert response.status_code == 413


def test_embed_image_rejects_wrong_content_type() -> None:
    files = {"file": ("test.png", _make_png_bytes(), "text/plain")}
    response = client.post("/v1/embed/image", files=files)
    assert response.status_code == 415


def test_embed_video_reaches_torch_boundary(tmp_path: Path) -> None:
    files = {"file": ("clip.mp4", _make_mp4_bytes(tmp_path), "video/mp4")}
    with pytest.raises(ModuleNotFoundError, match="torch"):
        client.post("/v1/embed/video", files=files)


def test_embed_video_rejects_empty_upload() -> None:
    files = {"file": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/v1/embed/video", files=files)
    assert response.status_code == 400


def test_embed_video_rejects_corrupt_bytes() -> None:
    files = {"file": ("bad.mp4", b"not a real video file", "video/mp4")}
    response = client.post("/v1/embed/video", files=files)
    assert response.status_code == 400


def test_embed_video_rejects_wrong_content_type(tmp_path: Path) -> None:
    files = {"file": ("clip.mp4", _make_mp4_bytes(tmp_path), "text/plain")}
    response = client.post("/v1/embed/video", files=files)
    assert response.status_code == 415


def test_embed_video_rejects_out_of_range_max_frames(tmp_path: Path) -> None:
    files = {"file": ("clip.mp4", _make_mp4_bytes(tmp_path), "video/mp4")}
    response = client.post("/v1/embed/video?max_frames=0", files=files)
    assert response.status_code == 422


def test_embed_rtsp_rejects_bad_scheme() -> None:
    response = client.post("/v1/embed/rtsp", json={"url": "http://example.com/stream"})
    assert response.status_code == 422


def test_embed_rtsp_rejects_out_of_range_max_frames() -> None:
    response = client.post(
        "/v1/embed/rtsp", json={"url": "rtsp://example.invalid/stream", "max_frames": 0}
    )
    assert response.status_code == 422


def test_embed_rtsp_returns_503_when_stream_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(source: str) -> None:
        raise VideoSourceError(f"Could not open video source: {source}")

    monkeypatch.setattr(api_app_module, "open_video_capture", _raise)
    response = client.post("/v1/embed/rtsp", json={"url": "rtsp://example.invalid/stream"})
    assert response.status_code == 503


def test_embed_rtsp_reaches_torch_boundary_once_stream_opens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Stand in for a live RTSP connection with a real local file -- both go
    # through the exact same cv2.VideoCapture API (see src/data/video.py),
    # so only the network hop is faked; frame sampling is fully real.
    video_path = tmp_path / "for_rtsp.mp4"
    video_path.write_bytes(_make_mp4_bytes(tmp_path))

    import cv2

    def _fake_open(source: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(video_path))

    monkeypatch.setattr(api_app_module, "open_video_capture", _fake_open)

    with pytest.raises(ModuleNotFoundError, match="torch"):
        client.post("/v1/embed/rtsp", json={"url": "rtsp://example.invalid/stream"})


def test_embed_text_reaches_torch_boundary() -> None:
    with pytest.raises(ModuleNotFoundError, match="torch"):
        client.post("/v1/embed/text", json={"texts": ["a scratch on metal"]})


def test_embed_text_rejects_empty_texts_list() -> None:
    response = client.post("/v1/embed/text", json={"texts": []})
    assert response.status_code == 422
