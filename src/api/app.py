"""FastAPI application exposing CLIP image/video/RTSP/text embedding endpoints.

Every request is logged as one structured JSON line (see
`src.api.logging_config`) with a `request_id`, method, path, status code
and duration -- the shape a container log-collector (or `journalctl -o
json` on the Jetson) can parse without a custom pipeline.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.dependencies import get_inference, get_metrics_collector, get_tegrastats_sampler
from src.api.logging_config import configure_logging
from src.api.metrics import gpu_memory_mb, render_prometheus
from src.api.schemas import (
    EmbedResponse,
    EndpointMetrics,
    FrameEmbedResponse,
    GpuMemoryMetrics,
    MetricsResponse,
    RTSPEmbedRequest,
    TegrastatsMetrics,
    TextEmbedRequest,
)
from src.data.video import VideoSourceError, open_video_capture, sample_frames
from src.models.memory import process_resident_memory_mb

configure_logging(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO))
logger = logging.getLogger("vlm_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the background `tegrastats` sampler (week 08 day 4) around the process lifetime.

    A no-op start/stop when `tegrastats` isn't on this host (every
    environment except a real Jetson) -- see `TegrastatsSampler`.
    """
    get_tegrastats_sampler().start()
    yield
    get_tegrastats_sampler().stop()


app = FastAPI(title="VLM Edge Inference API", version="0.1.0", lifespan=lifespan)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024


def _validate_upload_size(raw_bytes: bytes, max_bytes: int, kind: str) -> None:
    """Reject empty or too-large uploads with a clear HTTP error.

    Args:
        raw_bytes: The fully-read upload body.
        max_bytes: Size ceiling for this upload kind.
        kind: Human-readable label for the error message (e.g. `"image"`).

    Raises:
        HTTPException: 400 if empty, 413 if over `max_bytes`.
    """
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail=f"Empty {kind} upload.")
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{kind} upload too large: {len(raw_bytes)} bytes (max {max_bytes}).",
        )


def _validate_content_type(content_type: str | None, expected_prefix: str, kind: str) -> None:
    """Reject an explicitly-wrong content-type; a missing one is allowed through.

    Some clients (`curl -F`, some camera SDKs) don't set a reliable
    content-type at all -- only reject a content-type that's actually
    present and clearly wrong, and let the real parser (`PIL`/`cv2`) be
    the authoritative check on the bytes themselves.

    Args:
        content_type: The upload's declared content-type, if any.
        expected_prefix: Required prefix, e.g. `"image/"`.
        kind: Human-readable label for the error message.

    Raises:
        HTTPException: 415 if `content_type` is set and doesn't match.
    """
    if content_type is not None and not content_type.startswith(expected_prefix):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported {kind} content-type: {content_type!r} "
                f"(expected {expected_prefix}*)."
            ),
        )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request as one structured JSON line, with a request-id.

    Args:
        request: The incoming request.
        call_next: The next handler in the middleware chain.

    Returns:
        The response, unmodified except for an added `X-Request-ID` header.
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        # request.url.path is the raw requested path, not a matched route
        # template -- an attacker probing many distinct 404 paths could
        # grow per-endpoint cardinality unboundedly. Accepted for this
        # single-service, not-internet-facing deployment (see week 08 day 4
        # in course/DECISIONS.md); a real multi-tenant/public API should
        # key on `request.scope["route"].path` (the template) instead.
        get_metrics_collector().record(request.url.path, duration_ms, status_code)
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        `{"status": "ok"}` when the process is up.
    """
    return {"status": "ok"}


@app.get("/metrics")
def metrics_prometheus() -> Response:
    """Prometheus text-exposition-format metrics (week 08 day 4).

    Request latency (p50/p95/mean) and throughput per endpoint, this
    process's RSS and CUDA memory (when available), and `tegrastats`
    temperatures -- see `src.api.metrics` for what each metric means and
    the simplifications involved.

    Returns:
        `text/plain; version=0.0.4` body, ready for a Prometheus scrape config.
    """
    body = render_prometheus(
        get_metrics_collector().snapshot(),
        get_tegrastats_sampler().snapshot(),
        gpu_memory_mb(),
        process_resident_memory_mb(),
    )
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/metrics.json", response_model=MetricsResponse)
def metrics_json() -> MetricsResponse:
    """The same metrics as `GET /metrics`, as JSON (week 08 day 4).

    Returns:
        A `MetricsResponse` -- easier to consume from a script or a
        dashboard that isn't Prometheus itself.
    """
    snapshot = get_metrics_collector().snapshot()
    tegrastats_snapshot = get_tegrastats_sampler().snapshot()
    gpu_memory = gpu_memory_mb()
    return MetricsResponse(
        uptime_seconds=snapshot.uptime_seconds,
        process_rss_mb=process_resident_memory_mb(),
        endpoints={
            endpoint: EndpointMetrics(
                count=stats.count,
                mean_ms=stats.mean_ms,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                throughput_rps=stats.throughput_rps,
            )
            for endpoint, stats in snapshot.endpoints.items()
        },
        status_counts=snapshot.status_counts,
        gpu_memory=GpuMemoryMetrics(**gpu_memory) if gpu_memory is not None else None,
        tegrastats=TegrastatsMetrics(
            available=tegrastats_snapshot.available,
            latest_temps_c=tegrastats_snapshot.latest_temps_c,
            max_temps_c=tegrastats_snapshot.max_temps_c,
            thermal_warning=tegrastats_snapshot.thermal_warning,
            throttle_temp_c=tegrastats_snapshot.throttle_temp_c,
        ),
    )


@app.post("/v1/embed/image", response_model=EmbedResponse)
async def embed_image(file: UploadFile = File(...)) -> EmbedResponse:
    """Embed a single uploaded image.

    Args:
        file: Multipart-uploaded image file.

    Returns:
        The image's L2-normalized embedding vector.

    Raises:
        HTTPException: 400 empty/corrupt upload, 413 too large, 415 wrong
            content-type.
    """
    raw_bytes = await file.read()
    _validate_upload_size(raw_bytes, MAX_IMAGE_BYTES, "image")
    _validate_content_type(file.content_type, "image/", "image")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        logger.warning("invalid image upload", extra={"upload_filename": file.filename})
        raise HTTPException(status_code=400, detail=f"Not a valid image: {exc}") from exc

    result = get_inference().encode_image([image])
    vectors = result.vectors.tolist()
    return EmbedResponse(vectors=vectors, model_name=result.model_name, dim=len(vectors[0]))


@app.post("/v1/embed/video", response_model=FrameEmbedResponse)
async def embed_video(
    file: UploadFile = File(...),
    max_frames: int = Query(8, ge=1, le=64, description="Max frames to sample from the video."),
    frame_stride: int = Query(1, ge=1, le=300, description="Keep 1 out of every N frames read."),
) -> FrameEmbedResponse:
    """Embed frames sampled from an uploaded video file.

    Args:
        file: Multipart-uploaded video file.
        max_frames: Max number of frames to sample.
        frame_stride: Keep 1 out of every N frames read.

    Returns:
        One L2-normalized embedding vector per sampled frame.

    Raises:
        HTTPException: 400 empty/unreadable upload, 413 too large, 415
            wrong content-type.
    """
    raw_bytes = await file.read()
    _validate_upload_size(raw_bytes, MAX_VIDEO_BYTES, "video")
    _validate_content_type(file.content_type, "video/", "video")

    suffix = Path(file.filename).suffix if file.filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4") as tmp_file:
        tmp_file.write(raw_bytes)
        tmp_file.flush()
        try:
            capture = open_video_capture(tmp_file.name)
            frames = sample_frames(capture, max_frames=max_frames, frame_stride=frame_stride)
        except VideoSourceError as exc:
            logger.warning("unreadable video upload", extra={"upload_filename": file.filename})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not frames:
        raise HTTPException(
            status_code=400, detail="No frames could be read from the uploaded video."
        )

    result = get_inference().encode_image(frames)
    vectors = result.vectors.tolist()
    return FrameEmbedResponse(
        vectors=vectors, model_name=result.model_name, dim=len(vectors[0]), num_frames=len(frames)
    )


@app.post("/v1/embed/rtsp", response_model=FrameEmbedResponse)
def embed_rtsp(request: RTSPEmbedRequest) -> FrameEmbedResponse:
    """Embed frames sampled live from an RTSP stream.

    Args:
        request: Stream URL plus `max_frames`/`frame_stride` sampling
            parameters (`url`'s scheme is validated by `RTSPEmbedRequest`).

    Returns:
        One L2-normalized embedding vector per sampled frame.

    Raises:
        HTTPException: 503 if the stream can't be reached/read.
    """
    try:
        capture = open_video_capture(request.url)
        frames = sample_frames(
            capture, max_frames=request.max_frames, frame_stride=request.frame_stride
        )
    except VideoSourceError as exc:
        logger.warning("unreachable RTSP stream", extra={"url": request.url})
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not frames:
        raise HTTPException(status_code=503, detail="No frames could be read from the RTSP stream.")

    result = get_inference().encode_image(frames)
    vectors = result.vectors.tolist()
    return FrameEmbedResponse(
        vectors=vectors, model_name=result.model_name, dim=len(vectors[0]), num_frames=len(frames)
    )


@app.post("/v1/embed/text", response_model=EmbedResponse)
def embed_text(request: TextEmbedRequest) -> EmbedResponse:
    """Embed a batch of text prompts.

    Args:
        request: Text prompts to embed.

    Returns:
        One L2-normalized embedding vector per input prompt.
    """
    result = get_inference().encode_text(request.texts)
    vectors = result.vectors.tolist()
    return EmbedResponse(vectors=vectors, model_name=result.model_name, dim=len(vectors[0]))
