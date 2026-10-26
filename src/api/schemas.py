"""Pydantic request/response models for the embedding API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

RTSP_URL_SCHEMES = ("rtsp://", "rtsps://")


class TextEmbedRequest(BaseModel):
    """Request body for `POST /v1/embed/text`."""

    texts: list[str] = Field(..., min_length=1, description="Text prompts to embed.")


class EmbedResponse(BaseModel):
    """Response body shared by the image and text embedding endpoints."""

    vectors: list[list[float]] = Field(..., description="L2-normalized embedding vectors.")
    model_name: str = Field(..., description="Identifier of the model used.")
    dim: int = Field(..., description="Embedding dimensionality.")


class FrameEmbedResponse(BaseModel):
    """Response body for the video and RTSP endpoints: one vector per sampled frame."""

    vectors: list[list[float]] = Field(..., description="L2-normalized embedding vectors.")
    model_name: str = Field(..., description="Identifier of the model used.")
    dim: int = Field(..., description="Embedding dimensionality.")
    num_frames: int = Field(..., description="Number of frames actually sampled and embedded.")


class RTSPEmbedRequest(BaseModel):
    """Request body for `POST /v1/embed/rtsp`."""

    url: str = Field(..., description="rtsp:// or rtsps:// stream URL.")
    max_frames: int = Field(8, ge=1, le=64, description="Max frames to sample from the stream.")
    frame_stride: int = Field(
        1, ge=1, le=300, description="Keep 1 out of every N frames read from the stream."
    )

    @field_validator("url")
    @classmethod
    def _validate_rtsp_scheme(cls, value: str) -> str:
        if not value.startswith(RTSP_URL_SCHEMES):
            raise ValueError(f"url must start with one of {RTSP_URL_SCHEMES}, got: {value!r}")
        return value


class EndpointMetrics(BaseModel):
    """Latency/throughput stats for one endpoint, mirrors `src.api.metrics.EndpointStats`."""

    count: int = Field(..., description="Total requests since process start (unwindowed).")
    mean_ms: float = Field(
        ..., description="Mean request latency, over the last window_size requests."
    )
    p50_ms: float = Field(
        ..., description="Median request latency, over the last window_size requests."
    )
    p95_ms: float = Field(
        ..., description="95th-percentile request latency, over the last window_size requests."
    )
    throughput_rps: float = Field(
        ..., description="count / uptime_seconds, averaged since process start."
    )


class GpuMemoryMetrics(BaseModel):
    """`torch.cuda` memory usage, mirrors `src.api.metrics.gpu_memory_mb`'s return value."""

    allocated_mb: float
    reserved_mb: float


class TegrastatsMetrics(BaseModel):
    """Temperature readings, mirrors `src.api.metrics.TegrastatsSnapshot`."""

    available: bool = Field(
        ..., description="Whether the tegrastats binary was found on this host."
    )
    latest_temps_c: dict[str, float] = Field(
        ..., description="Most recent reading per sensor, Celsius."
    )
    max_temps_c: dict[str, float] = Field(
        ..., description="Peak reading per sensor since process start, Celsius."
    )
    thermal_warning: bool = Field(
        ...,
        description="True if any sensor is at/above throttle_temp_c (a configurable proxy).",
    )
    throttle_temp_c: float = Field(
        ..., description="The threshold thermal_warning was computed against."
    )


class MetricsResponse(BaseModel):
    """Response body for `GET /metrics.json`.

    The same data `GET /metrics` renders as Prometheus text.
    """

    uptime_seconds: float = Field(
        ..., description="Seconds since the MetricsCollector was created."
    )
    process_rss_mb: float = Field(..., description="This process's resident set size, in MiB.")
    endpoints: dict[str, EndpointMetrics] = Field(
        ..., description="Per-endpoint stats, plus '_all' for the aggregate across every endpoint."
    )
    status_counts: dict[str, int] = Field(
        ..., description="'{endpoint}:{status_code}' -> count, since process start."
    )
    gpu_memory: GpuMemoryMetrics | None = Field(
        ..., description="None if CUDA isn't available (e.g. a CPU dev deployment)."
    )
    tegrastats: TegrastatsMetrics
