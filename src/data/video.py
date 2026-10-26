"""Video/RTSP frame sampling via OpenCV (week 08, day 2).

`cv2.VideoCapture` opens a local file and an `rtsp://`/`rtsps://` stream
through the exact same API, so `sample_frames` is source-agnostic and
sequential-only (no seeking) -- it works identically for a finite,
seekable file and an unbounded, live stream. `cv2` is imported lazily so
importing this module never requires `opencv-python-headless` to be
installed, matching how `torch`/`transformers` are handled in
`src.models.clip`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class VideoSourceError(RuntimeError):
    """Raised when a video file or RTSP stream can't be opened or read."""


def open_video_capture(source: str | Path) -> Any:
    """Open a video file or RTSP/RTSPS URL via OpenCV.

    Args:
        source: Local file path or an `rtsp://`/`rtsps://` stream URL.

    Returns:
        An opened `cv2.VideoCapture`. Caller owns it -- `sample_frames`
        releases it, but a caller that never calls `sample_frames` must
        release it itself.

    Raises:
        VideoSourceError: If the source can't be opened -- missing file,
            unreachable stream, unsupported codec, refused connection.
    """
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoSourceError(f"Could not open video source: {source}")
    return capture


def sample_frames(capture: Any, max_frames: int, frame_stride: int = 1) -> list[Image.Image]:
    """Read up to `max_frames` frames from `capture`, keeping 1 of every `frame_stride`.

    Sequential-only (no seeking), so this works identically for a seekable
    video file and a live RTSP stream that can only be read forward.
    Always releases `capture`, even if reading raises.

    Args:
        capture: An opened `cv2.VideoCapture` (see `open_video_capture`).
        max_frames: Stop once this many frames have been collected.
        frame_stride: Keep 1 out of every `frame_stride` frames read (e.g.
            `frame_stride=5` on a 30fps source samples roughly 6fps).

    Returns:
        Up to `max_frames` RGB `PIL.Image`s, in stream order. Fewer than
        `max_frames` if the source ends first (normal for a short file).

    Raises:
        ValueError: If `max_frames < 1` or `frame_stride < 1`.
    """
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}.")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}.")

    import cv2

    frames: list[Image.Image] = []
    read_index = 0
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if read_index % frame_stride == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            read_index += 1
    finally:
        capture.release()
    return frames
