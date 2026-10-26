"""IoU-based multi-object tracking (SORT-style, tracking-by-detection) for video streams.

Week 13's core new piece: give each frame's detections a STABLE identity
across frames, so a defect pipeline can tell "the same crack, seen again"
from "a different crack that happens to be nearby" -- the prerequisite
`smoothing.py` needs to smooth a verdict PER OBJECT, not per frame.

Deliberately decoupled from week 03's
`ros2_ws/src/vision_bridge/vision_bridge/detection.py` -- same
label/score/bbox shape, NOT the same class. That module lives at a
nested, non-standard path reached only via a `sys.path` addition (see
its own docstring); a generic tracking library has no business depending
on one ROS2 colcon package's file layout. Converting a real
`vision_bridge.detection.Detection` into THIS module's `Detection` is a
pure attribute-copy (duck typing, no import needed) -- see
`adapters.py::from_vision_bridge_detection`.

No Kalman filter / velocity prediction (real SORT's other half, and
`motpy`/`filterpy`-class libraries' whole reason to exist) -- matches
each frame's detections against the LAST KNOWN box of each active track
by IoU directly. Deliberate simplification for a first pass (this
course's running principle: deterministic-first, simple code before
reaching for a heavier technique) -- good enough when frame-to-frame
object motion is small relative to box size (a product on a conveyor,
this course's actual scenario), degrades on fast motion between sampled
frames (see `frame_sampling.py`'s stride tradeoff for the other half of
that problem).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box, image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def area(self) -> float:
        """Box area, `0.0` for a degenerate (zero/negative width or height) box."""
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


@dataclass(frozen=True)
class Detection:
    """One per-frame detection -- same shape as week 03's `vision_bridge.detection.Detection`."""

    label: str
    score: float
    bbox: BBox


@dataclass(frozen=True)
class TrackedDetection:
    """One detection, now carrying a stable identity across frames.

    Attributes:
        track_id: Stable across frames for as long as the underlying
            object keeps getting matched (see `IoUTracker`'s docstring
            for exactly how long that survives a miss).
        label: This frame's raw label for the matched detection (NOT
            smoothed -- see `smoothing.py` for that).
        score: This frame's raw detector confidence.
        bbox: This frame's box.
        hits: Total frames this track has been matched to a detection,
            including this one.
        age: Frames since this track was first created (matched or not).
    """

    track_id: int
    label: str
    score: float
    bbox: BBox
    hits: int
    age: int


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes.

    Returns:
        A value in `[0, 1]`; `0.0` if the boxes don't overlap at all, or
        either is degenerate.
    """
    x_min = max(a.x_min, b.x_min)
    y_min = max(a.y_min, b.y_min)
    x_max = min(a.x_max, b.x_max)
    y_max = min(a.y_max, b.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = a.area() + b.area() - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


@dataclass
class _Track:
    """Internal, mutable -- `IoUTracker`'s own bookkeeping, never returned to callers."""

    track_id: int
    label: str
    score: float
    bbox: BBox
    hits: int = 1
    age: int = 1
    time_since_update: int = 0


class IoUTracker:
    """Hungarian-algorithm IoU-matching tracker -- spatial only, label-agnostic (see `_match`)."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 1) -> None:
        """
        Args:
            iou_threshold: Minimum IoU for a detection to be matched to an
                existing track instead of starting a new one.
            max_age: Frames a track survives with no matching detection
                before being dropped -- absorbs a brief occlusion/missed
                detection without losing the object's identity; too high
                and a track can outlive the object actually leaving frame,
                silently matching an unrelated later detection.
            min_hits: Minimum total matches before a track is reported by
                `update()` -- suppresses a one-off spurious detection from
                ever getting a stable id in the first place.

        Raises:
            ValueError: If `iou_threshold` isn't in `[0, 1]`, or
                `max_age`/`min_hits` is less than 1.
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold!r}.")
        if max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {max_age!r}.")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits!r}.")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    @property
    def active_track_count(self) -> int:
        """How many tracks (matched or not, confirmed or not) this tracker is currently holding.

        A real indicator of this tracker's own memory footprint -- O(active
        tracks), not O(frames) -- relevant on a Jetson running this
        alongside the detector/CLIP itself (see `tracking/README.md`'s
        Jetson-reality note).
        """
        return len(self._tracks)

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        """Advance the tracker by one frame.

        Args:
            detections: This frame's detections (unordered).

        Returns:
            One `TrackedDetection` per currently-active, confirmed
            (`hits >= min_hits`) track matched THIS frame. A track that
            went unmatched this frame (but hasn't aged out past
            `max_age`) is kept internally for future frames, not
            returned -- a caller never sees a "detection" for a frame
            where nothing was actually detected.
        """
        matched_track_indices, matched_detection_indices = self._match(detections)

        for track_index, detection_index in zip(
            matched_track_indices, matched_detection_indices, strict=True
        ):
            detection = detections[detection_index]
            track = self._tracks[track_index]
            track.bbox = detection.bbox
            track.score = detection.score
            track.label = detection.label
            track.hits += 1
            track.age += 1
            track.time_since_update = 0

        # Age only tracks that existed BEFORE this frame's new ones are
        # appended below -- a track just created from an unmatched
        # detection starts at time_since_update=0 and must stay there for
        # this frame; iterating over self._tracks AFTER appending (the
        # original, buggy version of this method) would immediately
        # increment every brand-new track's time_since_update to 1,
        # making it indistinguishable from a track that was just missed,
        # and silently dropping min_hits=1 tracks from update()'s return
        # value on the very frame they were created.
        num_existing_tracks = len(self._tracks)
        matched_track_set = set(matched_track_indices)
        for track_index in range(num_existing_tracks):
            if track_index in matched_track_set:
                continue
            track = self._tracks[track_index]
            track.age += 1
            track.time_since_update += 1

        matched_detection_set = set(matched_detection_indices)
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_set:
                continue
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    label=detection.label,
                    score=detection.score,
                    bbox=detection.bbox,
                )
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        return [
            TrackedDetection(
                track.track_id, track.label, track.score, track.bbox, track.hits, track.age
            )
            for track in self._tracks
            if track.time_since_update == 0 and track.hits >= self.min_hits
        ]

    def _match(self, detections: list[Detection]) -> tuple[list[int], list[int]]:
        """Hungarian-algorithm IoU matching between current tracks and this frame's detections.

        Spatial (IoU) only -- NOT gated by label equality. That's
        deliberate, not an oversight: a label flicker on the SAME
        physical object (the exact noise `smoothing.py` exists to clean
        up) must not also break tracking identity, or there'd be nothing
        for the smoother to smooth -- the track would just fragment into
        a new id every time the raw label flips. The cost is a real one
        too: two spatially-overlapping objects of genuinely DIFFERENT
        types could in principle merge into one track. Accepted for this
        first pass (course principle: simple code, the tradeoff
        documented rather than hidden) -- a production tracker would
        likely blend label similarity into the cost matrix instead of
        using it as a hard gate.

        Returns:
            Parallel lists of `(track_index, detection_index)` pairs that
            matched (both empty if there's nothing to match against, or
            no pair cleared `iou_threshold`).
        """
        if not self._tracks or not detections:
            return [], []

        import numpy as np
        from scipy.optimize import linear_sum_assignment

        iou_matrix = np.zeros((len(self._tracks), len(detections)))
        for i, track in enumerate(self._tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = iou(track.bbox, detection.bbox)

        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)

        matched_tracks = []
        matched_detections = []
        for t_idx, d_idx in zip(track_indices, detection_indices, strict=True):
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                matched_tracks.append(int(t_idx))
                matched_detections.append(int(d_idx))
        return matched_tracks, matched_detections
