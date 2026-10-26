"""System (tegrastats) and process (psutil) memory monitoring.

Two distinct metrics, tracked separately since they answer different
questions: `tegrastats` reports whole-device RAM usage (useful for "does
this fit on the 8 GB Orin Nano alongside everything else"), `psutil`
reports just this process's RSS (useful for "how much does this model
itself cost"). Duplicates tegrastats-line parsing from
`vision_bridge.power_mode` / week 01's `src/evaluation/jetson.py` rather
than importing either -- this script is meant to stand on its own (same
reasoning as `vision_bridge.power_mode`'s own docstring).
"""

from __future__ import annotations

import re
import subprocess

_TEGRASTATS_RAM_PATTERN = re.compile(r"RAM (\d+)/(\d+)MB")


def parse_tegrastats_ram_mb(line: str) -> int | None:
    """Extract used system RAM (MB) from one `tegrastats` output line.

    Args:
        line: A raw `tegrastats` stdout line, e.g. `"RAM 3241/7620MB (lfb ...) ..."`.

    Returns:
        Used RAM in MB, or `None` if the line doesn't match the expected pattern.
    """
    match = _TEGRASTATS_RAM_PATTERN.search(line)
    return int(match.group(1)) if match else None


class TegrastatsRamMonitor:
    """Runs `tegrastats` in the background for the duration of a `with` block."""

    def __init__(self, interval_ms: int = 500) -> None:
        """Args:
            interval_ms: Sampling interval passed to `tegrastats --interval`.
        """
        self._interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None
        self.lines: list[str] = []

    def __enter__(self) -> TegrastatsRamMonitor:
        self._process = subprocess.Popen(
            ["tegrastats", "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            text=True,
        )
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            stdout, _ = self._process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout, _ = self._process.communicate()
        self.lines = stdout.splitlines() if stdout else []

    def peak_ram_mb(self) -> int | None:
        """Peak used system RAM observed across all captured lines.

        Returns:
            The maximum value `parse_tegrastats_ram_mb` found across
            `self.lines`, or `None` if `tegrastats` wasn't available/no
            line matched (e.g. not running on a Jetson).
        """
        values = [v for line in self.lines if (v := parse_tegrastats_ram_mb(line)) is not None]
        return max(values) if values else None


class PeakRssTracker:
    """Tracks this process's peak resident set size (RSS) across repeated samples.

    Call `.sample()` periodically during the work being measured (e.g.
    after each processed image) -- there's no background thread here, so
    it only sees what it's told to look at.
    """

    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process()
        self._peak_mb = 0.0

    def sample(self) -> None:
        """Record the current RSS if it's higher than the tracked peak."""
        rss_mb = self._process.memory_info().rss / (1024**2)
        self._peak_mb = max(self._peak_mb, rss_mb)

    @property
    def peak_mb(self) -> float:
        """Peak RSS observed so far, in MiB (`0.0` if `.sample()` was never called)."""
        return self._peak_mb
