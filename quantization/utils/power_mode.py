"""`nvpmodel` power-mode switching + `tegrastats` sampling for `06_trt_inference.py`.

Duplicated from `scripts/jetson_benchmark.py`'s `TegrastatsMonitor`/
`set_nvpmodel` (week 01) rather than imported -- `quantization/` is a
separate top-level package with no cross-import from root `scripts/`
(only `src/` is shared across top-level packages, see `../README.md`),
and `jetson_benchmark.py`'s helpers aren't importable as a plain module
anyway (root `scripts/` isn't a package). Reuses
`src.evaluation.jetson.parse_nvpmodel_query` -- the hardware-free parsing
logic `jetson_benchmark.py` itself depends on rather than reimplements.

`06_trt_inference.py`'s own docstring covers *why* the day's ask ("run at
a fixed power mode: 15W as standard, MAXN Super for the upper bound")
lives here rather than being left to whoever runs the script manually.
"""

from __future__ import annotations

import subprocess
import time

from src.evaluation.jetson import parse_nvpmodel_query


def set_nvpmodel(mode_id: int, settle_seconds: float) -> None:
    """Switch the device power mode via `sudo nvpmodel -m <mode_id>` and let clocks settle.

    Args:
        mode_id: Target `nvpmodel` mode (see `src.evaluation.jetson.NVPMODEL_MODES`).
        settle_seconds: Time to sleep after switching, before measuring.

    Raises:
        subprocess.CalledProcessError: If `nvpmodel` exits non-zero.
    """
    subprocess.run(["sudo", "nvpmodel", "-m", str(mode_id)], check=True)
    time.sleep(settle_seconds)


def query_power_mode() -> dict[str, str | int | None] | None:
    """Run `nvpmodel -q` and return the parsed power-mode name/id, if available.

    Returns:
        `{"mode_name", "mode_id"}` (see `parse_nvpmodel_query`), or `None`
        if `nvpmodel` isn't present/runnable on this machine (i.e. this
        isn't running on a Jetson).
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvpmodel_query(result.stdout)


class TegrastatsMonitor:
    """Runs `tegrastats` in the background for the duration of a `with` block."""

    def __init__(self, interval_ms: int) -> None:
        """Args:
            interval_ms: Sampling interval passed to `tegrastats --interval`.
        """
        self._interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None
        self.lines: list[str] = []

    def __enter__(self) -> TegrastatsMonitor:
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
