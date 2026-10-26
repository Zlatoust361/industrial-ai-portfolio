"""nvpmodel power-mode check for the Jetson Orin Nano.

Standalone by design: a colcon build only sees files inside this ROS2
workspace, not the rest of the vlm-course monorepo, so this intentionally
duplicates the small `nvpmodel -q` parsing logic from
`src/evaluation/jetson.py::parse_nvpmodel_query` rather than importing
across the workspace boundary -- this package must build and run on its
own.
"""

from __future__ import annotations

import re
import subprocess

SUFFICIENT_POWER_MODES = frozenset({"15W", "MAXN_SUPER"})
"""Power modes expected to sustain stable detection FPS under continuous load.

`7W` is intentionally excluded: it's the Jetson Orin Nano's lowest power
preset and caps GPU/DLA clocks well below `15W`/`MAXN_SUPER`, which shows
up as dropped or delayed frames once a detector is running continuously
against a live camera stream rather than sitting idle.
"""

_NVPMODEL_QUERY_NAME_PATTERN = re.compile(r"NV Power Mode:\s*(\S+)")


def parse_nvpmodel_mode_name(output: str) -> str | None:
    """Extract the power-mode name from `nvpmodel -q` stdout.

    Args:
        output: Raw stdout of `nvpmodel -q`, e.g. `"NV Power Mode: MAXN_SUPER\\n0\\n"`.

    Returns:
        The mode name (e.g. `"MAXN_SUPER"`, `"15W"`, `"7W"`), or `None` if
        the expected pattern isn't found in `output`.
    """
    match = _NVPMODEL_QUERY_NAME_PATTERN.search(output)
    return match.group(1) if match else None


def is_power_mode_sufficient(mode_name: str) -> bool:
    """Whether `mode_name` is expected to sustain stable detection FPS.

    Args:
        mode_name: A power-mode name, e.g. from `parse_nvpmodel_mode_name`.

    Returns:
        `True` for `"15W"`/`"MAXN_SUPER"`, `False` otherwise (notably `"7W"`).
    """
    return mode_name in SUFFICIENT_POWER_MODES


def query_current_power_mode() -> str | None:
    """Run `nvpmodel -q` and return the parsed mode name, if available.

    Returns:
        The current power-mode name, or `None` if `nvpmodel` isn't
        present/runnable on this machine (i.e. this isn't running on a
        Jetson) or its output didn't match the expected format.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvpmodel_mode_name(result.stdout)
