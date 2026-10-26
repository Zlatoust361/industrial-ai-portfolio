"""Jetson-device detection for the quantization pipeline (week 09).

Self-contained, not imported from `pipeline/utils/environment.py` -- the
same "separate top-level directory, no cross-import" stance
`pipeline/`/`dataset_pipeline/` each already take, applied to this exact
`/etc/nv_tegra_release` check (duplicated a fourth time now, after
`pipeline/utils/environment.py` (week 04), `scripts/export_and_compare_jetson.py`
(week 07), and `ros2_ws/scripts/convert_owlvit_tensorrt.py` (week 03)).

Every script from `04_quantize_model.py` onward calls `require_jetson()`
before doing any TensorRT-specific work: a `.engine` file (or an INT4
calibration cache tuned against a specific GPU's kernel selection) built
off-device either won't load on a real Orin Nano at all (different GPU
architecture/TensorRT version -- same hard constraint week 03's
`convert_owlvit_tensorrt.py` and week 07's `export_and_compare_jetson.py`
document) or would silently calibrate against the wrong hardware,
producing an artifact that looks fine and isn't.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

DEFAULT_NV_TEGRA_RELEASE_PATH = Path("/etc/nv_tegra_release")
"""JetPack's marker file -- present only on a real Jetson device."""


def is_jetson(marker_path: Path = DEFAULT_NV_TEGRA_RELEASE_PATH) -> bool:
    """Whether this machine is a Jetson device, per JetPack's marker file.

    Args:
        marker_path: Path to check. Defaults to `/etc/nv_tegra_release`,
            present on every JetPack image and absent everywhere else.

    Returns:
        `True` if `marker_path` exists.
    """
    return marker_path.is_file()


def require_jetson(
    marker_path: Path,
    script_name: str,
    force: bool,
    stream: TextIO | None = None,
) -> None:
    """Exit the process with a clear error unless this looks like a Jetson (or `force`).

    Args:
        marker_path: Path checked for Jetson detection (see `is_jetson`).
        script_name: Calling script's filename, used in the error message.
        force: If `True`, skip the check entirely -- for exercising the
            script's logic on a dev machine; the resulting artifact won't
            be deployable/meaningful (see the printed message).
        stream: Where to print the error; `None` (the default) resolves
            `sys.stderr` at call time, not import time -- binding it as a
            plain default parameter value would capture the *original*
            stderr object before pytest's `capsys` (or any other stream
            swap) ever runs, the same gotcha `pipeline/README.md` documents
            for `build_environment_report`'s `marker_path` default.

    Side effects:
        Calls `sys.exit(1)` (without raising) if this isn't a Jetson and
        `force` is `False`. Callers don't need their own early-return --
        this either returns normally or the process ends here.
    """
    if force or is_jetson(marker_path):
        return
    print(
        f"Error: '{marker_path}' not found -- {script_name} must run ON a Jetson Orin Nano, "
        "not a dev machine. TensorRT engines/calibration caches built elsewhere either won't "
        "load at all (different GPU architecture/TensorRT version -- same constraint as week "
        "03's ros2_ws/scripts/convert_owlvit_tensorrt.py and week 07's "
        "scripts/export_and_compare_jetson.py) or would calibrate against the wrong hardware's "
        "kernel selection. Run this on the device, or pass --force to exercise the script's "
        "logic anyway (the resulting artifact won't be deployable/meaningful).",
        file=stream if stream is not None else sys.stderr,
    )
    sys.exit(1)
