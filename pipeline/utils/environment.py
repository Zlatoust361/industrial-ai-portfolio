"""Runtime-environment detection: Jetson vs dev machine, JetPack/L4T version,
available memory, current `nvpmodel` power mode.

Used by `scripts/01_environment_check.py` to write `outputs/env_report.json`.
Kept dependency-light (`psutil` only, already a root `requirements.txt`
dependency since week 01) and subprocess/filesystem access is isolated in
small functions that all accept their inputs as plain arguments (a path, a
string, ...), so the actual parsing logic is unit-testable without a real
Jetson, mocked or otherwise -- see `pipeline/tests/test_environment.py`.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NV_TEGRA_RELEASE_PATH = Path("/etc/nv_tegra_release")
"""JetPack's marker file -- present only on a real Jetson device."""

_NV_TEGRA_RELEASE_PATTERN = re.compile(r"R(\d+) \(release\), REVISION:\s*([\d.]+)")

# L4T major version -> JetPack *generation* (not the precise minor version --
# see `EnvironmentReport.jetpack_generation`'s docstring for why that's a
# deliberately coarser claim than "JetPack version").
_L4T_MAJOR_TO_JETPACK_GENERATION: dict[int, str] = {
    32: "4.x",
    34: "5.x",
    35: "5.x",
    36: "6.x",
}

_NVPMODEL_QUERY_NAME_PATTERN = re.compile(r"NV Power Mode:\s*(\S+)")


@dataclass(frozen=True)
class EnvironmentReport:
    """Everything `01_environment_check.py` writes to `outputs/env_report.json`.

    Attributes:
        is_jetson: Whether `marker_path` (normally `/etc/nv_tegra_release`) exists.
        marker_path: The path that was checked, recorded for traceability.
        l4t_version: Parsed `"R<major>.<revision>"` string from the marker
            file (e.g. `"R36.4.0"`), or `None` off-Jetson / unparseable.
        jetpack_generation: A deliberately coarse `"6.x"`-style label derived
            from `l4t_version`'s major number, or `None`. NOT the precise
            JetPack minor version (e.g. `6.1` vs `6.2`) -- that mapping
            changes between JetPack releases and isn't reliably inferable
            from `/etc/nv_tegra_release` alone; `sudo apt show
            nvidia-jetpack` (see `course/VENV.md`) is the authoritative
            source for the exact version.
        available_memory_mb: Currently available system RAM, in MiB.
        power_mode: Current `nvpmodel` power-mode name (e.g. `"15W"`), or
            `None` if `nvpmodel` isn't present/runnable (i.e. not a Jetson).
    """

    is_jetson: bool
    marker_path: str
    l4t_version: str | None
    jetpack_generation: str | None
    available_memory_mb: float
    power_mode: str | None


def is_jetson(marker_path: Path = DEFAULT_NV_TEGRA_RELEASE_PATH) -> bool:
    """Whether this machine is a Jetson device, per JetPack's marker file.

    Args:
        marker_path: Path to check. Defaults to `/etc/nv_tegra_release`,
            present on every JetPack image and absent everywhere else.

    Returns:
        `True` if `marker_path` exists.
    """
    return marker_path.is_file()


def parse_l4t_version(marker_text: str) -> str | None:
    """Extract the `"R<major>.<revision>"` L4T/BSP version from marker-file text.

    Args:
        marker_text: Contents of `/etc/nv_tegra_release`, e.g.
            `"# R36 (release), REVISION: 4.0, GCID: ..., BOARD: ..."`.

    Returns:
        `"R36.4.0"`-style string, or `None` if `marker_text` doesn't match
        the expected format.
    """
    match = _NV_TEGRA_RELEASE_PATTERN.search(marker_text)
    return f"R{match.group(1)}.{match.group(2)}" if match else None


def jetpack_generation_from_l4t(l4t_version: str | None) -> str | None:
    """Map an L4T major version to a coarse JetPack generation label.

    Args:
        l4t_version: `"R<major>.<revision>"` string from `parse_l4t_version`, or `None`.

    Returns:
        A `"6.x"`-style label, or `None` if `l4t_version` is `None` or its
        major version isn't in the known mapping (e.g. a JetPack release
        newer than this table).
    """
    if l4t_version is None:
        return None
    match = re.match(r"R(\d+)\.", l4t_version)
    if match is None:
        return None
    return _L4T_MAJOR_TO_JETPACK_GENERATION.get(int(match.group(1)))


def available_memory_mb() -> float:
    """Currently available system RAM, in MiB, via `psutil`.

    Returns:
        `psutil.virtual_memory().available`, converted to MiB.
    """
    import psutil

    return psutil.virtual_memory().available / (1024**2)


def parse_power_mode(nvpmodel_query_output: str) -> str | None:
    """Extract the power-mode name from `nvpmodel -q` stdout.

    Args:
        nvpmodel_query_output: Raw stdout, e.g. `"NV Power Mode: 15W\\n1\\n"`.

    Returns:
        The mode name (e.g. `"15W"`), or `None` if the expected pattern isn't found.
    """
    match = _NVPMODEL_QUERY_NAME_PATTERN.search(nvpmodel_query_output)
    return match.group(1) if match else None


def query_power_mode() -> str | None:
    """Run `nvpmodel -q` and return the parsed power-mode name, if available.

    Returns:
        The current power-mode name, or `None` if `nvpmodel` isn't
        present/runnable on this machine (i.e. this isn't running on a Jetson).
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_power_mode(result.stdout)


def build_environment_report(
    marker_path: Path = DEFAULT_NV_TEGRA_RELEASE_PATH,
) -> EnvironmentReport:
    """Assemble the full `EnvironmentReport` for the machine this runs on.

    Args:
        marker_path: Path checked for Jetson detection (see `is_jetson`).

    Returns:
        An `EnvironmentReport` with every field populated (or `None`/`False`
        for whatever doesn't apply off-Jetson).
    """
    on_jetson = is_jetson(marker_path)
    l4t_version = parse_l4t_version(marker_path.read_text(encoding="utf-8")) if on_jetson else None

    return EnvironmentReport(
        is_jetson=on_jetson,
        marker_path=str(marker_path),
        l4t_version=l4t_version,
        jetpack_generation=jetpack_generation_from_l4t(l4t_version),
        available_memory_mb=available_memory_mb(),
        power_mode=query_power_mode(),
    )
