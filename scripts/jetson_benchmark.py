"""Cross-power-mode latency/thermal benchmark for the Jetson Orin Nano.

Sweeps `sudo nvpmodel -m {0,1,2}` (MAXN Super / 15W / 7W), runs the
configured VLM image encoder N times per mode, and records latency
percentiles alongside `tegrastats` temperature readings to
`outputs/jetson_benchmark.json`. This is the baseline comparison point
referenced by week 09 (quantization/TensorRT) and week 10 (ops/monitoring/
reliability).

Run directly on the device:
    PYTHONPATH=. python scripts/jetson_benchmark.py --iterations 50

Run from a dev machine over SSH (re-executes this script on the device via
`ssh`, then copies the JSON report back via `scp`):
    python scripts/jetson_benchmark.py --host jetson@192.168.1.50 \\
        --remote-path ~/vlm-course --iterations 50
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation.jetson import (
    NVPMODEL_MODES,
    ModeBenchmarkResult,
    parse_tegrastats_temps,
    summarize_mode_result,
)


class NotOnJetsonError(RuntimeError):
    """Raised when this script is run on a machine without `nvpmodel` (not a real Jetson)."""


def check_nvpmodel_available() -> None:
    """Fail fast with a clear message if `nvpmodel` isn't on this machine.

    Without this check, `set_nvpmodel`'s `subprocess.run(..., check=True)`
    fails with a bare `CalledProcessError` + raw traceback the first time
    `benchmark_mode` is called -- correct, but useless for telling a dev
    machine apart from a real Jetson with a transient `sudo`/permissions
    problem. This turns "`sudo: nvpmodel: command not found`" into an
    actionable message before any benchmarking work starts.

    Raises:
        NotOnJetsonError: If `nvpmodel` isn't found on `PATH`.
    """
    import shutil

    if shutil.which("nvpmodel") is None:
        raise NotOnJetsonError(
            "'nvpmodel' not found on this machine -- this script sweeps real "
            "Jetson Orin Nano power modes and only runs ON the device. If "
            "you're on a dev machine, either run this directly on the "
            "Jetson, or use `--host user@jetson-ip` from here to run it "
            "there over SSH (see `python3 scripts/jetson_benchmark.py --help`)."
        )


def set_nvpmodel(mode_id: int, settle_seconds: float) -> None:
    """Switch the device power mode via `sudo nvpmodel -m <mode_id>` and let clocks settle.

    Args:
        mode_id: Target `nvpmodel` mode (see `NVPMODEL_MODES`).
        settle_seconds: Time to sleep after switching, before measuring.

    Raises:
        subprocess.CalledProcessError: If `nvpmodel` exits non-zero.
    """
    subprocess.run(["sudo", "nvpmodel", "-m", str(mode_id)], check=True)
    time.sleep(settle_seconds)


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


def benchmark_mode(
    mode_id: int,
    inference: Any,
    sample_image: Any,
    iterations: int,
    batch_size: int,
    tegrastats_interval_ms: int,
    settle_seconds: float,
) -> ModeBenchmarkResult:
    """Switch to `mode_id`, run `iterations` inference calls, and summarize latency + thermals.

    Args:
        mode_id: `nvpmodel -m` argument to switch to before measuring.
        inference: A `VLMInference` implementation (e.g. `CLIPInference`).
        sample_image: One representative PIL image, replicated to fill each batch.
        iterations: Number of timed inference calls.
        batch_size: Batch size used for each inference call.
        tegrastats_interval_ms: Sampling interval for the background `tegrastats` monitor.
        settle_seconds: Time to wait after switching `nvpmodel` mode before measuring.

    Returns:
        A `ModeBenchmarkResult` for this mode.
    """
    set_nvpmodel(mode_id, settle_seconds)

    batch = [sample_image] * batch_size
    durations_ms: list[float] = []

    with TegrastatsMonitor(interval_ms=tegrastats_interval_ms) as monitor:
        for _ in range(iterations):
            start = time.perf_counter()
            inference.encode_image(batch)
            durations_ms.append((time.perf_counter() - start) * 1000)

    temp_readings = parse_tegrastats_temps(monitor.lines)
    return summarize_mode_result(mode_id, iterations, batch_size, durations_ms, temp_readings)


def run_local(args: argparse.Namespace) -> None:
    """Run the full power-mode sweep on this device and write the JSON report.

    Args:
        args: Parsed CLI arguments from `parse_args`.
    """
    from PIL import Image

    from src.config import load_config
    from src.models.clip import CLIPInference, CLIPLoader

    check_nvpmodel_available()
    config = load_config(args.config)
    loader = CLIPLoader(config.model, config.memory)
    inference = CLIPInference(loader)

    sample_image = Image.new("RGB", (config.inference.image_size, config.inference.image_size))
    batch_size = args.batch_size if args.batch_size is not None else config.inference.batch_size

    results = [
        benchmark_mode(
            mode_id,
            inference,
            sample_image,
            iterations=args.iterations,
            batch_size=batch_size,
            tegrastats_interval_ms=args.tegrastats_interval_ms,
            settle_seconds=args.settle_seconds,
        )
        for mode_id in args.modes
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
    print(f"Jetson benchmark report written to '{output_path}'.")


def run_remote(args: argparse.Namespace) -> None:
    """Re-run this script on `args.host` over SSH, then copy the JSON report back via `scp`.

    Args:
        args: Parsed CLI arguments from `parse_args`; `args.host` must be set.

    Raises:
        subprocess.CalledProcessError: If the remote `ssh` run or the `scp` copy fails.
    """
    modes_arg = ",".join(str(mode_id) for mode_id in args.modes)
    remote_output = f"{args.remote_path.rstrip('/')}/{args.output}"
    remote_command = (
        f"cd {args.remote_path} && PYTHONPATH=. python3 scripts/jetson_benchmark.py "
        f"--config {args.config} --iterations {args.iterations} --modes {modes_arg} "
        f"--tegrastats-interval-ms {args.tegrastats_interval_ms} "
        f"--settle-seconds {args.settle_seconds} --output {args.output}"
    )
    subprocess.run(["ssh", args.host, remote_command], check=True)

    local_output = Path(args.output)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", f"{args.host}:{remote_output}", str(local_output)], check=True)
    print(f"Fetched remote benchmark report to '{local_output}'.")


def _parse_modes(raw: str) -> list[int]:
    """Parse a comma-separated `--modes` argument, validating against `NVPMODEL_MODES`.

    Args:
        raw: Comma-separated mode ids, e.g. `"0,1,2"`.

    Returns:
        The parsed mode ids, in the order given.

    Raises:
        argparse.ArgumentTypeError: If any id is not a key of `NVPMODEL_MODES`.
    """
    modes = [int(part) for part in raw.split(",") if part.strip()]
    unknown = sorted(set(modes) - set(NVPMODEL_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown nvpmodel mode id(s) {unknown}; expected any of {sorted(NVPMODEL_MODES)}."
        )
    return modes


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `jetson_benchmark.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument(
        "--iterations", type=int, default=30, help="Timed inference calls per power mode."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Overrides inference.batch_size from config."
    )
    parser.add_argument(
        "--modes",
        type=_parse_modes,
        default=[0, 1, 2],
        help="Comma-separated nvpmodel mode ids to sweep (default: 0,1,2 = MAXN Super,15W,7W).",
    )
    parser.add_argument("--tegrastats-interval-ms", type=int, default=500)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=3.0,
        help="Wait time after switching nvpmodel mode, before measuring.",
    )
    parser.add_argument("--output", default="outputs/jetson_benchmark.json")
    parser.add_argument(
        "--host", default=None, help="SSH target (e.g. jetson@192.168.1.50); omit to run locally."
    )
    parser.add_argument(
        "--remote-path",
        default="~/vlm-course",
        help="Repo path on the remote device, used only together with --host.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: run locally, or over SSH when `--host` is given."""
    args = parse_args()
    if args.host:
        run_remote(args)
    else:
        run_local(args)


if __name__ == "__main__":
    main()
