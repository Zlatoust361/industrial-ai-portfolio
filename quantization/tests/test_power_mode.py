"""Unit tests for `utils/power_mode.py`.

`TegrastatsMonitor`/`set_nvpmodel` need a real `tegrastats`/`sudo nvpmodel`
binary to exercise meaningfully -- same as `scripts/jetson_benchmark.py`'s
own untested-at-the-unit-level equivalents (this repo's established
precedent; see `src/evaluation/jetson.py`'s own tests for the
hardware-free parsing logic both depend on instead). `query_power_mode`
is cheap to check for real: no `nvpmodel` binary in this sandbox either,
so it must return `None` cleanly, not raise.
"""

from __future__ import annotations

from utils.power_mode import query_power_mode


def test_query_power_mode_returns_none_without_nvpmodel_binary() -> None:
    assert query_power_mode() is None
