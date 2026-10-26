"""Unit tests for resource_monitor.py's pure tegrastats-line parsing.

`TegrastatsRamMonitor` (subprocess) and `PeakRssTracker` (psutil) aren't
covered here -- same reasoning as `vision_bridge`'s tegrastats/nvpmodel
subprocess wrappers: side-effecting glue around hardware that isn't
present in this sandbox, reviewed rather than executed. Run with:
    cd ros2_ws/scripts && pytest test_resource_monitor.py
"""

from resource_monitor import parse_tegrastats_ram_mb


def test_parse_tegrastats_ram_mb_extracts_used_mb() -> None:
    line = "RAM 3241/7620MB (lfb 4x4MB) SWAP 0/3810MB (cached 0MB) CPU [15%@1907]"
    assert parse_tegrastats_ram_mb(line) == 3241


def test_parse_tegrastats_ram_mb_no_match_returns_none() -> None:
    assert parse_tegrastats_ram_mb("garbage output") is None


def test_parse_tegrastats_ram_mb_empty_string() -> None:
    assert parse_tegrastats_ram_mb("") is None
