"""Unit tests for `src.evaluation.jetson` (tegrastats parsing + result summarization)."""

from __future__ import annotations

import pytest

from src.evaluation.jetson import (
    NVPMODEL_MODES,
    parse_nvpmodel_query,
    parse_tegrastats_gpu_utilization,
    parse_tegrastats_ram_mb,
    parse_tegrastats_temps,
    summarize_mode_result,
)


def test_nvpmodel_modes_cover_maxn_15w_7w() -> None:
    assert NVPMODEL_MODES == {0: "MAXN_SUPER", 1: "15W", 2: "7W"}


def test_parse_tegrastats_temps_extracts_all_sensors() -> None:
    lines = [
        "RAM 3241/7620MB CPU [15%@1907] EMC_FREQ 0% GR3D_FREQ 0% "
        "AO@41C GPU@40.5C CPU@41C tj@41.5C SOC2@40C",
        "RAM 3300/7620MB CPU [20%@1907] EMC_FREQ 0% GR3D_FREQ 5% "
        "AO@42C GPU@41C CPU@42C tj@42.5C SOC2@41C",
    ]

    readings = parse_tegrastats_temps(lines)

    assert readings["GPU"] == [40.5, 41.0]
    assert readings["CPU"] == [41.0, 42.0]
    assert readings["tj"] == [41.5, 42.5]


def test_parse_tegrastats_temps_empty_input_returns_empty_dict() -> None:
    assert parse_tegrastats_temps([]) == {}


def test_summarize_mode_result_computes_percentiles_and_temp_stats() -> None:
    result = summarize_mode_result(
        mode_id=1,
        iterations=3,
        batch_size=4,
        durations_ms=[30.0, 10.0, 20.0],
        temp_readings={"GPU": [40.0, 42.0, 44.0], "CPU": [41.0]},
    )

    assert result.mode_name == "15W"
    assert result.iterations == 3
    assert result.batch_size == 4
    assert result.latency_p50_ms == 20.0
    assert result.latency_mean_ms == 20.0
    assert result.mean_temps_c == {"GPU": 42.0, "CPU": 41.0}
    assert result.max_temps_c == {"GPU": 44.0, "CPU": 41.0}


def test_summarize_mode_result_unknown_mode_raises_key_error() -> None:
    with pytest.raises(KeyError):
        summarize_mode_result(
            mode_id=99, iterations=1, batch_size=1, durations_ms=[1.0], temp_readings={}
        )


def test_parse_nvpmodel_query_extracts_mode_name_and_id() -> None:
    output = "NV Power Mode: MAXN_SUPER\n0\n"
    assert parse_nvpmodel_query(output) == {"mode_name": "MAXN_SUPER", "mode_id": 0}


def test_parse_nvpmodel_query_handles_unexpected_output() -> None:
    assert parse_nvpmodel_query("command not found") == {"mode_name": None, "mode_id": None}


def test_parse_tegrastats_ram_mb_extracts_used_and_total() -> None:
    line = "RAM 3241/7620MB (lfb 4x2MB) CPU [15%@1907] GR3D_FREQ 0%"
    assert parse_tegrastats_ram_mb(line) == (3241, 7620)


def test_parse_tegrastats_ram_mb_no_match_returns_none() -> None:
    assert parse_tegrastats_ram_mb("no ram field here") is None


def test_parse_tegrastats_gpu_utilization_with_bracketed_frequency() -> None:
    line = "RAM 3241/7620MB GR3D_FREQ 12%@[420] cpu@45.5C"
    assert parse_tegrastats_gpu_utilization(line) == 12.0


def test_parse_tegrastats_gpu_utilization_with_plain_frequency() -> None:
    line = "RAM 3241/7620MB GR3D_FREQ 34%@729 cpu@45.5C"
    assert parse_tegrastats_gpu_utilization(line) == 34.0


def test_parse_tegrastats_gpu_utilization_no_match_returns_none() -> None:
    assert parse_tegrastats_gpu_utilization("no gpu field here") is None
