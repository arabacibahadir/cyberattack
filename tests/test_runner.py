from __future__ import annotations

import os
from dataclasses import replace

import pytest

from gridimpair.config import load_config
from gridimpair.runner import (
    _merge_timing_rows,
    _peak_rss_mb,
    _select_scenario_matrix,
    build_parser,
)


def test_runner_accepts_only_public_stages() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        ["--stage", "audit", "--config", "configs/base.toml", "--resume", "--limit", "2"]
    )
    assert parsed.stage == "audit"
    assert parsed.resume is True
    assert parsed.limit == 2
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--stage", "unknown", "--config", "configs/base.toml"])
    assert error.value.code == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows memory API contract")
def test_windows_peak_rss_measurement_is_nonzero() -> None:
    assert _peak_rss_mb() > 0


def test_resume_timing_merge_preserves_completed_rows() -> None:
    existing = [
        {"scenario_id": "a", "wall_seconds": 2.0, "peak_rss_mb": 100.0}
    ]
    merged = _merge_timing_rows(existing, [])
    assert merged == existing
    updated = _merge_timing_rows(
        existing,
        [{"scenario_id": "a", "wall_seconds": 1.5, "peak_rss_mb": 110.0}],
    )
    assert updated == [
        {"scenario_id": "a", "wall_seconds": 1.5, "peak_rss_mb": 110.0}
    ]


def test_q1_v2_protocol_selects_new_isolated_matrices() -> None:
    base = load_config("configs/base.toml")
    q1 = replace(
        base,
        raw={
            **base.raw,
            "protocol": {"name": "q1_v2"},
        },
    )
    assert len(_select_scenario_matrix(q1, pilot=True)) == 48
    assert len(_select_scenario_matrix(q1, pilot=False)) == 1008
