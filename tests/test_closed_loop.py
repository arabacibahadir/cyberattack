from __future__ import annotations

import pandas as pd
import pytest

from gridimpair.evaluation.closed_loop import (
    paired_endpoint_delta,
    policy_regret_table,
    summarize_electrical_trace,
)


def test_electrical_endpoints_use_time_integrals_not_window_counts() -> None:
    frame = pd.DataFrame(
        {
            "vm_min": [1.0, 1.0, 0.94],
            "vm_max": [1.0, 1.06, 1.0],
            "loading_max": [90.0, 110.0, 105.0],
            "curtailed_mw": [0.0, 0.2, 0.1],
            "shed_mw": [0.0, 0.0, 0.3],
            "safe_mode": [0, 1, 1],
        }
    )
    result = summarize_electrical_trace(frame, step_seconds=1.0)
    assert result.voltage_violation_seconds == 2.0
    assert result.overload_integral_percent_seconds == 15.0
    assert result.curtailment_mwh == pytest.approx(0.3 / 3600)
    assert result.unserved_energy_mwh == pytest.approx(0.3 / 3600)
    assert result.safe_mode_fraction == pytest.approx(2 / 3)


def test_paired_endpoint_delta_is_impaired_minus_clean() -> None:
    clean = pd.DataFrame(
        {
            "vm_min": [1.0],
            "vm_max": [1.0],
            "loading_max": [90.0],
            "curtailed_mw": [0.0],
            "shed_mw": [0.0],
            "safe_mode": [0],
        }
    )
    impaired = clean.copy()
    impaired.loc[0, "loading_max"] = 120.0
    delta = paired_endpoint_delta(clean, impaired, step_seconds=1.0)
    assert delta["overload_integral_percent_seconds"] == 20.0


def test_policy_regret_keeps_clean_false_alarm_cost_visible() -> None:
    absolute = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "status": "clean",
                "policy": "P0",
                "endpoint": "curtailment_mwh",
                "value": 0.0,
            },
            {
                "pair_id": "p1",
                "status": "clean",
                "policy": "candidate",
                "endpoint": "curtailment_mwh",
                "value": 2.0,
            },
            {
                "pair_id": "p1",
                "status": "impaired",
                "policy": "P0",
                "endpoint": "curtailment_mwh",
                "value": 1.0,
            },
            {
                "pair_id": "p1",
                "status": "impaired",
                "policy": "candidate",
                "endpoint": "curtailment_mwh",
                "value": 2.5,
            },
        ]
    )
    regret = policy_regret_table(absolute, reference_policy="P0")
    candidate = regret.loc[regret["policy"] == "candidate"].set_index("status")
    assert candidate.loc["clean", "regret"] == 2.0
    assert candidate.loc["impaired", "regret"] == 1.5
