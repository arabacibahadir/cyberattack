from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gridimpair.contracts import ScenarioConfig
from gridimpair.sim.scenario import (
    build_pair,
    generate_main_scenario_matrix,
    generate_pilot_matrix,
    generate_q1_v2_pilot_matrix,
    generate_q1_v2_scenario_matrix,
    run_scenario,
    sample_impairment_episodes,
)


def _config(status: str) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=f"pair-1-{status}",
        pair_id="pair-1",
        split="development",
        regime="smooth",
        delay_steps=1,
        drop_probability=0.2,
        endpoint_fraction=0.5,
        status=status,
        profile_seed=11,
        schedule_seed=12,
        impairment_seed=13 if status == "impaired" else 0,
    )


def test_clean_and_impaired_pair_share_frozen_inputs() -> None:
    pair = build_pair(_config("impaired"))
    assert pair.clean.profile_seed == pair.impaired.profile_seed
    assert pair.clean.schedule_seed == pair.impaired.schedule_seed
    assert pair.clean.pair_id == pair.impaired.pair_id
    assert pair.clean.delay_steps == pair.impaired.delay_steps
    assert pair.clean.drop_probability == pair.impaired.drop_probability
    assert pair.clean.endpoint_fraction == pair.impaired.endpoint_fraction
    assert pair.clean.status == "clean"
    assert pair.impaired.status == "impaired"


def test_public_manifest_never_contains_exact_onset(tmp_path: Path) -> None:
    summary = run_scenario(_config("clean"), "watchdog", tmp_path, steps=2)
    public = json.loads(
        (tmp_path / "pair-1-clean" / "public_manifest.json").read_text(encoding="utf-8")
    )
    assert summary.steps == 2
    assert "onset" not in json.dumps(public).lower()
    assert (tmp_path / "restricted" / "pair-1-clean.json").exists()


def test_main_matrix_has_1320_runs_and_pair_preserving_splits() -> None:
    matrix = generate_main_scenario_matrix(base_seed=2026)
    assert len(matrix) == 1320
    counts = {}
    by_pair = {}
    for scenario in matrix:
        counts[scenario.split] = counts.get(scenario.split, 0) + 1
        by_pair.setdefault(scenario.pair_id, set()).add(scenario.split)
    assert counts == {
        "development": 792,
        "validation_a": 132,
        "validation_b": 132,
        "locked_test": 264,
    }
    assert max(map(len, by_pair.values())) == 1


def test_pilot_matrix_has_72_runs_outside_main_split() -> None:
    matrix = generate_pilot_matrix(base_seed=99)
    assert len(matrix) == 72
    assert {scenario.split for scenario in matrix} == {"pilot"}


def test_two_episodes_respect_warmup_duration_washout_and_horizon() -> None:
    first, second = sample_impairment_episodes(seed=17, steps=900)
    assert 180 <= first[0] <= 300
    assert 60 <= first[1] - first[0] <= 120
    assert 60 <= second[1] - second[0] <= 120
    assert second[0] - first[1] >= 120
    assert second[1] <= 780


def test_robustness_shift_and_frozen_policy_are_recorded(tmp_path: Path) -> None:
    run_scenario(
        _config("impaired"),
        "candidate",
        tmp_path,
        steps=2,
        impairment_kind="jitter",
        frozen_alarm_schedule=(False, True),
        write_event_ledger=False,
    )
    public = json.loads(
        (tmp_path / "pair-1-impaired" / "public_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert public["condition"]["impairment_kind"] == "jitter"
    assert public["policy"] == "candidate"
    assert not (tmp_path / "restricted" / "pair-1-impaired.events.json").exists()


def test_scenario_trace_records_age_of_information_not_time_since_delivery(
    tmp_path: Path,
) -> None:
    config = replace(
        _config("impaired"),
        delay_steps=3,
        drop_probability=0.0,
        endpoint_fraction=1.0,
    )
    run_scenario(config, "shadow", tmp_path, steps=5)
    columns = json.loads(
        (tmp_path / config.scenario_id / "trace_columns.json").read_text(
            encoding="utf-8"
        )
    )["columns"]
    trace = np.load(tmp_path / config.scenario_id / "trace.npy")
    age = trace[:, columns.index("message_age_max")]
    assert age[3] == 3
    assert age[4] == 3


def test_scenario_can_run_on_independent_ieee33_der_feeder(
    tmp_path: Path,
) -> None:
    config = replace(_config("clean"), network="ieee33_der")
    summary = run_scenario(
        config,
        "shadow",
        tmp_path,
        steps=1,
        step_seconds=2.0,
    )
    assert summary.converged_steps == 1


def test_q1_v2_matrix_sequesters_locked_topology_and_preserves_pairs() -> None:
    matrix = generate_q1_v2_scenario_matrix(base_seed=2026)
    assert len(matrix) == 1008
    counts: dict[str, int] = {}
    pair_splits: dict[str, set[str]] = {}
    for scenario in matrix:
        counts[scenario.split] = counts.get(scenario.split, 0) + 1
        pair_splits.setdefault(scenario.pair_id, set()).add(scenario.split)
        if scenario.split == "locked_test":
            assert scenario.network == "ieee33_der"
        else:
            assert scenario.network == "cigre_mv"
    assert counts == {
        "development": 432,
        "validation_a": 144,
        "validation_b": 144,
        "locked_test": 288,
    }
    assert max(map(len, pair_splits.values())) == 1
    assert {
        scenario.impairment_kind
        for scenario in matrix
        if scenario.status == "impaired"
    } == {"bernoulli", "burst", "jitter"}


def test_q1_v2_pilot_covers_both_feeders_without_main_ids() -> None:
    pilot = generate_q1_v2_pilot_matrix(base_seed=9026)
    assert len(pilot) == 48
    assert {scenario.network for scenario in pilot} == {
        "cigre_mv",
        "ieee33_der",
    }
    assert {scenario.split for scenario in pilot} == {"pilot"}
    assert all(scenario.pair_id.startswith("q1pilot-") for scenario in pilot)


def test_q1_v2_trace_contains_topology_invariant_observability_tiers(
    tmp_path: Path,
) -> None:
    config = replace(_config("clean"), network="ieee33_der")
    run_scenario(config, "shadow", tmp_path, steps=2)
    columns = json.loads(
        (tmp_path / config.scenario_id / "trace_columns.json").read_text(
            encoding="utf-8"
        )
    )["columns"]
    for name in (
        "obs_vm_p50",
        "obs_loading_p95",
        "obs_bus_p_std",
        "obs_line_i_p95",
        "message_age_mean",
        "stale_rejection_count",
        "telemetry_age",
    ):
        assert name in columns
