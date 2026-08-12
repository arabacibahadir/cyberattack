from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from gridimpair.contracts import (
    ControlAction,
    PowerState,
    RunSummary,
    ScenarioConfig,
    make_message_event,
    stable_hash,
)
from gridimpair.sim.controller import SafeModeController
from gridimpair.sim.events import (
    BernoulliImpairment,
    BurstLossImpairment,
    EndpointInbox,
    EventQueue,
    JitterImpairment,
    generate_event_ledger,
)
from gridimpair.sim.policies import local_mode_enabled, policy_alarm
from gridimpair.sim.plant import PowerFlowPlant
from gridimpair.sim.profiles import make_profile
from gridimpair.sim.schedule import freeze_remote_schedule


@dataclass(frozen=True)
class ScenarioPair:
    clean: ScenarioConfig
    impaired: ScenarioConfig


def build_pair(config: ScenarioConfig) -> ScenarioPair:
    base_id = config.pair_id
    clean = replace(
        config,
        scenario_id=f"{base_id}-a",
        status="clean",
        impairment_seed=0,
    )
    impaired = replace(config, scenario_id=f"{base_id}-b", status="impaired")
    return ScenarioPair(clean=clean, impaired=impaired)


def _split_for_replicate(replicate: int) -> str:
    if 0 <= replicate <= 5:
        return "development"
    if replicate == 6:
        return "validation_a"
    if replicate == 7:
        return "validation_b"
    if replicate in {8, 9}:
        return "locked_test"
    raise ValueError("main-matrix replicate must be in [0, 9]")


def _main_conditions() -> list[tuple[int, float, float]]:
    return [
        (delay, drop, fraction)
        for delay in (0, 1, 3)
        for drop in (0.0, 0.1, 0.3, 0.6)
        for fraction in (0.25, 1.0)
        if not (delay == 0 and drop == 0.0)
    ]


def sample_impairment_episodes(
    *, seed: int, steps: int
) -> tuple[tuple[int, int], ...]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if steps < 780:
        return ((0, steps),)
    rng = random.Random(seed)
    first_duration = rng.randint(60, 120)
    first_start = rng.randint(180, 300)
    first = (first_start, first_start + first_duration)
    second_duration = rng.randint(60, 120)
    earliest_second = first[1] + 120
    latest_second = 780 - second_duration
    second_start = rng.randint(earliest_second, latest_second)
    return first, (second_start, second_start + second_duration)


def generate_main_scenario_matrix(*, base_seed: int = 2026) -> list[ScenarioConfig]:
    """Return the preregistered 3 × 22 × 10 × 2 paired run matrix."""

    scenarios: list[ScenarioConfig] = []
    for regime_index, regime in enumerate(
        ("balanced", "high_load_low_der", "low_load_high_der")
    ):
        for condition_index, (delay, drop, fraction) in enumerate(_main_conditions()):
            for replicate in range(10):
                pair_id = (
                    f"r{regime_index}-c{condition_index:02d}-rep{replicate:02d}"
                )
                seed_base = (
                    base_seed
                    + regime_index * 100_000
                    + condition_index * 1_000
                    + replicate * 10
                )
                impaired = ScenarioConfig(
                    scenario_id=f"{pair_id}-b",
                    pair_id=pair_id,
                    split=_split_for_replicate(replicate),
                    regime=regime,
                    delay_steps=delay,
                    drop_probability=drop,
                    endpoint_fraction=fraction,
                    status="impaired",
                    profile_seed=seed_base + 1,
                    schedule_seed=seed_base + 2,
                    impairment_seed=seed_base + 3,
                )
                pair = build_pair(impaired)
                scenarios.extend((pair.clean, pair.impaired))
    return scenarios


def generate_pilot_matrix(*, base_seed: int = 9026) -> list[ScenarioConfig]:
    """Return 36 pairs (72 runs) outside the 6/1/1/2 main access pool."""

    anchors = ((1, 0.1, 0.25), (3, 0.3, 1.0), (3, 0.6, 1.0))
    scenarios: list[ScenarioConfig] = []
    for regime_index, regime in enumerate(
        ("balanced", "high_load_low_der", "low_load_high_der")
    ):
        for anchor_index, (delay, drop, fraction) in enumerate(anchors):
            for replicate in range(4):
                pair_id = f"pilot-r{regime_index}-a{anchor_index}-rep{replicate}"
                seed_base = (
                    base_seed
                    + regime_index * 10_000
                    + anchor_index * 1_000
                    + replicate * 10
                )
                pair = build_pair(
                    ScenarioConfig(
                        scenario_id=f"{pair_id}-b",
                        pair_id=pair_id,
                        split="pilot",
                        regime=regime,
                        delay_steps=delay,
                        drop_probability=drop,
                        endpoint_fraction=fraction,
                        status="impaired",
                        profile_seed=seed_base + 1,
                        schedule_seed=seed_base + 2,
                        impairment_seed=seed_base + 3,
                    )
                )
                scenarios.extend((pair.clean, pair.impaired))
    return scenarios


def _q1_v2_conditions() -> list[dict[str, float | int | str]]:
    conditions: list[dict[str, float | int | str]] = [
        {
            "name": f"delay-{delay}",
            "impairment_kind": "bernoulli",
            "delay_steps": delay,
            "drop_probability": 0.0,
            "endpoint_fraction": 1.0,
        }
        for delay in (1, 3, 5)
    ]
    conditions.extend(
        {
            "name": f"drop-{drop}-{fraction}",
            "impairment_kind": "bernoulli",
            "delay_steps": 0,
            "drop_probability": drop,
            "endpoint_fraction": fraction,
        }
        for drop in (0.1, 0.3, 0.6)
        for fraction in (0.25, 1.0)
    )
    conditions.extend(
        [
            {
                "name": "burst-mild",
                "impairment_kind": "burst",
                "delay_steps": 0,
                "drop_probability": 0.0,
                "endpoint_fraction": 1.0,
                "burst_enter_probability": 0.03,
                "burst_exit_probability": 0.40,
            },
            {
                "name": "burst-severe",
                "impairment_kind": "burst",
                "delay_steps": 0,
                "drop_probability": 0.0,
                "endpoint_fraction": 1.0,
                "burst_enter_probability": 0.08,
                "burst_exit_probability": 0.20,
            },
            {
                "name": "jitter-3",
                "impairment_kind": "jitter",
                "delay_steps": 0,
                "drop_probability": 0.0,
                "endpoint_fraction": 1.0,
                "jitter_min_steps": 0,
                "jitter_max_steps": 3,
            },
        ]
    )
    return conditions


def _q1_v2_split_and_network(replicate: int) -> tuple[str, str]:
    if 0 <= replicate <= 5:
        return "development", "cigre_mv"
    if replicate in {6, 7}:
        return "validation_a", "cigre_mv"
    if replicate in {8, 9}:
        return "validation_b", "cigre_mv"
    if 10 <= replicate <= 13:
        return "locked_test", "ieee33_der"
    raise ValueError("q1-v2 replicate must be in [0, 13]")


def generate_q1_v2_scenario_matrix(
    *, base_seed: int = 2026
) -> list[ScenarioConfig]:
    scenarios: list[ScenarioConfig] = []
    for regime_index, regime in enumerate(
        ("balanced", "high_load_low_der", "low_load_high_der")
    ):
        for condition_index, condition in enumerate(
            _q1_v2_conditions()
        ):
            for replicate in range(14):
                split, network = _q1_v2_split_and_network(replicate)
                pair_id = (
                    f"q1v2-r{regime_index}-c{condition_index:02d}"
                    f"-rep{replicate:02d}"
                )
                seed_base = (
                    base_seed
                    + regime_index * 100_000
                    + condition_index * 1_000
                    + replicate * 10
                )
                pair = build_pair(
                    ScenarioConfig(
                        scenario_id=f"{pair_id}-b",
                        pair_id=pair_id,
                        split=split,
                        regime=regime,
                        delay_steps=int(condition["delay_steps"]),
                        drop_probability=float(
                            condition["drop_probability"]
                        ),
                        endpoint_fraction=float(
                            condition["endpoint_fraction"]
                        ),
                        status="impaired",
                        profile_seed=seed_base + 1,
                        schedule_seed=seed_base + 2,
                        impairment_seed=seed_base + 3,
                        network=network,
                        impairment_kind=str(
                            condition["impairment_kind"]
                        ),
                        burst_enter_probability=float(
                            condition.get(
                                "burst_enter_probability", 0.08
                            )
                        ),
                        burst_exit_probability=float(
                            condition.get(
                                "burst_exit_probability", 0.25
                            )
                        ),
                        jitter_min_steps=int(
                            condition.get("jitter_min_steps", 0)
                        ),
                        jitter_max_steps=int(
                            condition.get("jitter_max_steps", 5)
                        ),
                    )
                )
                scenarios.extend((pair.clean, pair.impaired))
    return scenarios


def generate_q1_v2_pilot_matrix(
    *, base_seed: int = 9026
) -> list[ScenarioConfig]:
    anchors = (
        _q1_v2_conditions()[1],
        _q1_v2_conditions()[4],
        _q1_v2_conditions()[9],
        _q1_v2_conditions()[11],
    )
    scenarios: list[ScenarioConfig] = []
    for network_index, network in enumerate(
        ("cigre_mv", "ieee33_der")
    ):
        for regime_index, regime in enumerate(
            ("balanced", "high_load_low_der", "low_load_high_der")
        ):
            for condition_index, condition in enumerate(anchors):
                pair_id = (
                    f"q1pilot-n{network_index}-r{regime_index}"
                    f"-c{condition_index:02d}"
                )
                seed_base = (
                    base_seed
                    + network_index * 100_000
                    + regime_index * 10_000
                    + condition_index * 100
                )
                pair = build_pair(
                    ScenarioConfig(
                        scenario_id=f"{pair_id}-b",
                        pair_id=pair_id,
                        split="pilot",
                        regime=regime,
                        delay_steps=int(condition["delay_steps"]),
                        drop_probability=float(
                            condition["drop_probability"]
                        ),
                        endpoint_fraction=float(
                            condition["endpoint_fraction"]
                        ),
                        status="impaired",
                        profile_seed=seed_base + 1,
                        schedule_seed=seed_base + 2,
                        impairment_seed=seed_base + 3,
                        network=network,
                        impairment_kind=str(
                            condition["impairment_kind"]
                        ),
                        burst_enter_probability=float(
                            condition.get(
                                "burst_enter_probability", 0.08
                            )
                        ),
                        burst_exit_probability=float(
                            condition.get(
                                "burst_exit_probability", 0.25
                            )
                        ),
                        jitter_min_steps=int(
                            condition.get("jitter_min_steps", 0)
                        ),
                        jitter_max_steps=int(
                            condition.get("jitter_max_steps", 5)
                        ),
                    )
                )
                scenarios.extend((pair.clean, pair.impaired))
    return scenarios


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_scenario(
    config: ScenarioConfig,
    policy_name: str,
    output_dir: Path,
    *,
    steps: int = 900,
    recycle: bool = False,
    storage_duration_hours: float = 2.0,
    step_seconds: float = 1.0,
    plant_factory: Callable[..., PowerFlowPlant] = PowerFlowPlant,
    impairment_kind: str | None = None,
    frozen_alarm_schedule: tuple[bool, ...] | None = None,
    write_event_ledger: bool = True,
) -> RunSummary:
    """Run the frozen causal order without wall-clock pacing.

    Each step uses the previous power state, delivers due messages, chooses an
    action, and only then solves the current power flow. The current solution
    therefore cannot influence the decision that produced it.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    impairment_kind = impairment_kind or config.impairment_kind
    if impairment_kind not in {"bernoulli", "burst", "jitter"}:
        raise ValueError("impairment_kind must be bernoulli, burst, or jitter")
    if frozen_alarm_schedule is not None and len(frozen_alarm_schedule) != steps:
        raise ValueError("frozen alarm schedule must match simulation steps")
    run_dir = Path(output_dir) / config.scenario_id
    run_dir.mkdir(parents=True, exist_ok=True)
    profile = make_profile(
        config.regime,
        config.profile_seed,
        steps,
        step_seconds=step_seconds,
    )
    impairment_intervals = (
        sample_impairment_episodes(seed=config.impairment_seed, steps=steps)
        if config.status == "impaired"
        else ()
    )
    plant = plant_factory(
        network_name=config.network,
        recycle=recycle,
        storage_duration_hours=storage_duration_hours,
        step_seconds=step_seconds,
    )
    schedule = freeze_remote_schedule(
        profile, plant.net, seed=config.schedule_seed
    )
    endpoint_rng = random.Random(config.schedule_seed + 17)
    affected_count = max(
        1, math.ceil(config.endpoint_fraction * len(schedule.endpoint_ids))
    )
    affected_endpoints = set(
        endpoint_rng.sample(list(schedule.endpoint_ids), k=affected_count)
    )
    endpoint_selection_hash = stable_hash(sorted(affected_endpoints))
    specs = [
        (time, endpoint, (schedule.p_setpoint[time, index],))
        for time in range(steps)
        for index, endpoint in enumerate(schedule.endpoint_ids)
    ]
    def make_impairment(seed: int):
        if impairment_kind == "bernoulli":
            return BernoulliImpairment(config.drop_probability, seed=seed)
        if impairment_kind == "burst":
            return BurstLossImpairment(
                config.burst_enter_probability,
                config.burst_exit_probability,
                seed=seed,
            )
        return JitterImpairment(
            config.jitter_min_steps,
            config.jitter_max_steps,
            seed=seed,
        )

    impairments = (
        [make_impairment(config.impairment_seed)]
        if config.status == "impaired"
        else []
    )
    events = generate_event_ledger(
        specs,
        impairments,
        fixed_delay=config.delay_steps if config.status == "impaired" else 0,
        affected_endpoints=affected_endpoints,
        active_intervals=impairment_intervals if config.status == "impaired" else (),
    )
    queue = EventQueue()
    controller = SafeModeController()
    inbox = EndpointInbox()
    latest_telemetry = (1.0, 0.0, 0.0, 0.0, 0.0)
    trace_columns = [
        "vm_max",
        "vm_min",
        "loading_max",
        "bus_p_sum",
        "bus_q_sum",
        "line_i_max",
        "delivered_p_sum",
        "message_age_max",
        "delivery_fraction",
        "detector_alarm",
        "curtailed_mw",
        "shed_mw",
        "safe_mode",
        "obs_vm_min",
        "obs_vm_p05",
        "obs_vm_p50",
        "obs_vm_p95",
        "obs_vm_max",
        "obs_loading_p50",
        "obs_loading_p95",
        "obs_loading_max",
        "obs_bus_p_mean",
        "obs_bus_p_std",
        "obs_bus_q_mean",
        "obs_bus_q_std",
        "obs_line_i_p50",
        "obs_line_i_p95",
        "obs_line_i_max",
        "message_age_mean",
        "stale_rejection_count",
        "telemetry_age",
        "local_mode",
    ]
    trace = np.lib.format.open_memmap(
        run_dir / "trace.npy",
        mode="w+",
        dtype=np.float32,
        shape=(steps, len(trace_columns)),
    )
    measurement_rng = np.random.default_rng(config.profile_seed + 500_003)
    event_index = 0
    telemetry_pending = []
    telemetry_impairment = (
        make_impairment(config.impairment_seed + 100_003)
        if config.status == "impaired"
        else None
    )
    previous_state: PowerState | None = None
    converged = 0
    for logical_time in range(steps):
        for pending in telemetry_pending:
            queue.push(pending)
        telemetry_pending = []
        while event_index < len(events) and events[event_index].production_time == logical_time:
            queue.push(events[event_index])
            event_index += 1
        for event in queue.pop_due(logical_time):
            if inbox.deliver(event, logical_time=logical_time):
                if event.endpoint_id == "telemetry-grid":
                    latest_telemetry = event.payload
        remote_p = tuple(
            (inbox.payload(endpoint) or (0.0,))[0]
            for endpoint in schedule.endpoint_ids
        )
        ages = [
            age
            if (age := inbox.age_of_information(endpoint, logical_time=logical_time))
            is not None
            else logical_time + 1
            for endpoint in schedule.endpoint_ids
        ]
        maximum_age = max(ages, default=0)
        detector_alarm = policy_alarm(
            policy_name,
            logical_time,
            message_age=maximum_age,
            impairment_intervals=impairment_intervals,
            frozen_alarm_schedule=frozen_alarm_schedule,
        )
        local_mode = local_mode_enabled(
            policy_name, detector_alarm=detector_alarm
        )
        local_state = previous_state if local_mode else None
        observation = {
            "vm_pu": local_state.vm_pu if local_state is not None else (1.0,),
            "loading_percent": (
                local_state.loading_percent
                if local_state is not None
                else (0.0,)
            ),
            "delivered_endpoint_count": sum(
                inbox.payload(endpoint) is not None
                for endpoint in schedule.endpoint_ids
            ),
            "remote_p_mw": remote_p,
            "message_age": maximum_age,
            "detector_alarm": detector_alarm,
            "local_mode": local_mode,
        }
        action: ControlAction = controller.act(observation, previous_state)
        current_state = plant.step(action, profile.at(logical_time))
        converged += int(current_state.converged)
        maximum_voltage = max(current_state.vm_pu, default=np.nan)
        minimum_voltage = min(current_state.vm_pu, default=np.nan)
        maximum_loading = max(current_state.loading_percent, default=np.nan)
        total_p = sum(current_state.p_mw)
        total_q = sum(current_state.q_mvar)
        maximum_current = max(current_state.i_ka, default=np.nan)
        observed_voltage = np.asarray(
            current_state.vm_pu, dtype=float
        ) + measurement_rng.normal(
            0.0, 0.001, len(current_state.vm_pu)
        )
        observed_loading = np.asarray(
            current_state.loading_percent, dtype=float
        ) + measurement_rng.normal(
            0.0, 0.10, len(current_state.loading_percent)
        )
        observed_p = np.asarray(current_state.p_mw, dtype=float)
        observed_p = observed_p + measurement_rng.normal(
            0.0,
            0.001 * max(1.0, float(np.max(np.abs(observed_p)))),
            observed_p.size,
        )
        observed_q = np.asarray(current_state.q_mvar, dtype=float)
        observed_q = observed_q + measurement_rng.normal(
            0.0,
            0.001 * max(1.0, float(np.max(np.abs(observed_q)))),
            observed_q.size,
        )
        observed_current = np.asarray(current_state.i_ka, dtype=float)
        observed_current = observed_current + measurement_rng.normal(
            0.0,
            0.0001,
            observed_current.size,
        )
        voltage_quantiles = np.quantile(
            observed_voltage, (0.0, 0.05, 0.5, 0.95, 1.0)
        )
        loading_quantiles = np.quantile(
            observed_loading, (0.5, 0.95, 1.0)
        )
        current_quantiles = np.quantile(
            observed_current, (0.5, 0.95, 1.0)
        )
        intended_p = sum(schedule.at(logical_time))
        applied_p = sum(action.p_mw)
        curtailed_mw = max(0.0, intended_p - applied_p)
        shed_mw = (
            current_state.load_p_mw
            / max(1e-8, 1 - action.shed_fraction)
            * action.shed_fraction
        )
        trace[logical_time] = (
            maximum_voltage,
            minimum_voltage,
            maximum_loading,
            total_p,
            total_q,
            maximum_current,
            sum(remote_p),
            maximum_age,
            sum(
                inbox.payload(endpoint) is not None
                for endpoint in schedule.endpoint_ids
            )
            / max(1, len(schedule.endpoint_ids)),
            int(detector_alarm),
            curtailed_mw,
            shed_mw,
            int(action.safe_mode),
            *voltage_quantiles,
            *loading_quantiles,
            float(np.mean(observed_p)),
            float(np.std(observed_p)),
            float(np.mean(observed_q)),
            float(np.std(observed_q)),
            *current_quantiles,
            float(np.mean(ages)),
            inbox.rejected_stale,
            (
                inbox.age_of_information(
                    "telemetry-grid", logical_time=logical_time
                )
                if inbox.age_of_information(
                    "telemetry-grid", logical_time=logical_time
                )
                is not None
                else logical_time + 1
            ),
            int(local_mode),
        )
        telemetry_is_active = any(
            start <= logical_time + 1 < stop
            for start, stop in impairment_intervals
        )
        telemetry_event = make_message_event(
            production_time=logical_time + 1,
            intended_delivery_time=logical_time + 1,
            actual_delivery_time=(
                logical_time
                + 1
                + (
                    config.delay_steps
                    if config.status == "impaired" and telemetry_is_active
                    else 0
                )
            ),
            sequence=len(events) + logical_time,
            endpoint_id="telemetry-grid",
            payload=(
                maximum_voltage,
                maximum_loading,
                total_p,
                total_q,
                maximum_current,
            ),
            dropped=False,
        )
        if telemetry_impairment is not None and telemetry_is_active:
            telemetry_event = telemetry_impairment.apply(telemetry_event)
        telemetry_pending.append(telemetry_event)
        previous_state = current_state
    for pending in telemetry_pending:
        queue.push(pending)
    trace.flush()
    public_manifest = {
        "scenario_id": config.scenario_id,
        "pair_id": config.pair_id,
        "split": config.split,
        "status": config.status if config.split != "locked_test" else "restricted",
        "regime": config.regime,
        "condition": {
            "delay_steps": config.delay_steps,
            "drop_probability": config.drop_probability,
            "endpoint_fraction": config.endpoint_fraction,
            "impairment_kind": impairment_kind,
        },
        "policy": policy_name,
        "profile_hash": profile.profile_hash,
        "schedule_hash": schedule.schedule_hash,
        "endpoint_selection_hash": endpoint_selection_hash,
        "steps": steps,
    }
    restricted_manifest = {
        "scenario_id": config.scenario_id,
        "pair_id": config.pair_id,
        "status": config.status,
        "communication_intervals": [
            list(interval) for interval in impairment_intervals
        ],
        "impairment_seed": config.impairment_seed,
    }
    event_rows = [asdict(event) for event in queue.ledger]
    _write_json(run_dir / "public_manifest.json", public_manifest)
    _write_json(run_dir / "trace_columns.json", {"columns": trace_columns})
    event_ledger_path = (
        Path(output_dir) / "restricted" / f"{config.scenario_id}.events.json"
        if config.status == "impaired"
        else run_dir / "event_ledger.json"
    )
    if write_event_ledger:
        _write_json(event_ledger_path, event_rows)
    _write_json(
        Path(output_dir) / "restricted" / f"{config.scenario_id}.json",
        restricted_manifest,
    )
    artifact_hash = stable_hash(
        {"public": public_manifest, "events": event_rows, "trace_shape": trace.shape}
    )
    summary = RunSummary(
        scenario_id=config.scenario_id,
        pair_id=config.pair_id,
        status=config.status,
        steps=steps,
        converged_steps=converged,
        event_count=len(event_rows),
        trace_path=str((run_dir / "trace.npy").resolve()),
        artifact_hash=artifact_hash,
    )
    _write_json(run_dir / "summary.json", asdict(summary))
    return summary
