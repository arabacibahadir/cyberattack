from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from gridimpair.config import AppConfig, load_config
from gridimpair.contracts import ScenarioConfig, stable_hash
from gridimpair.ledger import StageLedger


STAGES = ("audit", "pilot", "simulate", "train", "evaluate", "figures", "all")


def _merge_timing_rows(
    existing: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_scenario = {
        str(row["scenario_id"]): dict(row)
        for row in existing
    }
    for row in new_rows:
        by_scenario[str(row["scenario_id"])] = dict(row)
    return [by_scenario[key] for key in sorted(by_scenario)]


def _pilot_gate(output: Path, scenarios: Sequence[ScenarioConfig]) -> dict[str, object]:
    by_pair: dict[str, dict[str, np.ndarray]] = {}
    finite = True
    fully_converged = True
    vm_min = float("inf")
    vm_max = float("-inf")
    for scenario in scenarios:
        scenario_dir = output / scenario.scenario_id
        trace = np.load(scenario_dir / "trace.npy")
        columns = json.loads(
            (scenario_dir / "trace_columns.json").read_text(encoding="utf-8")
        )["columns"]
        summary = json.loads(
            (scenario_dir / "summary.json").read_text(encoding="utf-8")
        )
        finite = finite and bool(np.isfinite(trace).all())
        fully_converged = fully_converged and (
            int(summary["converged_steps"]) == int(summary["steps"])
        )
        vm_min = min(vm_min, float(np.min(trace[:, columns.index("vm_min")])))
        vm_max = max(vm_max, float(np.max(trace[:, columns.index("vm_max")])))
        observed = trace[
            :,
            [
                columns.index("obs_vm_p50"),
                columns.index("obs_loading_p95"),
                columns.index("delivery_fraction"),
                columns.index("message_age_max"),
            ],
        ]
        by_pair.setdefault(scenario.pair_id, {})[scenario.status] = observed
    paired_differences = [
        float(np.mean(np.abs(statuses["impaired"] - statuses["clean"])))
        for statuses in by_pair.values()
        if {"clean", "impaired"} <= statuses.keys()
    ]
    detectable_fraction = float(
        np.mean(np.asarray(paired_differences) > 1e-8)
    ) if paired_differences else 0.0
    checks = {
        "all_values_finite": finite,
        "all_power_flows_converged": fully_converged,
        "broad_voltage_sanity": vm_min >= 0.5 and vm_max <= 1.5,
        "paired_impairment_detectability": detectable_fraction >= 0.25,
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "scenario_count": len(scenarios),
        "pair_count": len(paired_differences),
        "minimum_voltage_pu": vm_min,
        "maximum_voltage_pu": vm_max,
        "detectable_pair_fraction": detectable_fraction,
        "mean_paired_observation_difference": (
            float(np.mean(paired_differences))
            if paired_differences else 0.0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridimpair",
        description="Local causal grid-impairment experiment runner",
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _config_hash(config: AppConfig) -> str:
    return stable_hash(config.raw)


def _peak_rss_mb() -> float:
    if os.name == "nt":
        from ctypes import wintypes

        class _MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_MemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        success = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise OSError(
                ctypes.get_last_error(),
                "GetProcessMemoryInfo failed",
            )
        return counters.PeakWorkingSetSize / 2**20
    import resource

    units = 1024 if sys.platform != "darwin" else 1
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units / 2**20


def _run_one_scenario(
    scenario: ScenarioConfig,
    output: Path,
    steps: int,
    recycle: bool,
    storage_duration_hours: float,
    step_seconds: float,
) -> tuple[dict[str, object], float, float]:
    from gridimpair.sim.scenario import run_scenario

    started = time.perf_counter()
    summary = run_scenario(
        scenario,
        "shadow",
        output,
        steps=steps,
        recycle=recycle,
        storage_duration_hours=storage_duration_hours,
        step_seconds=step_seconds,
    )
    return asdict(summary), time.perf_counter() - started, _peak_rss_mb()


def _run_one_robustness(
    item: object,
    output: Path,
    steps: int,
    recycle: bool,
    storage_duration_hours: float,
    step_seconds: float,
) -> tuple[dict[str, object], float, float]:
    from gridimpair.sim.scenario import run_scenario

    started = time.perf_counter()
    summary = run_scenario(
        item.config,
        "shadow",
        output,
        steps=steps,
        recycle=recycle,
        storage_duration_hours=storage_duration_hours,
        step_seconds=step_seconds,
        impairment_kind=item.impairment_kind,
    )
    return asdict(summary), time.perf_counter() - started, _peak_rss_mb()


def _run_robustness_scenarios(
    config: AppConfig, *, resume: bool, limit: int | None
) -> None:
    if not bool(config.raw.get("robustness", {}).get("enabled", True)):
        return
    from gridimpair.sim.robustness import generate_robustness_matrix

    settings = config.raw.get("robustness", {})
    items = generate_robustness_matrix(
        base_seed=config.runtime.seed + 40_000,
        replicates=int(settings.get("replicates_per_regime_shift", 3)),
    )
    allowed = set(settings.get("impairment_kinds", ["burst", "jitter"]))
    items = [item for item in items if item.impairment_kind in allowed]
    if limit is not None:
        items = items[:limit]
    output = config.paths.artifact_root / "robustness"
    ledger = StageLedger(config.paths.artifact_root / "ledgers" / "stages.sqlite")
    claimed = []
    for item in items:
        item_hash = stable_hash(
            {
                "scenario": asdict(item.config),
                "impairment_kind": item.impairment_kind,
            }
        )
        claim = ledger.claim(
            "robustness",
            item.config.scenario_id,
            item_hash,
            resume=resume,
        )
        if claim.claimed:
            claimed.append((item, item_hash))
    timings = []
    with ProcessPoolExecutor(
        max_workers=config.runtime.simulation_workers
    ) as executor:
        futures = {
            executor.submit(
                _run_one_robustness,
                item,
                output,
                config.simulation.steps,
                config.simulation.recycle,
                config.simulation.storage_duration_hours,
                config.simulation.step_seconds,
            ): (item, item_hash)
            for item, item_hash in claimed
        }
        for future in as_completed(futures):
            item, item_hash = futures[future]
            try:
                summary, wall_seconds, peak_rss_mb = future.result()
            except BaseException as error:
                ledger.fail(
                    "robustness",
                    item.config.scenario_id,
                    item_hash,
                    error,
                )
                for pending in futures:
                    pending.cancel()
                raise
            ledger.complete(
                "robustness",
                item.config.scenario_id,
                item_hash,
                artifact_hash=str(summary["artifact_hash"]),
            )
            timings.append(
                {
                    "scenario_id": item.config.scenario_id,
                    "impairment_kind": item.impairment_kind,
                    "logical_steps": int(summary["steps"]),
                    "wall_seconds": wall_seconds,
                    "peak_rss_mb": peak_rss_mb,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    timing_path = output / "timings.csv"
    existing = []
    if timing_path.exists():
        with timing_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    by_id = {
        str(row["scenario_id"]): row for row in [*existing, *timings]
    }
    if by_id:
        temporary = timing_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scenario_id",
                    "impairment_kind",
                    "logical_steps",
                    "wall_seconds",
                    "peak_rss_mb",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [by_id[key] for key in sorted(by_id)]
            )
        temporary.replace(timing_path)


def _audit(config: AppConfig, *, resume: bool, limit: int | None) -> None:
    from gridimpair.data.audit import audit_source_dataset

    audit_source_dataset(config.paths.data_root, config.paths.artifact_root / "audit")


def _select_scenario_matrix(
    config: AppConfig, *, pilot: bool
) -> list[ScenarioConfig]:
    from gridimpair.sim.scenario import (
        generate_main_scenario_matrix,
        generate_pilot_matrix,
        generate_q1_v2_pilot_matrix,
        generate_q1_v2_scenario_matrix,
    )

    protocol = str(
        config.raw.get("protocol", {}).get("name", "legacy")
    )
    seed = config.runtime.seed
    if protocol == "q1_v2":
        return (
            generate_q1_v2_pilot_matrix(base_seed=seed + 7000)
            if pilot
            else generate_q1_v2_scenario_matrix(base_seed=seed)
        )
    if protocol != "legacy":
        raise ValueError("protocol.name must be legacy or q1_v2")
    return (
        generate_pilot_matrix(base_seed=seed + 7000)
        if pilot
        else generate_main_scenario_matrix(base_seed=seed)
    )


def _run_scenarios(
    config: AppConfig,
    *,
    resume: bool,
    limit: int | None,
    pilot: bool,
) -> None:
    scenarios = _select_scenario_matrix(config, pilot=pilot)
    projected_main_runs = len(
        _select_scenario_matrix(config, pilot=False)
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        scenarios = scenarios[:limit]
    namespace = "pilot" if pilot else "simulation"
    output = config.paths.artifact_root / namespace
    ledger = StageLedger(config.paths.artifact_root / "ledgers" / "stages.sqlite")
    timings: list[dict[str, object]] = []
    claimed: list[tuple[ScenarioConfig, str]] = []
    for scenario in scenarios:
        item_hash = stable_hash(asdict(scenario))
        claim = ledger.claim(namespace, scenario.scenario_id, item_hash, resume=resume)
        if claim.claimed:
            claimed.append((scenario, item_hash))
    steps = config.simulation.steps
    with ProcessPoolExecutor(
        max_workers=config.runtime.simulation_workers
    ) as executor:
        futures = {
            executor.submit(
                _run_one_scenario,
                scenario,
                output,
                steps,
                config.simulation.recycle,
                config.simulation.storage_duration_hours,
                config.simulation.step_seconds,
            ): (
                scenario,
                item_hash,
            )
            for scenario, item_hash in claimed
        }
        for future in as_completed(futures):
            scenario, item_hash = futures[future]
            try:
                summary, wall_seconds, peak_rss_mb = future.result()
            except BaseException as error:
                ledger.fail(namespace, scenario.scenario_id, item_hash, error)
                for pending in futures:
                    pending.cancel()
                raise
            ledger.complete(
                namespace,
                scenario.scenario_id,
                item_hash,
                artifact_hash=str(summary["artifact_hash"]),
            )
            timings.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "logical_steps": int(summary["steps"]),
                    "wall_seconds": wall_seconds,
                    "peak_rss_mb": peak_rss_mb,
                }
            )
    timings.sort(key=lambda row: str(row["scenario_id"]))
    timings_path = output / "timings.csv"
    existing_timings: list[dict[str, object]] = []
    if timings_path.exists():
        with timings_path.open(newline="", encoding="utf-8") as handle:
            existing_timings = list(csv.DictReader(handle))
    cumulative_timings = _merge_timing_rows(existing_timings, timings)
    if cumulative_timings:
        temporary_timings = timings_path.with_suffix(".csv.tmp")
        with temporary_timings.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "scenario_id",
                "logical_steps",
                "wall_seconds",
                "peak_rss_mb",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cumulative_timings)
        temporary_timings.replace(timings_path)
    if pilot:
        total_seconds = sum(
            float(row["wall_seconds"]) for row in cumulative_timings
        )
        average = total_seconds / max(1, len(cumulative_timings))
        peak_rss = max(
            (
                float(row["peak_rss_mb"])
                for row in cumulative_timings
            ),
            default=0.0,
        )
        _atomic_json(
            output / "ETA.json",
            {
                "measured_runs": len(cumulative_timings),
                "mean_wall_seconds_per_run": average,
                "projected_main_runs": projected_main_runs,
                "projected_wall_hours": (
                    average
                    * projected_main_runs
                    / 3600
                    / config.runtime.simulation_workers
                ),
                "projected_cpu_hours": (
                    average * projected_main_runs / 3600
                ),
                "maximum_worker_peak_rss_mb": peak_rss,
                "simulation_workers": config.runtime.simulation_workers,
                "logical_time_is_not_wall_clock": True,
            },
        )
        _atomic_json(output / "PILOT_GATE.json", _pilot_gate(output, scenarios))
    else:
        locked_records = []
        for public_path in sorted(output.glob("*/public_manifest.json")):
            public = json.loads(public_path.read_text(encoding="utf-8"))
            if public.get("split") != "locked_test":
                continue
            restricted_path = output / "restricted" / f"{public['scenario_id']}.json"
            if not restricted_path.exists():
                raise FileNotFoundError(restricted_path)
            restricted = json.loads(restricted_path.read_text(encoding="utf-8"))
            locked_records.append(
                {
                    "scenario_id": public["scenario_id"],
                    "pair_id": public["pair_id"],
                    "status": restricted["status"],
                    "communication_intervals": restricted[
                        "communication_intervals"
                    ],
                }
            )
        locked_manifest = {"records": locked_records}
        locked_dir = config.paths.artifact_root / "locked"
        _atomic_json(locked_dir / "manifest.json", locked_manifest)
        _atomic_json(
            locked_dir / "manifest_hash.json",
            {
                "manifest_hash": stable_hash(locked_manifest),
                "record_count": len(locked_records),
            },
        )
        _run_robustness_scenarios(
            config, resume=resume, limit=limit
        )


def _train(config: AppConfig, *, resume: bool, limit: int | None) -> None:
    from gridimpair.workflow import run_training_workflow

    run_training_workflow(config, resume=resume, limit=limit)


def _evaluate(config: AppConfig, *, resume: bool, limit: int | None) -> None:
    from gridimpair.workflow import run_evaluation_workflow

    run_evaluation_workflow(config, resume=resume, limit=limit)


def _figures(config: AppConfig, *, resume: bool, limit: int | None) -> None:
    from gridimpair.reporting.figures import render_figure_manifest
    from gridimpair.reporting.result_tables import prepare_result_tables
    from gridimpair.reporting.sources import prepare_figure_sources

    manifest_value = config.raw.get("reporting", {}).get(
        "figure_manifest", "figure_manifest.json"
    )
    manifest = Path(manifest_value)
    if not manifest.is_absolute():
        manifest = (config.source_path.parent / manifest).resolve()
    figure_dir = config.paths.artifact_root / "figures"
    generated_manifest = figure_dir / "render_manifest.json"
    prepare_figure_sources(
        artifact_root=config.paths.artifact_root,
        configured_manifest=manifest,
        source_dir=config.paths.artifact_root / "figure_sources",
        generated_manifest=generated_manifest,
        smoke=limit is not None,
    )
    figure_outputs = render_figure_manifest(generated_manifest, figure_dir)
    table_outputs = prepare_result_tables(
        config.paths.artifact_root,
        config.paths.artifact_root / "tables",
    )
    reporting_root = Path(__file__).resolve().parent / "reporting"
    reporting_implementation = [
        {
            "path": path.relative_to(reporting_root).as_posix(),
            "content": path.read_text(encoding="utf-8"),
        }
        for path in sorted(reporting_root.rglob("*.py"))
    ]
    status_path = config.paths.artifact_root / "figure_sources" / "figure_status.csv"
    with status_path.open(encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))
    _atomic_json(
        figure_dir / "stage_manifest.json",
        {
            "status": "completed",
            "figure_count": len(figure_outputs),
            "table_count": len(table_outputs),
            "ready_count": sum(row["status"] == "ready" for row in status_rows),
            "missing_count": sum(row["status"] == "missing" for row in status_rows),
            "reporting_implementation_hash": stable_hash(
                reporting_implementation
            ),
            "configured_manifest_hash": stable_hash(
                json.loads(manifest.read_text(encoding="utf-8"))
            ),
            "render_manifest_hash": stable_hash(
                json.loads(generated_manifest.read_text(encoding="utf-8"))
            ),
            "post_locked_reporting_only": True,
        },
    )


HANDLERS: dict[str, Callable[..., None]] = {
    "audit": _audit,
    "pilot": lambda config, **kwargs: _run_scenarios(config, pilot=True, **kwargs),
    "simulate": lambda config, **kwargs: _run_scenarios(config, pilot=False, **kwargs),
    "train": _train,
    "evaluate": _evaluate,
    "figures": _figures,
}


def run(stage: str, config: AppConfig, *, resume: bool, limit: int | None) -> None:
    selected = (
        ("audit", "pilot", "simulate", "train", "evaluate", "figures")
        if stage == "all"
        else (stage,)
    )
    for name in selected:
        HANDLERS[name](config, resume=resume, limit=limit)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        run(
            arguments.stage,
            config,
            resume=arguments.resume,
            limit=arguments.limit,
        )
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as error:
        print(f"gridimpair: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
