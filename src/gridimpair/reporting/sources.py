from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


FrameBuilder = Callable[[Path], pd.DataFrame | None]


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _data_quality(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "data_quality.csv")
    if frame is None:
        return None
    issues = [
        column
        for column in ("nan_count", "inf_count", "duplicate_rows", "irregular_time_grid")
        if column in frame
    ]
    return pd.DataFrame(
        {"file_id": frame["file_id"], "issue_count": frame[issues].sum(axis=1)}
    )


def _leakage(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "leakage_diagnostics.csv")
    if frame is None:
        return None
    return frame[["split_strategy", "auprc"]].rename(columns={"auprc": "score"})


def _scenario_matrix(root: Path) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    for summary_path in sorted((root / "simulation").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({"condition": summary["status"], "scenario_id": summary["scenario_id"]})
    if not rows:
        return None
    return (
        pd.DataFrame(rows)
        .groupby("condition", as_index=False)["scenario_id"]
        .count()
        .rename(columns={"scenario_id": "run_count"})
    )


def _causal_timeline(root: Path) -> pd.DataFrame | None:
    ledgers = sorted((root / "simulation").glob("*/event_ledger.json"))
    if not ledgers:
        return None
    events = json.loads(ledgers[0].read_text(encoding="utf-8"))[:40]
    rows = []
    for sequence, event in enumerate(events):
        rows.extend(
            [
                {
                    "sequence": sequence,
                    "logical_time": event["production_time"],
                    "lane": 0,
                    "event": "produced",
                },
                {
                    "sequence": sequence,
                    "logical_time": event["intended_delivery_time"],
                    "lane": 1,
                    "event": "intended",
                },
            ]
        )
        if event.get("actual_delivery_time") is not None:
            rows.append(
                {
                    "sequence": sequence,
                    "logical_time": event["actual_delivery_time"],
                    "lane": 2,
                    "event": "delivered",
                }
            )
    return pd.DataFrame(rows)


def _open_metric(root: Path, y: str, *, baselines_only: bool = False) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "locked_metrics.csv")
    if frame is None:
        frame = _read_csv(root / "results" / "open_metrics.csv")
        if frame is not None and "split" in frame:
            preferred = frame.loc[frame["split"] == "validation_a"]
            if not preferred.empty:
                frame = preferred
    if frame is None or y not in frame:
        return None
    if baselines_only:
        neural_prefixes = ("candidate", "tcn", "tail_mil", "mil_")
        frame = frame.loc[
            ~frame["model"].str.startswith(neural_prefixes)
        ]
    result = frame[["model", y]].dropna()
    return _label_models(result) if baselines_only else result


MODEL_LABELS = {
    "candidate": "Proposed-H",
    "candidate_physical": "Proposed-P",
    "message_age_watchdog": "AoI watchdog",
    "ridge_residual": "AR residual",
    "flatline": "Flatline",
    "derivative": "Derivative",
    "ewma_cusum": "EWMA-CUSUM",
    "tcn_physical": "TCN-P",
    "tcn_cyber": "TCN-C",
    "tcn_hybrid": "TCN-H",
    "xgboost_physical": "XGB-P",
    "xgboost_cyber": "XGB-C",
    "xgboost_hybrid": "XGB-H",
    "multirocket_hydra_physical": "MR-Hydra-P",
    "multirocket_hydra_cyber": "MR-Hydra-C",
    "multirocket_hydra_hybrid": "MR-Hydra-H",
}

MAIN_PROBABILITY_MODELS = {
    "candidate",
    "candidate_physical",
    "message_age_watchdog",
    "tcn_physical",
    "tcn_cyber",
    "tcn_hybrid",
    "xgboost_physical",
    "xgboost_cyber",
    "xgboost_hybrid",
}

POLICY_LABELS = {
    "candidate": "Proposed-H",
    "tcn_hybrid": "TCN-H",
}


def _label_models(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model"] = result["model"].replace(MODEL_LABELS)
    return result


def _main_open_metric(root: Path, metric: str) -> pd.DataFrame | None:
    if metric == "macro_f1":
        predictions = _read_csv(root / "results" / "predictions_locked.csv")
        if predictions is not None:
            rows: list[dict[str, object]] = []
            rng = np.random.default_rng(2026)
            for model, group in predictions.loc[
                predictions["model"].isin(MODEL_LABELS)
            ].groupby("model", sort=True):
                pair_counts = []
                for _, pair in group.groupby("pair_id", sort=True):
                    truth = pair["label"].to_numpy(int)
                    predicted = (pair["score"].to_numpy(float) >= 0.5).astype(int)
                    pair_counts.append(
                        [
                            np.sum((truth == 1) & (predicted == 1)),
                            np.sum((truth == 0) & (predicted == 0)),
                            np.sum((truth == 0) & (predicted == 1)),
                            np.sum((truth == 1) & (predicted == 0)),
                        ]
                    )
                if not pair_counts:
                    continue
                counts = np.asarray(pair_counts, dtype=float)
                observed = f1_score(
                    group["label"],
                    (group["score"] >= 0.5).astype(int),
                    average="macro",
                    zero_division=0,
                )
                chosen = rng.integers(
                    0, len(counts), size=(2_000, len(counts))
                )
                totals = counts[chosen].sum(axis=1)
                tp, tn, fp, fn = totals.T
                positive = np.divide(
                    2 * tp,
                    2 * tp + fp + fn,
                    out=np.zeros_like(tp),
                    where=(2 * tp + fp + fn) != 0,
                )
                negative = np.divide(
                    2 * tn,
                    2 * tn + fp + fn,
                    out=np.zeros_like(tn),
                    where=(2 * tn + fp + fn) != 0,
                )
                samples = (positive + negative) / 2
                lower, upper = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "model": model,
                        metric: float(observed),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
            if rows:
                return _label_models(pd.DataFrame(rows))
    frame = _open_metric(root, metric)
    if frame is None:
        return None
    selected = frame.loc[frame["model"].isin(MODEL_LABELS)].copy()
    return _label_models(selected) if not selected.empty else None


def _calibration(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "predictions_locked.csv")
    if frame is None:
        frame = _read_csv(root / "training" / "predictions_open.csv")
    if frame is None or frame.empty:
        return None
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby("model", sort=True):
        bins = pd.cut(
            group["score"],
            bins=np.linspace(0.0, 1.0, 11),
            include_lowest=True,
            duplicates="drop",
        )
        for _, bucket in group.groupby(bins, observed=True):
            rows.append(
                {
                    "model": model,
                    "predicted_probability": float(bucket["score"].mean()),
                    "observed_frequency": float(bucket["label"].mean()),
                    "count": int(len(bucket)),
                }
            )
    if not rows:
        return None
    rows.extend(
        [
            {
                "model": "ideal",
                "predicted_probability": 0.0,
                "observed_frequency": 0.0,
                "count": 0,
            },
            {
                "model": "ideal",
                "predicted_probability": 1.0,
                "observed_frequency": 1.0,
                "count": 0,
            },
        ]
    )
    result = pd.DataFrame(rows)
    selected = result.loc[
        result["model"].isin({*MAIN_PROBABILITY_MODELS, "ideal"})
    ]
    return _label_models(selected)


def _risk_coverage(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "predictions_locked.csv")
    if frame is None:
        frame = _read_csv(root / "training" / "predictions_open.csv")
    if frame is None or frame.empty:
        return None
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby("model", sort=True):
        ordered = group.assign(
            confidence=(group["score"] - 0.5).abs() * 2,
            error=((group["score"] >= 0.5).astype(int) != group["label"]).astype(float),
        ).sort_values("confidence", ascending=False)
        for count in range(1, len(ordered) + 1):
            rows.append(
                {
                    "model": model,
                    "coverage": count / len(ordered),
                    "risk": float(ordered["error"].iloc[:count].mean()),
                }
            )
    result = pd.DataFrame(rows)
    selected = result.loc[result["model"].isin(MAIN_PROBABILITY_MODELS)]
    return _label_models(selected)


def _schema(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "file_fingerprints.csv")
    if frame is None:
        return None
    return (
        frame.groupby("schema_hash", as_index=False)["file_id"]
        .count()
        .rename(columns={"schema_hash": "schema", "file_id": "file_count"})
    )


def _missingness(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "channel_missingness.csv")
    if frame is None:
        return None
    return (
        frame.groupby("channel", as_index=False)["missing_fraction"]
        .mean()
        .sort_values("missing_fraction", ascending=False)
    )


def _variance(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "channel_variance.csv")
    return None if frame is None else frame[["variance"]]


def _univariate(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "univariate_channel_diagnostics.csv")
    if frame is None:
        return None
    return frame[["channel", "grouped_auprc"]].rename(
        columns={"grouped_auprc": "score"}
    )


def _fingerprint_embedding(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "fingerprint_embedding.csv")
    if frame is None:
        return None
    return frame[["component_1", "component_2", "label", "file_id"]]


def _time_intervals(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "audit" / "time_grid_intervals.csv")
    return None if frame is None else frame[["interval"]]


def _profile_regimes(root: Path) -> pd.DataFrame | None:
    del root
    from gridimpair.sim.profiles import make_profile

    rows = []
    for index, regime in enumerate(
        ("balanced", "high_load_low_der", "low_load_high_der")
    ):
        profile = make_profile(regime, 2027 + index * 100_000, 900)
        for logical_time, scale in enumerate(profile.load_scale):
            rows.append(
                {
                    "regime": regime,
                    "logical_time": logical_time,
                    "scale": float(scale),
                }
            )
    return pd.DataFrame(rows)


def _clean_trace(root: Path, column: str, output: str) -> pd.DataFrame | None:
    for summary_path in sorted((root / "simulation").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "clean":
            continue
        run_dir = summary_path.parent
        columns = json.loads(
            (run_dir / "trace_columns.json").read_text(encoding="utf-8")
        )["columns"]
        if column not in columns:
            return None
        trace = np.load(run_dir / "trace.npy")
        values = trace[:, columns.index(column)]
        return pd.DataFrame({"logical_time": np.arange(len(values)), output: values})
    return None


def _event_replay(root: Path) -> pd.DataFrame | None:
    ledgers = sorted((root / "simulation").glob("*/event_ledger.json"))
    if not ledgers:
        return None
    events = json.loads(ledgers[0].read_text(encoding="utf-8"))
    delivered = [event for event in events if not event.get("dropped", False)][:500]
    return pd.DataFrame(
        {
            "sequence": np.arange(len(delivered)),
            "delivery_time": [event["actual_delivery_time"] for event in delivered],
        }
    )


def _training_runs(root: Path) -> pd.DataFrame | None:
    rows: list[pd.DataFrame] = []
    training = root / "training"
    for path in sorted(training.glob("**/metrics.csv")):
        relative = path.relative_to(training).parts
        if len(relative) < 4:
            continue
        phase, model, run_id = relative[:3]
        frame = pd.read_csv(path)
        frame.insert(0, "run_id", run_id)
        frame.insert(0, "model", model)
        frame.insert(0, "run_phase", phase)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else None


def _hpo(root: Path) -> pd.DataFrame | None:
    frame = _training_runs(root)
    if frame is None:
        return None
    frame = frame.loc[frame["run_phase"] == "hpo"].copy()
    if frame.empty:
        return None
    frame["trial_id"] = frame["run_id"].str.extract(r"(\d+)").astype(int)
    return (
        frame.sort_values(["model", "trial_id", "validation_loss"])
        .groupby(["model", "trial_id"], as_index=False)
        .first()[["model", "trial_id", "validation_loss"]]
    )


def _training_curves(root: Path) -> pd.DataFrame | None:
    frame = _training_runs(root)
    if frame is None:
        return None
    frame = frame.loc[frame["run_phase"] == "final"].copy()
    if frame.empty:
        return None
    grouped = frame.groupby(["run_phase", "model", "epoch"], as_index=False)
    result = grouped.agg(
        train_loss=("train_loss", "mean"),
        validation_loss=("validation_loss", "mean"),
        validation_std=("validation_loss", "std"),
        learning_rate=("learning_rate", "mean"),
        seed_count=("run_id", "nunique"),
    )
    result["validation_std"] = result["validation_std"].fillna(0.0)
    result["validation_lower"] = (
        result["validation_loss"] - result["validation_std"]
    )
    result["validation_upper"] = (
        result["validation_loss"] + result["validation_std"]
    )
    return result


def _seed_variance(root: Path) -> pd.DataFrame | None:
    frame = _training_runs(root)
    if frame is None:
        return None
    frame = frame.loc[frame["run_phase"] == "final"].copy()
    if frame.empty:
        return None
    frame["seed"] = frame["run_id"].str.extract(r"(\d+)").astype(int)
    return (
        frame.sort_values(["model", "seed", "validation_loss"])
        .groupby(["model", "seed"], as_index=False)
        .first()[["model", "seed", "validation_loss"]]
    )


def _temporal_table(root: Path, name: str) -> pd.DataFrame | None:
    for namespace in ("locked", "open"):
        filename = {
            "metrics": f"temporal_{namespace}_metrics.csv",
            "per_run": f"temporal_{namespace}_per_run.csv",
            "attention": f"attention_{namespace}_examples.csv",
        }[name]
        frame = _read_csv(root / "results" / filename)
        if frame is not None:
            return frame
    return None


def _temporal_metric(root: Path, metric: str) -> pd.DataFrame | None:
    frame = _temporal_table(root, "metrics")
    if frame is None or metric not in frame:
        return None
    return frame[["model", metric]].dropna()


def _main_temporal_metric(root: Path, metric: str) -> pd.DataFrame | None:
    keep = {
        "candidate",
        "candidate_physical",
        "message_age_watchdog",
        "ridge_residual",
        "flatline",
        "tcn_physical",
        "tcn_cyber",
        "tcn_hybrid",
    }
    per_run = _read_csv(root / "results" / "temporal_locked_per_run.csv")
    if per_run is not None and metric in per_run:
        rows: list[dict[str, object]] = []
        rng = np.random.default_rng(2027)
        for model, group in per_run.loc[
            per_run["model"].isin(keep)
        ].groupby("model", sort=True):
            pair_summaries = np.asarray(
                [
                    [pair[metric].sum(), pair[metric].count()]
                    for _, pair in group.groupby("pair_id", sort=True)
                ],
                dtype=float,
            )
            chosen = rng.integers(
                0,
                len(pair_summaries),
                size=(2_000, len(pair_summaries)),
            )
            totals = pair_summaries[chosen].sum(axis=1)
            samples = totals[:, 0] / totals[:, 1]
            lower, upper = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    metric: float(group[metric].mean()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
        if rows:
            return _label_models(pd.DataFrame(rows))
    frame = _temporal_metric(root, metric)
    if frame is None:
        return None
    selected = frame.loc[frame["model"].isin(keep)].copy()
    return _label_models(selected) if not selected.empty else None


def _severity(
    root: Path, dimension: str, output: str
) -> pd.DataFrame | None:
    frame = _temporal_table(root, "per_run")
    if frame is None or dimension not in frame:
        return None
    impaired = frame.loc[frame["status"] == "impaired"].dropna(
        subset=[dimension, "vus_pr"]
    )
    if impaired.empty:
        return None
    return (
        impaired.groupby(["model", dimension], as_index=False)["vus_pr"]
        .mean()
        .rename(columns={dimension: output, "vus_pr": "score"})
    )


def _attention(root: Path) -> pd.DataFrame | None:
    frame = _temporal_table(root, "attention")
    if frame is None or frame.empty:
        return None
    candidate = frame.loc[frame["model"] == "candidate"]
    if candidate.empty:
        candidate = frame
    scenario = candidate.iloc[0]["scenario_id"]
    return candidate.loc[
        candidate["scenario_id"] == scenario,
        ["logical_time", "attention", "label", "model", "scenario_id"],
    ]


def _detection_delays(root: Path) -> pd.DataFrame | None:
    frame = _temporal_table(root, "per_run")
    if frame is None:
        return None
    return frame.loc[
        (frame["status"] == "impaired") & frame["detection_delay"].notna(),
        ["model", "detection_delay"],
    ].rename(columns={"detection_delay": "delay"})


def _false_alarms(root: Path) -> pd.DataFrame | None:
    frame = _temporal_table(root, "metrics")
    if frame is None:
        return None
    return frame[["model", "false_alarms_per_run"]].dropna()


def _paired_statistics(root: Path) -> pd.DataFrame | None:
    for namespace in ("locked", "open"):
        frame = _read_csv(
            root / "results" / f"paired_statistics_{namespace}.csv"
        )
        if frame is not None:
            result = frame[["baseline", "effect", "ci_lower", "ci_upper"]].copy()
            result["comparison"] = "candidate vs " + result["baseline"].astype(str)
            return result
    return None


def _compute_tradeoff(root: Path) -> pd.DataFrame | None:
    training = _training_runs(root)
    if training is None or "epoch_seconds" not in training:
        return None
    metrics = None
    for namespace in ("locked", "open"):
        metrics = _read_csv(root / "results" / f"{namespace}_metrics.csv")
        if metrics is not None:
            break
    if metrics is None:
        return None
    compute = (
        training.groupby("model", as_index=False)["epoch_seconds"]
        .sum()
        .rename(columns={"epoch_seconds": "seconds"})
    )
    compute["gpu_hours"] = compute["seconds"] / 3600
    return compute.merge(
        metrics[["model", "auprc"]].rename(columns={"auprc": "score"}),
        on="model",
        how="inner",
    )


def _latency(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "inference_latency.csv")
    return None if frame is None else frame[["model", "latency_ms", "device", "batch_size"]]


def _failure_cases(root: Path) -> pd.DataFrame | None:
    frame = _temporal_table(root, "per_run")
    if frame is None:
        return None
    candidate = frame.loc[
        (frame["model"] == "candidate") & (frame["status"] == "impaired")
    ].copy()
    if candidate.empty:
        return None
    candidate["activity"] = (
        candidate["delay_steps"].astype(float) / 3.0
        + candidate["drop_probability"].astype(float)
        + candidate["endpoint_fraction"].astype(float)
    )
    candidate["error"] = 1.0 - candidate["vus_pr"].astype(float)
    return candidate[
        [
            "scenario_id",
            "regime",
            "delay_steps",
            "drop_probability",
            "endpoint_fraction",
            "activity",
            "error",
        ]
    ]


def _closed_loop_endpoint(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "closed_loop_endpoints.csv")
    if frame is None:
        return None
    selected = frame.loc[
        frame["endpoint"] == "voltage_violation_seconds",
        ["policy", "mean_delta"],
    ].rename(columns={"mean_delta": "endpoint_delta"})
    if selected.empty:
        return None
    selected["policy"] = selected["policy"].replace(POLICY_LABELS)
    return selected


def _gap_closure_source(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "closed_loop_gap_closure.csv")
    if frame is None:
        return None
    return frame[["model", "gap_closure", "status"]].dropna(
        subset=["gap_closure"]
    )


def _robustness_source(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "robustness_metrics.csv")
    if frame is None:
        return None
    keep = {
        "candidate",
        "candidate_physical",
        "message_age_watchdog",
        "tcn_physical",
        "tcn_cyber",
        "tcn_hybrid",
    }
    selected = frame.loc[
        frame["model"].isin(keep), ["model", "shift", "vus_pr"]
    ]
    return _label_models(selected).rename(columns={"vus_pr": "score"})


def _observability_frontier(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "locked_metrics.csv")
    if frame is None:
        frame = _read_csv(root / "results" / "open_metrics.csv")
        if frame is not None and "split" in frame:
            preferred = frame.loc[frame["split"] == "validation_a"]
            if not preferred.empty:
                frame = preferred
    if frame is None or "auprc" not in frame:
        return None
    selected = frame.loc[
        frame["model"].str.contains(
            "physical|cyber|hybrid|candidate", regex=True, na=False
        ),
        ["model", "auprc"],
    ].copy()
    selected["tier"] = np.select(
        [
            selected["model"].str.endswith("_physical"),
            selected["model"].str.endswith("_cyber"),
            selected["model"].str.endswith("_hybrid"),
            selected["model"].eq("candidate"),
        ],
        ["physical", "cyber", "hybrid", "hybrid"],
        default="unspecified",
    )
    return _label_models(selected).rename(columns={"auprc": "score"})


def _policy_regret_source(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(root / "results" / "closed_loop_policy_regret.csv")
    if frame is None:
        return None
    endpoint = "voltage_violation_seconds"
    selected = frame.loc[
        frame["endpoint"] == endpoint,
        ["policy", "mean_regret"],
    ].copy()
    selected["policy"] = selected["policy"].replace(POLICY_LABELS)
    return selected.rename(columns={"mean_regret": "regret"})


def _absolute_endpoints_source(root: Path) -> pd.DataFrame | None:
    frame = _read_csv(
        root / "results" / "closed_loop_absolute_endpoints.csv"
    )
    if frame is None:
        return None
    endpoint = "voltage_violation_seconds"
    selected = frame.loc[
        frame["endpoint"] == endpoint,
        ["policy", "status", "value"],
    ]
    if selected.empty:
        return None
    result = (
        selected.groupby(["policy", "status"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "endpoint_value"})
    )
    result["policy"] = result["policy"].replace(POLICY_LABELS)
    return result


BUILDERS: dict[str, FrameBuilder] = {
    "fig01-data-quality": _data_quality,
    "fig02-split-leakage": _leakage,
    "fig03-causal-timeline": _causal_timeline,
    "fig04-scenario-matrix": _scenario_matrix,
    "fig05-detectability-floor": lambda root: _open_metric(
        root, "auprc", baselines_only=True
    ),
    "fig06-open-loop": lambda root: _main_open_metric(root, "macro_f1"),
    "fig07-calibration": _calibration,
    "fig08-risk-coverage": _risk_coverage,
    "fig09-localization": lambda root: _main_temporal_metric(root, "vus_pr"),
    "fig10-closed-loop": _closed_loop_endpoint,
    "fig11-gap-closure": _gap_closure_source,
    "fig12-robustness": _robustness_source,
    "figs01-schema": _schema,
    "figs02-missingness": _missingness,
    "figs03-channel-variance": _variance,
    "figs04-fingerprints": _fingerprint_embedding,
    "figs05-best-channel": _univariate,
    "figs06-time-grid": _time_intervals,
    "figs07-profile-regimes": _profile_regimes,
    "figs08-clean-voltage": lambda root: _clean_trace(root, "vm_max", "vm_pu"),
    "figs09-clean-loading": lambda root: _clean_trace(
        root, "loading_max", "loading_percent"
    ),
    "figs10-event-replay": _event_replay,
    "figs11-delay-severity": lambda root: _severity(
        root, "delay_steps", "delay_steps"
    ),
    "figs12-drop-severity": lambda root: _severity(
        root, "drop_probability", "drop_probability"
    ),
    "figs13-endpoint-fraction": lambda root: _severity(
        root, "endpoint_fraction", "endpoint_fraction"
    ),
    "figs14-regime-heterogeneity": lambda root: _severity(
        root, "regime", "regime"
    ),
    "figs15-seed-variance": _seed_variance,
    "figs16-hpo-response": _hpo,
    "figs17-training-curves": _training_curves,
    "figs18-attention": _attention,
    "figs19-delay-distribution": _detection_delays,
    "figs20-false-alarms": _false_alarms,
    "figs21-bootstrap": _paired_statistics,
    "figs22-compute": _compute_tradeoff,
    "figs23-latency": _latency,
    "figs24-failure-cases": _failure_cases,
    "figs25-observability": _observability_frontier,
    "figs26-policy-regret": _policy_regret_source,
    "figs27-absolute-endpoints": _absolute_endpoints_source,
}


GROUPS = {
    "fig07-calibration": "model",
    "fig08-risk-coverage": "model",
    "fig03-causal-timeline": "event",
    "figs04-fingerprints": "label",
    "figs07-profile-regimes": "regime",
    "figs15-seed-variance": "model",
    "figs16-hpo-response": "model",
    "figs17-training-curves": "model",
    "figs11-delay-severity": "model",
    "figs12-drop-severity": "model",
    "figs13-endpoint-fraction": "model",
    "figs14-regime-heterogeneity": "model",
    "figs19-delay-distribution": "model",
    "fig12-robustness": "model",
    "figs25-observability": "tier",
    "figs27-absolute-endpoints": "status",
}


def prepare_figure_sources(
    *,
    artifact_root: Path,
    configured_manifest: Path,
    source_dir: Path,
    generated_manifest: Path,
    smoke: bool,
) -> pd.DataFrame:
    """Build traceable source tables and a renderable subset without inventing data."""
    artifact_root = Path(artifact_root).resolve()
    configured_manifest = Path(configured_manifest).resolve()
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    configured = json.loads(configured_manifest.read_text(encoding="utf-8"))
    rendered: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for entry in configured["figures"]:
        figure_id = str(entry["id"])
        builder = BUILDERS.get(figure_id)
        frame = builder(artifact_root) if builder else None
        if frame is None or frame.empty:
            statuses.append(
                {
                    "id": figure_id,
                    "section": entry.get("section", ""),
                    "status": "missing",
                    "reason": "required_analysis_not_available",
                }
            )
            continue
        source = source_dir / f"{figure_id}.csv"
        frame.to_csv(source, index=False)
        ready = dict(entry)
        ready["source"] = str(source.resolve())
        if figure_id in GROUPS:
            ready["group"] = GROUPS[figure_id]
        if figure_id == "figs15-seed-variance":
            ready["y"] = "validation_loss"
            ready["y_label"] = "Best validation loss"
        if figure_id == "figs17-training-curves":
            ready["y_lower"] = "validation_lower"
            ready["y_upper"] = "validation_upper"
            ready["x_label"] = "Epoch"
            ready["y_label"] = "Validation loss"
        if figure_id in {"fig06-open-loop", "fig09-localization"} and {
            "ci_lower",
            "ci_upper",
        } <= set(frame.columns):
            ready["y_lower"] = "ci_lower"
            ready["y_upper"] = "ci_upper"
        if figure_id == "figs21-bootstrap":
            ready["y_lower"] = "ci_lower"
            ready["y_upper"] = "ci_upper"
            ready["y_label"] = "Candidate minus baseline margin"
        if figure_id == "fig10-closed-loop":
            ready["y_label"] = "Voltage-violation delta (s)"
        if figure_id == "figs26-policy-regret":
            ready["y_label"] = "Regret versus P0 (s; negative is improvement)"
        if figure_id == "figs27-absolute-endpoints":
            ready["y_label"] = "Voltage-violation duration (s)"
        if smoke and figure_id.startswith(("fig05", "fig06", "fig07", "fig08")):
            ready["title"] = f"[SMOKE ONLY] {ready['title']}"
            ready["evidence_status"] = "smoke_only"
        rendered.append(ready)
        statuses.append(
            {
                "id": figure_id,
                "section": entry.get("section", ""),
                "status": "smoke_ready" if smoke else "ready",
                "reason": "",
            }
        )
    generated_manifest.parent.mkdir(parents=True, exist_ok=True)
    generated_manifest.write_text(
        json.dumps(
            {"style": configured.get("style", {}), "figures": rendered},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    status_frame = pd.DataFrame(statuses)
    status_frame.to_csv(source_dir / "figure_status.csv", index=False)
    return status_frame
