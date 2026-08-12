from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

from gridimpair.config import AppConfig, TrainingConfig
from gridimpair.contracts import stable_hash
from gridimpair.evaluation.closed_loop import (
    LockedManifestReader,
    hash_unlock_token,
    paired_endpoint_delta,
    policy_regret_table,
    summarize_electrical_trace,
    validate_paired_results,
)
from gridimpair.evaluation.metrics import (
    bag_metrics,
    derive_effective_onsets,
    temporal_metrics,
    three_onset_delays,
)
from gridimpair.evaluation.operating_points import (
    apply_alarm_policy,
    fit_temporal_operating_points,
)
from gridimpair.evaluation.temporal import (
    interval_labels,
    rolling_derivative_energy,
    rolling_flatline_score,
)
from gridimpair.evaluation.statistics import (
    holm_correction,
    paired_cluster_bootstrap,
    sign_flip_test,
)
from gridimpair.models.candidate import CausalARResidualMIL
from gridimpair.models.classical import (
    CausalRidgeResidual,
    MultiRocketHydraDetector,
    XGBoostDetector,
)
from gridimpair.models.heuristics import (
    AgeWatchdog,
    DerivativeEnergyDetector,
    EwmaCusumDetector,
    FlatlineDetector,
)
from gridimpair.models.tail_mil import TailMIL
from gridimpair.models.tcn import StandardCausalTCN
from gridimpair.train.checkpoints import load_checkpoint
from gridimpair.train.engine import LoaderBundle, fit_model
from gridimpair.train.hpo import TrialConfig, make_lhs_trials
from gridimpair.ledger import StageLedger


PHYSICAL_CHANNELS = (
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
)
CYBER_CHANNELS = (
    "message_age_mean",
    "message_age_max",
    "delivery_fraction",
    "stale_rejection_count",
    "telemetry_age",
)
HYBRID_CHANNELS = (*PHYSICAL_CHANNELS, *CYBER_CHANNELS)


def _select_named_channels(
    values: np.ndarray,
    columns: Iterable[str],
    selected: Iterable[str],
) -> np.ndarray:
    available = list(columns)
    requested = tuple(selected)
    if missing := set(requested) - set(available):
        raise ValueError(f"trace is missing preregistered channels: {sorted(missing)}")
    indices = [available.index(name) for name in requested]
    return np.asarray(values)[:, indices, :]


@dataclass(frozen=True)
class TraceRecord:
    scenario_id: str
    pair_id: str
    split: str
    label: int
    trace_path: Path

    def load(self) -> np.ndarray:
        return np.load(self.trace_path, mmap_mode="r")


@dataclass
class ProbabilityCalibrator:
    kind: str
    estimator: Any = None
    constant: float | None = None

    def predict(self, values: np.ndarray) -> np.ndarray:
        scores = np.asarray(values, dtype=float).reshape(-1)
        if self.kind == "constant":
            return np.full(scores.shape, float(self.constant), dtype=float)
        if self.kind == "platt":
            return self.estimator.predict_proba(scores[:, None])[:, 1]
        if self.kind == "isotonic":
            return np.asarray(self.estimator.predict(scores), dtype=float)
        raise RuntimeError(f"unknown calibrator kind {self.kind!r}")


def _execution_training_plan(
    config: TrainingConfig,
    *,
    configured_seeds: Iterable[int],
    configured_trials: int,
    limit: int | None,
) -> tuple[TrainingConfig, tuple[int, ...], int]:
    seeds = tuple(int(seed) for seed in configured_seeds)
    if not seeds:
        raise ValueError("at least one final seed is required")
    if configured_trials < 1:
        raise ValueError("configured_trials must be positive")
    if limit is None:
        return config, seeds, configured_trials
    if limit < 0:
        raise ValueError("limit must be non-negative")
    smoke_epochs = max(1, min(3, limit or 1))
    bounded = replace(
        config,
        max_epochs=smoke_epochs,
        min_epochs=1,
        patience=1,
        checkpoint_every=1,
    )
    return bounded, (seeds[0],), min(configured_trials, limit)


def _training_run_hash(
    *,
    training_config: dict[str, Any],
    input_artifacts: list[dict[str, Any]],
    limit: int | None,
    implementation_hash: str,
) -> str:
    return stable_hash(
        {
            "training_config": training_config,
            "input_artifacts": input_artifacts,
            "limit": limit,
            "implementation_hash": implementation_hash,
        }
    )


def _source_implementation_hash() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_locked_protocol(
    config: AppConfig,
    training_dir: Path,
    result_dir: Path,
    *,
    manifest_hash: str,
    preprocessing: dict[str, Any],
) -> Path:
    registry_path = training_dir / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    checkpoints = sorted(
        {
            Path(seed["checkpoint"]).resolve()
            for entry in registry
            for seed in entry.get("seeds", [])
        },
        key=str,
    )
    operating_path = result_dir / "temporal_operating_points.json"
    freeze = {
        "protocol": config.raw.get("protocol", {}),
        "config_hash": stable_hash(config.raw),
        "implementation_hash": _source_implementation_hash(),
        "split_hash": preprocessing["split_hash"],
        "locked_manifest_hash": manifest_hash,
        "preprocessing_hash": _sha256_file(
            training_dir / "preprocessing.json"
        ),
        "model_registry_hash": _sha256_file(registry_path),
        "calibrators_hash": _sha256_file(
            training_dir / "calibrators.joblib"
        ),
        "checkpoint_hashes": {
            str(path): _sha256_file(path) for path in checkpoints
        },
        "temporal_operating_points_hash": (
            _sha256_file(operating_path) if operating_path.exists() else None
        ),
        "locked_access_count_before_freeze": 0,
    }
    freeze["freeze_hash"] = stable_hash(freeze)
    path = config.paths.artifact_root / "PROTOCOL_FREEZE.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != freeze:
            raise RuntimeError(
                "locked protocol changed after freeze; create a new artifact root"
            )
    else:
        _write_json_atomic(path, freeze)
    return path


def _archive_training_run(run_dir: Path, run_hash: str) -> Path:
    run_dir = Path(run_dir)
    archive_root = run_dir.parent / "training_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    base = archive_root / f"{run_dir.name}-{run_hash[:12]}"
    archive = base
    suffix = 1
    while archive.exists():
        archive = archive_root / f"{base.name}-{suffix}"
        suffix += 1
    run_dir.replace(archive)
    run_dir.mkdir(parents=True, exist_ok=True)
    return archive


def _fit_calibrator(
    scores: np.ndarray, labels: np.ndarray
) -> ProbabilityCalibrator:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    prevalence = float(
        np.clip(labels.mean() if labels.size else 0.5, 1e-5, 1 - 1e-5)
    )
    if np.unique(labels).size < 2 or np.unique(scores).size < 2:
        return ProbabilityCalibrator("constant", constant=prevalence)
    platt_model = LogisticRegression(random_state=2026).fit(
        scores[:, None], labels
    )
    platt = ProbabilityCalibrator("platt", estimator=platt_model)
    isotonic_model = IsotonicRegression(
        y_min=1e-5, y_max=1 - 1e-5, out_of_bounds="clip"
    ).fit(scores, labels)
    isotonic = ProbabilityCalibrator("isotonic", estimator=isotonic_model)
    platt_loss = log_loss(labels, platt.predict(scores), labels=[0, 1])
    isotonic_loss = log_loss(labels, isotonic.predict(scores), labels=[0, 1])
    return isotonic if isotonic_loss < platt_loss else platt


def discover_training_records(simulation_root: Path) -> list[TraceRecord]:
    """Discover non-locked records without opening a locked trace or label file."""

    records: list[TraceRecord] = []
    for manifest_path in sorted(Path(simulation_root).glob("*/public_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["split"] == "locked_test":
            continue
        status = manifest["status"]
        if status not in {"clean", "impaired"}:
            raise ValueError(f"unexpected open status {status!r}")
        trace_path = manifest_path.parent / "trace.npy"
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        records.append(
            TraceRecord(
                scenario_id=manifest["scenario_id"],
                pair_id=manifest["pair_id"],
                split=manifest["split"],
                label=int(status == "impaired"),
                trace_path=trace_path,
            )
        )
    return records


def discover_public_labeled_records(
    root: Path, *, expected_split: str
) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for manifest_path in sorted(Path(root).glob("*/public_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != expected_split:
            continue
        status = manifest.get("status")
        if status not in {"clean", "impaired"}:
            raise ValueError(f"unexpected public status {status!r}")
        records.append(
            TraceRecord(
                scenario_id=str(manifest["scenario_id"]),
                pair_id=str(manifest["pair_id"]),
                split=expected_split,
                label=int(status == "impaired"),
                trace_path=manifest_path.parent / "trace.npy",
            )
        )
    return records


def _discover_locked_records(
    simulation_root: Path, locked_manifest: dict[str, Any]
) -> list[TraceRecord]:
    labels = {
        record["scenario_id"]: int(record["status"] == "impaired")
        for record in locked_manifest.get("records", [])
    }
    records: list[TraceRecord] = []
    for manifest_path in sorted(Path(simulation_root).glob("*/public_manifest.json")):
        public = json.loads(manifest_path.read_text(encoding="utf-8"))
        if public.get("split") != "locked_test":
            continue
        scenario_id = public["scenario_id"]
        if scenario_id not in labels:
            raise ValueError(f"locked label is missing for {scenario_id!r}")
        records.append(
            TraceRecord(
                scenario_id=scenario_id,
                pair_id=public["pair_id"],
                split="locked_test",
                label=labels[scenario_id],
                trace_path=manifest_path.parent / "trace.npy",
            )
        )
    if set(labels) != {record.scenario_id for record in records}:
        raise ValueError("locked manifest and trace cache contain different scenarios")
    return records


def _load_equal_length(
    records: Iterable[TraceRecord],
) -> tuple[list[TraceRecord], np.ndarray, np.ndarray]:
    selected = list(records)
    if not selected:
        raise ValueError("no simulation traces were found")
    arrays = [np.asarray(record.load(), dtype=np.float32).T for record in selected]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        minimum = min(array.shape[-1] for array in arrays)
        arrays = [array[..., :minimum] for array in arrays]
    return selected, np.stack(arrays), np.asarray([record.label for record in selected])


def _trace_columns(records: Iterable[TraceRecord]) -> list[str]:
    observed: list[list[str]] = []
    for record in records:
        path = record.trace_path.parent / "trace_columns.json"
        if not path.exists():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        observed.append([str(column) for column in value["columns"]])
    if not observed or any(columns != observed[0] for columns in observed[1:]):
        raise ValueError("simulation traces do not share one ordered schema")
    return observed[0]


def _normalize(
    values: np.ndarray, development_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, list[float]]]:
    development = values[development_mask]
    mean = development.transpose(1, 0, 2).reshape(values.shape[1], -1).mean(axis=1)
    std = development.transpose(1, 0, 2).reshape(values.shape[1], -1).std(axis=1)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean[None, :, None]) / std[None, :, None]
    return normalized.astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


def _residual_views(
    values: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, CausalRidgeResidual]:
    ridge = CausalRidgeResidual(lags=4, alpha=1.0).fit_many(
        [case.T for case in values[fit_mask]],
        split="development",
    )
    residuals = []
    for case in values:
        residual = ridge.transform(case.T).T
        residuals.append(np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(residuals).astype(np.float32), ridge


def _prepare_feature_tiers(
    raw_values_all: np.ndarray,
    trace_columns: Iterable[str],
    *,
    development_mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, list[float]]],
    dict[str, np.ndarray],
    dict[str, CausalRidgeResidual],
]:
    channel_sets = {
        "physical": PHYSICAL_CHANNELS,
        "cyber": CYBER_CHANNELS,
        "hybrid": HYBRID_CHANNELS,
    }
    development = np.asarray(development_mask, dtype=bool)
    targets = np.asarray(labels, dtype=int)
    if (
        development.shape != targets.shape
        or development.shape != (len(raw_values_all),)
    ):
        raise ValueError("development mask and labels must match cases")
    clean_development = development & (targets == 0)
    if not clean_development.any():
        raise ValueError("feature tiers require clean development cases")
    tiers: dict[str, np.ndarray] = {}
    normalizations: dict[str, dict[str, list[float]]] = {}
    residuals: dict[str, np.ndarray] = {}
    ridges: dict[str, CausalRidgeResidual] = {}
    for tier, channels in channel_sets.items():
        selected = _select_named_channels(
            raw_values_all, trace_columns, channels
        )
        normalized, normalization = _normalize(selected, development)
        residual, ridge = _residual_views(
            normalized, clean_development
        )
        tiers[tier] = normalized
        normalizations[tier] = normalization
        residuals[tier] = residual
        ridges[tier] = ridge
    return tiers, normalizations, residuals, ridges


def _apply_frozen_feature_tiers(
    raw_values_all: np.ndarray,
    trace_columns: Iterable[str],
    preprocessing: dict[str, Any],
    training_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tier_specs = preprocessing.get("tiers")
    if not tier_specs:
        tier_specs = {
            "physical": {
                "channels": preprocessing.get(
                    "selected_channels", list(PHYSICAL_CHANNELS)
                ),
                "normalization": preprocessing["normalization"],
                "ridge_artifact": "frozen_ridge.joblib",
            }
        }
    values_by_tier: dict[str, np.ndarray] = {}
    residuals_by_tier: dict[str, np.ndarray] = {}
    for tier, spec in tier_specs.items():
        raw = _select_named_channels(
            raw_values_all, trace_columns, spec["channels"]
        )
        mean = np.asarray(spec["normalization"]["mean"], dtype=np.float32)
        std = np.asarray(spec["normalization"]["std"], dtype=np.float32)
        values = ((raw - mean[None, :, None]) / std[None, :, None]).astype(
            np.float32
        )
        ridge: CausalRidgeResidual = joblib.load(
            training_dir
            / spec.get("ridge_artifact", f"frozen_ridge_{tier}.joblib")
        )
        residuals = np.stack(
            [
                np.nan_to_num(
                    ridge.transform(case.T).T,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                for case in values
            ]
        ).astype(np.float32)
        values_by_tier[tier] = values
        residuals_by_tier[tier] = residuals
    return values_by_tier, residuals_by_tier


def _family_base_name(family: str) -> str:
    for suffix in ("_physical", "_cyber", "_hybrid"):
        if family.endswith(suffix):
            return family[: -len(suffix)]
    return family


def _family_feature_tier(
    family: str, configured: dict[str, str]
) -> str:
    tier = str(configured.get(family, "physical"))
    if tier not in {"physical", "cyber", "hybrid"}:
        raise ValueError(
            f"feature tier for {family!r} must be physical, cyber, or hybrid"
        )
    return tier


def _model(
    family: str, channels: int, values: dict[str, Any]
) -> torch.nn.Module:
    base_family = _family_base_name(family)
    hidden = int(values.get("hidden", 48))
    dropout = float(values.get("dropout", 0.1))
    if base_family == "tcn":
        return StandardCausalTCN(
            channels,
            hidden,
            dropout=dropout,
            pooling=str(values.get("pooling", "attention")),
        )
    if base_family == "tail_mil":
        return TailMIL(channels, hidden, dropout=dropout)
    if base_family == "candidate":
        return CausalARResidualMIL(channels, hidden, dropout=dropout)
    if base_family.startswith("mil_"):
        return StandardCausalTCN(
            channels,
            hidden,
            dropout=dropout,
            pooling=base_family.removeprefix("mil_"),
        )
    raise ValueError(f"unknown neural family {family!r}")


def _loaders(
    family: str,
    values: np.ndarray,
    residuals: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    split_hash: str,
    batch_size: int,
) -> LoaderBundle:
    tensors = torch.from_numpy(values)
    targets = torch.from_numpy(labels.astype(np.float32))
    if _family_base_name(family) == "candidate":
        dataset = TensorDataset(tensors, torch.from_numpy(residuals), targets)
    else:
        dataset = TensorDataset(tensors, targets)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not validation_indices.size:
        validation_indices = train_indices
    train_subset = torch.utils.data.Subset(dataset, train_indices.tolist())
    validation_subset = torch.utils.data.Subset(
        dataset, validation_indices.tolist()
    )
    generator = torch.Generator().manual_seed(2026)
    return LoaderBundle(
        train=DataLoader(
            train_subset,
            batch_size=min(batch_size, max(1, len(train_subset))),
            shuffle=True,
            generator=generator,
        ),
        validation=DataLoader(
            validation_subset,
            batch_size=min(batch_size, max(1, len(validation_subset))),
            shuffle=False,
        ),
        split_hash=split_hash,
    )


@torch.no_grad()
def _neural_sequence_scores(
    model: torch.nn.Module,
    family: str,
    values: np.ndarray,
    residuals: np.ndarray,
    *,
    device_name: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    model.to(device).eval()
    bag_parts: list[np.ndarray] = []
    instance_parts: list[np.ndarray] = []
    attention_parts: list[np.ndarray] = []
    has_attention = True
    for start in range(0, len(values), batch_size):
        stop = min(len(values), start + batch_size)
        raw = torch.from_numpy(values[start:stop]).to(device)
        if _family_base_name(family) == "candidate":
            output = model(
                raw, torch.from_numpy(residuals[start:stop]).to(device)
            )
        else:
            output = model(raw)
        bag_parts.append(torch.sigmoid(output.bag_logit).cpu().numpy())
        instance_parts.append(
            torch.sigmoid(output.instance_logits).cpu().numpy()
        )
        if output.attention is None:
            has_attention = False
        else:
            attention_parts.append(output.attention.cpu().numpy())
    attention = (
        np.concatenate(attention_parts)
        if has_attention and attention_parts
        else None
    )
    return (
        np.concatenate(bag_parts),
        np.concatenate(instance_parts),
        attention,
    )


def _neural_scores(
    model: torch.nn.Module,
    family: str,
    values: np.ndarray,
    residuals: np.ndarray,
    *,
    device_name: str,
) -> np.ndarray:
    bag, _, _ = _neural_sequence_scores(
        model,
        family,
        values,
        residuals,
        device_name=device_name,
    )
    return bag


def _read_hpo_space(config: AppConfig) -> dict[str, dict[str, Any]]:
    configured = config.raw.get("training", {}).get("hpo_space", "hpo_space.json")
    path = Path(configured)
    if not path.is_absolute():
        path = (config.source_path.parent / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))["space"]


def _baseline_predictions(
    values_by_tier: dict[str, np.ndarray],
    age_values: np.ndarray,
    labels: np.ndarray,
    development_mask: np.ndarray,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    values = values_by_tier["physical"]
    train_series = np.concatenate(
        [case.T for case in values[development_mask]], axis=0
    )
    rows: list[dict[str, Any]] = []
    availability: list[dict[str, str]] = []
    detectors = {
        "flatline": FlatlineDetector().fit(train_series, split="development"),
        "derivative": DerivativeEnergyDetector().fit(
            train_series, split="development"
        ),
        "ewma_cusum": EwmaCusumDetector().fit(train_series, split="development"),
    }
    residuals, ridge = _residual_views(
        values, development_mask & (labels == 0)
    )
    detectors["ridge_residual"] = ridge
    for name, detector in detectors.items():
        joblib.dump(detector, run_dir / f"{name}.joblib")
        for index, case in enumerate(values):
            rows.append(
                {
                    "record_index": index,
                    "model": name,
                    "score": detector.score(case.T),
                }
            )
    watchdog = AgeWatchdog(max_age=2.0).fit(
        np.concatenate(age_values[development_mask]), split="development"
    )
    joblib.dump(watchdog, run_dir / "message_age_watchdog.joblib")
    for index, ages in enumerate(age_values):
        rows.append(
            {
                "record_index": index,
                "model": "message_age_watchdog",
                "score": float(np.max(ages) > watchdog.max_age),
            }
        )
    for tier, tier_values in values_by_tier.items():
        for base_name, detector in (
            ("xgboost", XGBoostDetector()),
            ("multirocket_hydra", MultiRocketHydraDetector()),
        ):
            name = f"{base_name}_{tier}"
            try:
                detector.fit(
                    tier_values[development_mask],
                    labels[development_mask],
                    split="development",
                )
                joblib.dump(detector, run_dir / f"{name}.joblib")
                for index, case in enumerate(tier_values):
                    rows.append(
                        {
                            "record_index": index,
                            "model": name,
                            "score": detector.score(case),
                        }
                    )
                availability.append(
                    {"model": name, "feature_tier": tier, "status": "fitted"}
                )
            except (RuntimeError, ValueError, ImportError) as error:
                availability.append(
                    {
                        "model": name,
                        "feature_tier": tier,
                        "status": f"unavailable: {type(error).__name__}",
                    }
                )
    return rows, availability


def _training_signal_gate(
    predictions: pd.DataFrame, *, minimum_auprc_lift: float
) -> dict[str, Any]:
    preferred = predictions.loc[predictions["split"] == "validation_a"]
    if preferred.empty:
        preferred = predictions.loc[predictions["split"] == "development"]
    rows = []
    for model, group in preferred.groupby("model", sort=True):
        labels = group["label"].to_numpy(int)
        prevalence = float(np.mean(labels))
        auprc = float(
            average_precision_score(
                labels, group["raw_score"].to_numpy(float)
            )
        )
        rows.append(
            {
                "model": model,
                "prevalence": prevalence,
                "raw_auprc": auprc,
                "auprc_lift_over_prevalence": auprc - prevalence,
            }
        )
    by_model = {row["model"]: row for row in rows}
    candidate = by_model.get("candidate")
    passed = bool(
        candidate
        and candidate["auprc_lift_over_prevalence"]
        >= minimum_auprc_lift
    )
    return {
        "decision": "GO" if passed else "NO_GO",
        "selection_split": (
            "validation_a"
            if (predictions["split"] == "validation_a").any()
            else "development"
        ),
        "preregistered_minimum_candidate_auprc_lift": minimum_auprc_lift,
        "candidate_model": "candidate",
        "models": rows,
    }


def run_training_workflow(
    config: AppConfig, *, resume: bool, limit: int | None
) -> None:
    simulation_root = config.paths.artifact_root / "simulation"
    run_dir = config.paths.artifact_root / "training"
    run_dir.mkdir(parents=True, exist_ok=True)
    discovered_records = discover_training_records(simulation_root)
    training_raw = config.raw.get("training", {})
    requested_trials = int(training_raw.get("hpo_trials", 24))
    eta_path = config.paths.artifact_root / "pilot" / "ETA.json"
    eta_reduction = False
    if eta_path.exists():
        eta = json.loads(eta_path.read_text(encoding="utf-8"))
        if float(eta.get("projected_wall_hours", 0.0)) > 48.0:
            requested_trials = min(requested_trials, 16)
            eta_reduction = True
    (
        execution_training_config,
        final_seeds,
        trial_count,
    ) = _execution_training_plan(
        config.training,
        configured_seeds=training_raw.get(
            "final_seeds", [2026, 2027, 2028]
        ),
        configured_trials=requested_trials,
        limit=limit,
    )
    input_artifacts = []
    for record in discovered_records:
        summary_path = record.trace_path.parent / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        input_artifacts.append(
            {
                "scenario_id": record.scenario_id,
                "split": record.split,
                "artifact_hash": summary["artifact_hash"],
            }
        )
    training_run_hash = _training_run_hash(
        training_config={
            **asdict(execution_training_config),
            "hpo_families": training_raw.get(
                "hpo_families", ["tcn", "tail_mil", "candidate"]
            ),
            "fixed_mil_families": training_raw.get(
                "fixed_mil_families",
                ["mil_mean", "mil_topk", "mil_attention"],
            ),
            "hpo_space": _read_hpo_space(config),
            "hpo_trials": trial_count,
            "final_seeds": final_seeds,
            "feature_tier_by_family": training_raw.get(
                "feature_tier_by_family", {}
            ),
            "training_gate": config.raw.get("training_gate", {}),
        },
        input_artifacts=input_artifacts,
        limit=limit,
        implementation_hash=_source_implementation_hash(),
    )
    run_manifest_path = run_dir / "run_manifest.json"
    if resume and run_manifest_path.exists():
        prior_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        required_outputs = (
            run_dir / "predictions_open.csv",
            run_dir / "model_registry.json",
            run_dir / "calibrators.joblib",
        )
        if (
            prior_manifest.get("status") == "completed"
            and prior_manifest.get("training_run_hash")
            == training_run_hash
            and all(path.exists() for path in required_outputs)
        ):
            return
        if prior_manifest.get("status") == "completed":
            _archive_training_run(
                run_dir,
                str(prior_manifest.get("training_run_hash", "unknown")),
            )
            run_manifest_path = run_dir / "run_manifest.json"
    records, raw_values_all, labels = _load_equal_length(
        discovered_records
    )
    trace_columns = _trace_columns(records)
    age_values = raw_values_all[:, trace_columns.index("message_age_max"), :]
    splits = np.asarray([record.split for record in records])
    development = splits == "development"
    validation_a = splits == "validation_a"
    if not development.any():
        raise ValueError("training requires development scenarios")
    (
        values_by_tier,
        normalizations,
        residuals_by_tier,
        ridges,
    ) = _prepare_feature_tiers(
        raw_values_all,
        trace_columns,
        development_mask=development,
        labels=labels,
    )
    split_hash = stable_hash(
        [
            {"scenario_id": record.scenario_id, "split": record.split}
            for record in records
        ]
    )
    (run_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "tiers": {
                    tier: {
                        "channels": list(
                            PHYSICAL_CHANNELS
                            if tier == "physical"
                            else CYBER_CHANNELS
                            if tier == "cyber"
                            else HYBRID_CHANNELS
                        ),
                        "normalization": normalizations[tier],
                        "ridge_artifact": f"frozen_ridge_{tier}.joblib",
                    }
                    for tier in ("physical", "cyber", "hybrid")
                },
                "watchdog_channel": "message_age_max",
                "split_hash": split_hash,
                "record_count": len(records),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for tier, ridge in ridges.items():
        joblib.dump(ridge, run_dir / f"frozen_ridge_{tier}.joblib")
    prediction_rows, availability = _baseline_predictions(
        values_by_tier, age_values, labels, development, run_dir
    )

    hpo_families = list(
        training_raw.get("hpo_families", ["tcn", "tail_mil", "candidate"])
    )
    fixed_families = list(
        training_raw.get("fixed_mil_families", ["mil_mean", "mil_topk", "mil_attention"])
    )
    configured_tiers = {
        str(key): str(value)
        for key, value in training_raw.get(
            "feature_tier_by_family", {}
        ).items()
    }
    space = _read_hpo_space(config)
    trials = make_lhs_trials(space, n=max(1, trial_count), seed=config.runtime.seed)
    trial_manifest = {
        "requested_trials_per_family": requested_trials,
        "effective_trials_per_family": trial_count,
        "seed": config.runtime.seed,
        "reduced_by_48h_gate": eta_reduction,
        "space": space,
        "trials": [asdict(trial) for trial in trials] if trial_count else [],
    }
    trial_manifest["hash"] = stable_hash(trial_manifest)
    (run_dir / "hpo_trials.json").write_text(
        json.dumps(trial_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    registry: list[dict[str, Any]] = []
    if trial_count:
        hpo_started = time.monotonic()
        total_hpo_seconds = float(
            training_raw.get("total_hpo_gpu_hours", 12.0)
        ) * 3600
        family_hpo_seconds = float(
            training_raw.get("family_hpo_gpu_hours", 4.0)
        ) * 3600
        all_families = [(family, True) for family in hpo_families] + [
            (family, False) for family in fixed_families
        ]
        for family, tune in all_families:
            feature_tier = _family_feature_tier(family, configured_tiers)
            values = values_by_tier[feature_tier]
            residuals = residuals_by_tier[feature_tier]
            family_started = time.monotonic()
            family_trials = trials if tune else [
                TrialConfig(
                    0,
                    {
                        "hidden": 48,
                        "dropout": 0.1,
                        "learning_rate": execution_training_config.learning_rate,
                    },
                )
            ]
            outcomes: list[tuple[float, TrialConfig]] = []
            for trial in family_trials[:trial_count]:
                if (
                    time.monotonic() - family_started >= family_hpo_seconds
                    or time.monotonic() - hpo_started >= total_hpo_seconds
                ):
                    break
                torch.manual_seed(config.runtime.seed + trial.trial_id)
                model = _model(family, values.shape[1], trial.values)
                train_config = replace(
                    execution_training_config,
                    learning_rate=float(
                        trial.values.get(
                            "learning_rate",
                            execution_training_config.learning_rate,
                        )
                    ),
                )
                trial_dir = run_dir / "hpo" / family / f"trial-{trial.trial_id:03d}"
                result = fit_model(
                    model,
                    _loaders(
                        family,
                        values,
                        residuals,
                        labels,
                        development,
                        validation_a,
                        split_hash,
                        int(training_raw.get("batch_size", 32)),
                    ),
                    train_config,
                    trial_dir,
                    precision=config.runtime.precision,
                    device_name=config.runtime.device,
                    resume=resume,
                )
                outcomes.append((result.best_metric, trial))
            if not outcomes:
                raise RuntimeError(
                    f"HPO budget expired before family {family!r} produced a trial"
                )
            _, selected = min(outcomes, key=lambda item: item[0])
            seed_scores = []
            seed_entries = []
            for seed in final_seeds:
                torch.manual_seed(int(seed))
                model = _model(family, values.shape[1], selected.values)
                final_config = replace(
                    execution_training_config,
                    learning_rate=float(
                        selected.values.get(
                            "learning_rate",
                            execution_training_config.learning_rate,
                        )
                    ),
                    final_refit=True,
                )
                final_dir = run_dir / "final" / family / f"seed-{seed}"
                fit_model(
                    model,
                    _loaders(
                        family,
                        values,
                        residuals,
                        labels,
                        development,
                        validation_a,
                        split_hash,
                        int(training_raw.get("batch_size", 32)),
                    ),
                    final_config,
                    final_dir,
                    precision=config.runtime.precision,
                    device_name=config.runtime.device,
                    resume=resume,
                )
                load_checkpoint(
                    final_dir / "best.pt",
                    model=model,
                    expected_config_hash=stable_hash(asdict(final_config)),
                    expected_split_hash=split_hash,
                    restore_rng=False,
                )
                seed_scores.append(
                    _neural_scores(
                        model,
                        family,
                        values,
                        residuals,
                        device_name=config.runtime.device,
                    )
                )
                seed_entries.append(
                    {
                        "seed": int(seed),
                        "checkpoint": str((final_dir / "best.pt").resolve()),
                        "config_hash": stable_hash(asdict(final_config)),
                    }
                )
            ensemble_score = np.mean(seed_scores, axis=0)
            for index, score in enumerate(ensemble_score):
                prediction_rows.append(
                    {
                        "record_index": index,
                        "model": family,
                        "score": float(score),
                    }
                )
            registry.append(
                {
                    "family": family,
                    "feature_tier": feature_tier,
                    "selected_trial": asdict(selected),
                    "seeds": seed_entries,
                }
            )

    metadata = pd.DataFrame(
        [
            {
                "record_index": index,
                "scenario_id": record.scenario_id,
                "pair_id": record.pair_id,
                "split": record.split,
                "label": record.label,
            }
            for index, record in enumerate(records)
        ]
    )
    predictions = pd.DataFrame(prediction_rows).merge(
        metadata, on="record_index", validate="many_to_one"
    )
    calibrator_split = (
        "validation_b"
        if (predictions["split"] == "validation_b").any()
        else "validation_a"
        if (predictions["split"] == "validation_a").any()
        else "development"
    )
    calibrators: dict[str, ProbabilityCalibrator] = {}
    predictions = predictions.rename(columns={"score": "raw_score"})
    predictions["score"] = np.nan
    for model_name, group in predictions.groupby("model", sort=True):
        calibration_rows = group[group["split"] == calibrator_split]
        calibrator = _fit_calibrator(
            calibration_rows["raw_score"].to_numpy(float),
            calibration_rows["label"].to_numpy(int),
        )
        calibrators[model_name] = calibrator
        predictions.loc[group.index, "score"] = calibrator.predict(
            group["raw_score"].to_numpy(float)
        )
    joblib.dump(calibrators, run_dir / "calibrators.joblib")
    predictions.to_csv(run_dir / "predictions_open.csv", index=False)
    gate_config = config.raw.get("training_gate", {})
    _write_json_atomic(
        run_dir / "TRAINING_GATE.json",
        _training_signal_gate(
            predictions,
            minimum_auprc_lift=float(
                gate_config.get("minimum_candidate_auprc_lift", 0.02)
            ),
        ),
    )
    pd.DataFrame(availability).to_csv(run_dir / "model_availability.csv", index=False)
    (run_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )
    prediction_hash = hashlib.sha256(
        (run_dir / "predictions_open.csv").read_bytes()
    ).hexdigest()
    _write_json_atomic(
        run_manifest_path,
        {
            "status": "completed",
            "training_run_hash": training_run_hash,
            "implementation_hash": _source_implementation_hash(),
            "input_artifact_count": len(input_artifacts),
            "prediction_hash": prediction_hash,
            "effective_trials_per_family": trial_count,
            "final_seeds": list(final_seeds),
            "max_epochs": execution_training_config.max_epochs,
            "limit": limit,
        },
    )


def _score_frozen_models(
    config: AppConfig,
    records: list[TraceRecord],
    training_dir: Path,
    *,
    latency_output: Path | None = None,
) -> pd.DataFrame:
    selected, raw_values_all, labels = _load_equal_length(records)
    preprocessing = json.loads(
        (training_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    trace_columns = _trace_columns(selected)
    values_by_tier, residuals_by_tier = _apply_frozen_feature_tiers(
        raw_values_all, trace_columns, preprocessing, training_dir
    )
    age_values = raw_values_all[
        :, trace_columns.index(preprocessing.get("watchdog_channel", "message_age_max")), :
    ]
    calibrators: dict[str, ProbabilityCalibrator] = joblib.load(
        training_dir / "calibrators.joblib"
    )
    raw_scores: dict[str, np.ndarray] = {}
    latency_rows: list[dict[str, Any]] = []
    for model_name in calibrators:
        artifact = training_dir / f"{model_name}.joblib"
        if not artifact.exists():
            continue
        detector = joblib.load(artifact)
        started = time.perf_counter()
        if model_name == "message_age_watchdog":
            scores = [
                float(np.max(case) > detector.max_age) for case in age_values
            ]
        elif model_name.startswith(("xgboost_", "multirocket_hydra_")):
            tier = model_name.rsplit("_", 1)[-1]
            scores = [
                detector.score(case) for case in values_by_tier[tier]
            ]
        else:
            scores = [
                detector.score(case.T)
                for case in values_by_tier["physical"]
            ]
        latency_rows.append(
            {
                "model": model_name,
                "latency_ms": (time.perf_counter() - started)
                * 1_000
                / max(1, len(selected)),
                "batch_size": 1,
                "device": "cpu",
            }
        )
        raw_scores[model_name] = np.asarray(scores, dtype=float)
    registry_path = training_dir / "model_registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else []
    )
    for entry in registry:
        family = entry["family"]
        feature_tier = entry.get("feature_tier", "physical")
        values = values_by_tier[feature_tier]
        residuals = residuals_by_tier[feature_tier]
        trial_values = entry["selected_trial"]["values"]
        ensemble = []
        elapsed = 0.0
        for seed_entry in entry["seeds"]:
            model = _model(family, values.shape[1], trial_values)
            load_checkpoint(
                seed_entry["checkpoint"],
                model=model,
                expected_config_hash=seed_entry["config_hash"],
                expected_split_hash=preprocessing["split_hash"],
                restore_rng=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            ensemble.append(
                _neural_scores(
                    model,
                    family,
                    values,
                    residuals,
                    device_name=config.runtime.device,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
        raw_scores[family] = np.mean(ensemble, axis=0)
        latency_rows.append(
            {
                "model": family,
                "latency_ms": elapsed * 1_000 / max(1, len(values)),
                "batch_size": min(64, len(values)),
                "device": (
                    "cuda"
                    if config.runtime.device == "auto" and torch.cuda.is_available()
                    else config.runtime.device
                ),
            }
        )
    rows = []
    for model_name, scores in raw_scores.items():
        probability = calibrators[model_name].predict(scores)
        for index, record in enumerate(selected):
            rows.append(
                {
                    "scenario_id": record.scenario_id,
                    "pair_id": record.pair_id,
                    "split": record.split,
                    "status": "impaired" if record.label else "clean",
                    "label": record.label,
                    "model": model_name,
                    "raw_score": float(scores[index]),
                    "score": float(probability[index]),
                }
            )
    if latency_output is not None:
        Path(latency_output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(latency_rows).to_csv(latency_output, index=False)
    return pd.DataFrame(rows)


def _restricted_intervals(
    simulation_root: Path, scenario_id: str
) -> list[tuple[int, int]]:
    path = simulation_root / "restricted" / f"{scenario_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return [
        (int(start), int(stop))
        for start, stop in value.get("communication_intervals", [])
    ]


def _temporal_frozen_scores(
    config: AppConfig,
    records: list[TraceRecord],
    training_dir: Path,
) -> tuple[
    list[TraceRecord],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    selected, raw_values_all, _ = _load_equal_length(records)
    preprocessing = json.loads(
        (training_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    trace_columns = _trace_columns(selected)
    values_by_tier, residuals_by_tier = _apply_frozen_feature_tiers(
        raw_values_all, trace_columns, preprocessing, training_dir
    )
    values = values_by_tier["physical"]
    residuals = residuals_by_tier["physical"]
    age_values = raw_values_all[
        :, trace_columns.index(preprocessing.get("watchdog_channel", "message_age_max")), :
    ]
    calibrators: dict[str, ProbabilityCalibrator] = joblib.load(
        training_dir / "calibrators.joblib"
    )
    raw: dict[str, np.ndarray] = {}
    if "ridge_residual" in calibrators:
        raw["ridge_residual"] = np.mean(np.abs(residuals), axis=1)
    if "derivative" in calibrators:
        raw["derivative"] = np.stack(
            [rolling_derivative_energy(case.T) for case in values]
        )
    if "flatline" in calibrators:
        raw["flatline"] = np.stack(
            [rolling_flatline_score(case.T) for case in values]
        )
    ewma_path = training_dir / "ewma_cusum.joblib"
    if "ewma_cusum" in calibrators and ewma_path.exists():
        detector: EwmaCusumDetector = joblib.load(ewma_path)
        raw["ewma_cusum"] = np.stack(
            [detector._scores(case.T) for case in values]
        )
    watchdog_path = training_dir / "message_age_watchdog.joblib"
    if "message_age_watchdog" in calibrators and watchdog_path.exists():
        watchdog: AgeWatchdog = joblib.load(watchdog_path)
        raw["message_age_watchdog"] = (
            age_values > watchdog.max_age
        ).astype(float)

    attentions: dict[str, np.ndarray] = {}
    registry = json.loads(
        (training_dir / "model_registry.json").read_text(encoding="utf-8")
    )
    for entry in registry:
        family = entry["family"]
        feature_tier = entry.get("feature_tier", "physical")
        values = values_by_tier[feature_tier]
        residuals = residuals_by_tier[feature_tier]
        trial_values = entry["selected_trial"]["values"]
        instance_ensemble = []
        attention_ensemble = []
        for seed_entry in entry["seeds"]:
            model = _model(family, values.shape[1], trial_values)
            load_checkpoint(
                seed_entry["checkpoint"],
                model=model,
                expected_config_hash=seed_entry["config_hash"],
                expected_split_hash=preprocessing["split_hash"],
                restore_rng=False,
            )
            _, instances, attention = _neural_sequence_scores(
                model,
                family,
                values,
                residuals,
                device_name=config.runtime.device,
            )
            instance_ensemble.append(instances)
            if attention is not None:
                attention_ensemble.append(attention)
        raw[family] = np.mean(instance_ensemble, axis=0)
        if attention_ensemble:
            attentions[family] = np.mean(attention_ensemble, axis=0)

    return selected, raw_values_all, raw, attentions


def _write_temporal_evaluation(
    config: AppConfig,
    records: list[TraceRecord],
    training_dir: Path,
    result_dir: Path,
    *,
    namespace: str,
    simulation_root: Path | None = None,
) -> None:
    selected, raw_values, scores_by_model, attentions = _temporal_frozen_scores(
        config, records, training_dir
    )
    simulation_root = (
        Path(simulation_root)
        if simulation_root is not None
        else config.paths.artifact_root / "simulation"
    )
    columns = _trace_columns(selected)
    operating_path = result_dir / "temporal_operating_points.json"
    if namespace == "open":
        splits = np.asarray([record.split for record in selected])
        labels = np.asarray([record.label for record in selected])
        source_split = next(
            (
                split
                for split in ("validation_b", "validation_a", "development")
                if np.any((splits == split) & (labels == 0))
            ),
            None,
        )
        if source_split is None:
            raise ValueError(
                "temporal operating points require clean open scenarios"
            )
        temporal_config = config.raw.get("temporal", {})
        operating_points = fit_temporal_operating_points(
            scores_by_model,
            clean_validation_mask=(splits == source_split) & (labels == 0),
            target_false_alarms_per_run=float(
                temporal_config.get(
                    "target_false_alarms_per_run", 0.10
                )
            ),
            min_on_steps=int(temporal_config.get("min_on_steps", 3)),
            min_off_steps=int(temporal_config.get("min_off_steps", 3)),
        )
        _write_json_atomic(
            operating_path,
            {
                "status": "frozen_from_clean_open_runs",
                "source_split": source_split,
                "models": operating_points,
                "hash": stable_hash(operating_points),
            },
        )
    else:
        if not operating_path.exists():
            raise FileNotFoundError(
                "open temporal operating points must be frozen before "
                f"{namespace} evaluation"
            )
        operating_points = json.loads(
            operating_path.read_text(encoding="utf-8")
        )["models"]
    rows: list[dict[str, Any]] = []
    score_lookup: dict[tuple[str, str], np.ndarray] = {}
    alarm_lookup: dict[tuple[str, str], np.ndarray] = {}
    label_lookup: dict[str, np.ndarray] = {}
    for index, record in enumerate(selected):
        intervals = _restricted_intervals(simulation_root, record.scenario_id)
        labels = interval_labels(raw_values.shape[-1], intervals)
        label_lookup[record.scenario_id] = labels
        public = json.loads(
            (record.trace_path.parent / "public_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        condition = public.get("condition", {})
        for model, all_scores in scores_by_model.items():
            scores = all_scores[index]
            score_lookup[(record.scenario_id, model)] = scores
            point = operating_points[model]
            alarms = apply_alarm_policy(
                scores,
                threshold=float(point["threshold"]),
                min_on_steps=int(point["min_on_steps"]),
                min_off_steps=int(point["min_off_steps"]),
            )
            alarm_lookup[(record.scenario_id, model)] = alarms
            rows.append(
                {
                    "scenario_id": record.scenario_id,
                    "pair_id": record.pair_id,
                    "split": record.split,
                    "status": "impaired" if record.label else "clean",
                    "model": model,
                    "regime": public.get("regime"),
                    "delay_steps": condition.get("delay_steps"),
                    "drop_probability": condition.get("drop_probability"),
                    "endpoint_fraction": condition.get("endpoint_fraction"),
                    "shift": condition.get("impairment_kind"),
                    "operating_threshold": float(point["threshold"]),
                    "alarm_fraction": float(np.mean(alarms)),
                    **temporal_metrics(
                        labels,
                        scores,
                        threshold=None,
                        alarms=alarms,
                        run_count=1,
                    ),
                }
            )

    record_by_pair_status = {
        (record.pair_id, record.label): (index, record)
        for index, record in enumerate(selected)
    }
    onset_rows: list[dict[str, Any]] = []
    for (pair_id, label), (impaired_index, impaired) in record_by_pair_status.items():
        if label != 1 or (pair_id, 0) not in record_by_pair_status:
            continue
        clean_index, _ = record_by_pair_status[(pair_id, 0)]
        intervals = _restricted_intervals(simulation_root, impaired.scenario_id)
        onset_records = derive_effective_onsets(
            intervals,
            raw_values[clean_index, columns.index("delivered_p_sum")],
            raw_values[impaired_index, columns.index("delivered_p_sum")],
            raw_values[clean_index][
                [columns.index("vm_min"), columns.index("vm_max"), columns.index("loading_max")]
            ].T,
            raw_values[impaired_index][
                [columns.index("vm_min"), columns.index("vm_max"), columns.index("loading_max")]
            ].T,
            command_tolerance=1e-6,
            electrical_tolerance=1e-4,
        )
        for model in scores_by_model:
            onset_rows.append(
                {
                    "scenario_id": impaired.scenario_id,
                    "pair_id": pair_id,
                    "split": impaired.split,
                    "model": model,
                    **three_onset_delays(
                        alarm_lookup[(impaired.scenario_id, model)].astype(
                            float
                        ),
                        threshold=0.5,
                        onset_records=onset_records,
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    onset_frame = pd.DataFrame(onset_rows)
    frame.to_csv(result_dir / f"temporal_{namespace}_per_run.csv", index=False)
    onset_frame.to_csv(
        result_dir / f"onset_delays_{namespace}_per_run.csv", index=False
    )
    numeric = [
        column
        for column in (
            "event_recall",
            "false_alarms_per_run",
            "interval_iou",
            "temporal_auprc",
            "vus_pr",
            "detection_delay",
            "median_detection_delay",
            "p95_detection_delay",
        )
        if column in frame
    ]
    summary = (
        frame.groupby(["model", "split"], as_index=False)[numeric]
        .mean()
        .merge(
            frame.groupby(["model", "split"], as_index=False)
            .size()
            .rename(columns={"size": "run_count"}),
            on=["model", "split"],
        )
    )
    summary.to_csv(result_dir / f"temporal_{namespace}_metrics.csv", index=False)

    attention_rows: list[dict[str, Any]] = []
    for model, all_attention in attentions.items():
        impaired_indices = [
            index for index, record in enumerate(selected) if record.label == 1
        ][:3]
        for index in impaired_indices:
            record = selected[index]
            labels = label_lookup[record.scenario_id]
            for logical_time, attention in enumerate(all_attention[index]):
                attention_rows.append(
                    {
                        "scenario_id": record.scenario_id,
                        "split": record.split,
                        "model": model,
                        "logical_time": logical_time,
                        "attention": float(attention),
                        "label": int(labels[logical_time]),
                    }
                )
    pd.DataFrame(attention_rows).to_csv(
        result_dir / f"attention_{namespace}_examples.csv", index=False
    )
    fitted_models = set(
        joblib.load(training_dir / "calibrators.joblib").keys()
    )
    pd.DataFrame(
        [
            {
                "model": model,
                "temporal_status": (
                    "available"
                    if model in scores_by_model
                    else "bag_only"
                ),
            }
            for model in sorted(fitted_models | set(scores_by_model))
        ]
    ).to_csv(
        result_dir / f"temporal_{namespace}_availability.csv", index=False
    )
    np.savez_compressed(
        result_dir / f"temporal_{namespace}_scores.npz",
        scenario_ids=np.asarray(
            [record.scenario_id for record in selected], dtype="U64"
        ),
        **{
            f"model__{model}": scores.astype(np.float32)
            for model, scores in scores_by_model.items()
        },
    )


def _metric_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, split), group in predictions.groupby(["model", "split"], sort=True):
        scores = group["score"].to_numpy(float)
        if not np.all((0 <= scores) & (scores <= 1)):
            raise ValueError(f"model {model!r} emitted invalid calibrated probabilities")
        rows.append(
            {
                "model": model,
                "split": split,
                **bag_metrics(group["label"].to_numpy(int), scores),
            }
        )
    return rows


def _write_paired_statistics(
    predictions: pd.DataFrame,
    simulation_root: Path,
    result_dir: Path,
    *,
    namespace: str,
) -> None:
    predictions = predictions.copy()
    if "status" not in predictions:
        predictions["status"] = np.where(
            predictions["label"].astype(int) == 1, "impaired", "clean"
        )
    if "candidate" not in set(predictions["model"]):
        return
    metadata: dict[str, str] = {}
    for pair_id, group in predictions.groupby("pair_id", sort=False):
        scenario_id = str(group.iloc[0]["scenario_id"])
        public = json.loads(
            (
                simulation_root / scenario_id / "public_manifest.json"
            ).read_text(encoding="utf-8")
        )
        condition = public.get("condition", {})
        metadata[str(pair_id)] = "|".join(
            [
                str(public.get("regime", "unknown")),
                str(condition.get("delay_steps", "na")),
                str(condition.get("drop_probability", "na")),
                str(condition.get("endpoint_fraction", "na")),
            ]
        )
    margins: dict[str, pd.Series] = {}
    for model, group in predictions.groupby("model", sort=True):
        pivot = group.pivot(index="pair_id", columns="status", values="score")
        if {"clean", "impaired"} <= set(pivot):
            margins[model] = pivot["impaired"] - pivot["clean"]
    candidate = margins["candidate"]
    pending: list[tuple[str, Any, Any]] = []
    p_values: dict[str, float] = {}
    for model, baseline in margins.items():
        if model == "candidate":
            continue
        common = candidate.index.intersection(baseline.index)
        comparison = pd.DataFrame(
            {
                "pair_id": common,
                "candidate_margin": candidate.loc[common].to_numpy(float),
                "baseline_margin": baseline.loc[common].to_numpy(float),
                "stratum": [metadata[str(pair)] for pair in common],
            }
        )
        bootstrap = paired_cluster_bootstrap(
            comparison,
            candidate_column="candidate_margin",
            baseline_column="baseline_margin",
            n_resamples=10_000 if len(comparison) >= 20 else 1_000,
        )
        permutation = sign_flip_test(
            comparison["candidate_margin"] - comparison["baseline_margin"],
            n_permutations=100_000 if len(comparison) >= 20 else 1_000,
        )
        pending.append((model, bootstrap, permutation))
        p_values[model] = permutation.p_value
    adjusted = holm_correction(p_values)
    rows = [
        {
            "candidate": "candidate",
            "baseline": model,
            "endpoint": "paired_discrimination_margin",
            "effect": bootstrap.observed,
            "ci_lower": bootstrap.lower,
            "ci_upper": bootstrap.upper,
            "confidence": bootstrap.confidence,
            "n_pairs": bootstrap.n_pairs,
            "sign_flip_p": permutation.p_value,
            "holm_p": adjusted[model],
            "permutation_status": permutation.status,
        }
        for model, bootstrap, permutation in pending
    ]
    pd.DataFrame(rows).to_csv(
        result_dir / f"paired_statistics_{namespace}.csv", index=False
    )


def _evaluate_robustness(
    config: AppConfig,
    training_dir: Path,
    result_dir: Path,
    *,
    resume: bool,
) -> None:
    if not bool(config.raw.get("robustness", {}).get("enabled", True)):
        return
    robustness_root = config.paths.artifact_root / "robustness"
    records = discover_public_labeled_records(
        robustness_root, expected_split="robustness"
    )
    if not records:
        return
    prediction_path = result_dir / "predictions_robustness.csv"
    if resume and prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
    else:
        predictions = _score_frozen_models(config, records, training_dir)
        validate_paired_results(predictions)
        predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(_metric_rows(predictions)).to_csv(
        result_dir / "robustness_bag_metrics.csv", index=False
    )
    temporal_path = result_dir / "temporal_robustness_per_run.csv"
    if not (resume and temporal_path.exists()):
        _write_temporal_evaluation(
            config,
            records,
            training_dir,
            result_dir,
            namespace="robustness",
            simulation_root=robustness_root,
        )
    temporal = pd.read_csv(temporal_path)
    impaired = temporal.loc[temporal["status"] == "impaired"]
    clean = temporal.loc[temporal["status"] == "clean"]
    detected = (
        impaired.groupby(["model", "shift"], as_index=False)[
            ["vus_pr", "event_recall", "detection_delay", "interval_iou"]
        ]
        .mean()
    )
    false_alarm = (
        clean.groupby(["model", "shift"], as_index=False)[
            "false_alarms_per_run"
        ]
        .mean()
    )
    detected.merge(
        false_alarm, on=["model", "shift"], how="left"
    ).to_csv(result_dir / "robustness_metrics.csv", index=False)


def _closed_loop_worker(
    scenario: Any,
    runtime_policy: str,
    frozen_alarm_schedule: tuple[bool, ...] | None,
    output_root: str,
    *,
    steps: int,
    recycle: bool,
    storage_duration_hours: float,
    step_seconds: float,
) -> dict[str, Any]:
    from gridimpair.sim.scenario import run_scenario

    summary = run_scenario(
        scenario,
        runtime_policy,
        Path(output_root),
        steps=steps,
        recycle=recycle,
        storage_duration_hours=storage_duration_hours,
        step_seconds=step_seconds,
        frozen_alarm_schedule=frozen_alarm_schedule,
        write_event_ledger=False,
    )
    return asdict(summary)


def _trace_frame(trace_path: Path) -> pd.DataFrame:
    trace_path = Path(trace_path)
    columns = json.loads(
        (trace_path.parent / "trace_columns.json").read_text(encoding="utf-8")
    )["columns"]
    return pd.DataFrame(np.asarray(np.load(trace_path)), columns=columns)


def _run_closed_loop(
    config: AppConfig,
    result_dir: Path,
    *,
    resume: bool,
) -> None:
    closed_config = config.raw.get("closed_loop", {})
    if not bool(closed_config.get("enabled", True)):
        return
    score_path = result_dir / "temporal_locked_scores.npz"
    open_metrics_path = result_dir / "temporal_open_metrics.csv"
    if not score_path.exists() or not open_metrics_path.exists():
        return
    from gridimpair.sim.scenario import (
        generate_main_scenario_matrix,
        generate_q1_v2_scenario_matrix,
    )

    score_bundle = np.load(score_path)
    operating_points = json.loads(
        (result_dir / "temporal_operating_points.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    scenario_ids = [str(value) for value in score_bundle["scenario_ids"]]
    index_by_id = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    available_models = {
        key.removeprefix("model__")
        for key in score_bundle.files
        if key.startswith("model__")
    }
    if "candidate" not in available_models:
        raise ValueError("closed-loop evaluation requires candidate temporal scores")
    validation = pd.read_csv(open_metrics_path)
    validation = validation.loc[
        (validation["model"] != "candidate")
        & validation["model"].isin(available_models)
    ]
    preferred = validation.loc[validation["split"] == "validation_a"]
    if not preferred.empty:
        validation = preferred
    if validation.empty:
        raise ValueError("no temporal baseline is available for closed-loop selection")
    strongest = str(
        validation.sort_values("vus_pr", ascending=False).iloc[0]["model"]
    )
    policies = ("P0", "P1", "P2", "P3", strongest, "candidate")
    policy_runtime = {
        "P0": "P0",
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        strongest: "baseline",
        "candidate": "candidate",
    }
    protocol = str(config.raw.get("protocol", {}).get("name", "legacy"))
    scenario_matrix = (
        generate_q1_v2_scenario_matrix(base_seed=config.runtime.seed)
        if protocol == "q1_v2"
        else generate_main_scenario_matrix(base_seed=config.runtime.seed)
    )
    scenario_map = {
        scenario.scenario_id: scenario
        for scenario in scenario_matrix
        if scenario.split == "locked_test"
        and scenario.scenario_id in index_by_id
    }
    if set(scenario_ids) != set(scenario_map):
        raise ValueError("locked temporal scores and scenario matrix disagree")
    closed_root = config.paths.artifact_root / "closed_loop"
    ledger = StageLedger(
        config.paths.artifact_root / "ledgers" / "stages.sqlite"
    )
    path_map: dict[tuple[str, str], Path] = {}
    work: list[
        tuple[Any, str, str, tuple[bool, ...] | None, str]
    ] = []
    for policy in policies:
        slug = policy.replace("/", "_").replace(":", "_")
        for scenario_id in scenario_ids:
            scenario = scenario_map[scenario_id]
            schedule = None
            if policy in {strongest, "candidate"}:
                point = operating_points[policy]
                schedule = tuple(
                    apply_alarm_policy(
                        score_bundle[f"model__{policy}"][
                            index_by_id[scenario_id]
                        ],
                        threshold=float(point["threshold"]),
                        min_on_steps=int(point["min_on_steps"]),
                        min_off_steps=int(point["min_off_steps"]),
                    ).tolist()
                )
            reuse_primary = (
                policy == "P0"
                or (
                    scenario.status == "clean"
                    and policy in {"P2", "P3"}
                )
                or (
                    policy in {strongest, "candidate"}
                    and schedule is not None
                    and not any(schedule)
                )
            )
            if reuse_primary:
                path_map[(policy, scenario_id)] = (
                    config.paths.artifact_root
                    / "simulation"
                    / scenario_id
                    / "trace.npy"
                )
                continue
            output_root = closed_root / slug
            trace_path = output_root / scenario_id / "trace.npy"
            item_hash = stable_hash(
                {
                    "scenario": asdict(scenario),
                    "policy": policy,
                    "schedule_hash": stable_hash(schedule)
                    if schedule is not None
                    else None,
                }
            )
            claim = ledger.claim(
                f"closed_loop_{slug}",
                scenario_id,
                item_hash,
                resume=resume,
            )
            path_map[(policy, scenario_id)] = trace_path
            if claim.claimed:
                work.append(
                    (
                        scenario,
                        policy,
                        policy_runtime[policy],
                        schedule,
                        item_hash,
                    )
                )
            elif not trace_path.exists():
                raise FileNotFoundError(trace_path)
    futures = {}
    with ProcessPoolExecutor(
        max_workers=config.runtime.simulation_workers
    ) as executor:
        for scenario, policy, runtime_policy, schedule, item_hash in work:
            slug = policy.replace("/", "_").replace(":", "_")
            future = executor.submit(
                _closed_loop_worker,
                scenario,
                runtime_policy,
                schedule,
                str((closed_root / slug).resolve()),
                steps=config.simulation.steps,
                recycle=config.simulation.recycle,
                storage_duration_hours=config.simulation.storage_duration_hours,
                step_seconds=config.simulation.step_seconds,
            )
            futures[future] = (scenario, policy, item_hash)
        for future in as_completed(futures):
            scenario, policy, item_hash = futures[future]
            slug = policy.replace("/", "_").replace(":", "_")
            try:
                summary = future.result()
            except BaseException as error:
                ledger.fail(
                    f"closed_loop_{slug}",
                    scenario.scenario_id,
                    item_hash,
                    error,
                )
                for pending in futures:
                    pending.cancel()
                raise
            ledger.complete(
                f"closed_loop_{slug}",
                scenario.scenario_id,
                item_hash,
                artifact_hash=str(summary["artifact_hash"]),
            )

    paired_rows: list[dict[str, Any]] = []
    absolute_rows: list[dict[str, Any]] = []
    scenarios_by_pair: dict[str, dict[str, Any]] = {}
    for scenario in scenario_map.values():
        scenarios_by_pair.setdefault(scenario.pair_id, {})[
            scenario.status
        ] = scenario
    for policy in policies:
        for pair_id, pair in scenarios_by_pair.items():
            clean = pair["clean"]
            impaired = pair["impaired"]
            clean_frame = _trace_frame(
                path_map[(policy, clean.scenario_id)]
            )
            impaired_frame = _trace_frame(
                path_map[(policy, impaired.scenario_id)]
            )
            for status, frame in (
                ("clean", clean_frame),
                ("impaired", impaired_frame),
            ):
                summary = asdict(
                    summarize_electrical_trace(
                        frame,
                        step_seconds=float(
                            closed_config.get(
                                "step_seconds",
                                config.simulation.step_seconds,
                            )
                        ),
                    )
                )
                for endpoint, value in summary.items():
                    absolute_rows.append(
                        {
                            "pair_id": pair_id,
                            "status": status,
                            "policy": policy,
                            "endpoint": endpoint,
                            "value": value,
                        }
                    )
            delta = paired_endpoint_delta(
                clean_frame,
                impaired_frame,
                step_seconds=float(
                    closed_config.get(
                        "step_seconds", config.simulation.step_seconds
                    )
                ),
            )
            for endpoint, value in delta.items():
                paired_rows.append(
                    {
                        "pair_id": pair_id,
                        "policy": policy,
                        "endpoint": endpoint,
                        "delta": value,
                    }
                )
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(
        result_dir / "closed_loop_per_pair.csv", index=False
    )
    absolute_frame = pd.DataFrame(absolute_rows)
    absolute_frame.to_csv(
        result_dir / "closed_loop_absolute_endpoints.csv",
        index=False,
    )
    regret_frame = policy_regret_table(
        absolute_frame, reference_policy="P0"
    )
    regret_frame.to_csv(
        result_dir / "closed_loop_policy_regret_per_pair.csv",
        index=False,
    )
    (
        regret_frame.groupby(
            ["policy", "status", "endpoint"], as_index=False
        )["regret"]
        .agg(
            mean_regret="mean",
            median_regret="median",
            std_regret="std",
            pair_count="count",
        )
        .to_csv(
            result_dir / "closed_loop_policy_regret.csv",
            index=False,
        )
    )
    endpoint_summary = (
        paired_frame.groupby(["policy", "endpoint"])["delta"]
        .agg(
            mean_delta="mean",
            median_delta="median",
            std_delta="std",
            pair_count="count",
        )
        .reset_index()
    )
    endpoint_summary.to_csv(
        result_dir / "closed_loop_endpoints.csv", index=False
    )
    primary_endpoint = str(
        closed_config.get(
            "primary_endpoint",
            "overload_integral_percent_seconds",
        )
    )
    primary = endpoint_summary.loc[
        endpoint_summary["endpoint"] == primary_endpoint
    ].set_index("policy")["mean_delta"]
    if not {"P0", "P3"}.issubset(primary.index):
        raise ValueError(
            f"closed-loop primary endpoint {primary_endpoint!r} is unavailable"
        )
    denominator = float(primary["P0"] - primary["P3"])
    gap_rows = []
    for model in (strongest, "candidate"):
        if denominator <= 0:
            value, status = np.nan, "no_oracle_headroom"
        else:
            value = float(
                (primary["P0"] - primary[model]) / denominator
            )
            status = "ok"
        gap_rows.append(
            {"model": model, "gap_closure": value, "status": status}
        )
    pd.DataFrame(gap_rows).to_csv(
        result_dir / "closed_loop_gap_closure.csv", index=False
    )
    _write_json_atomic(
        result_dir / "closed_loop_manifest.json",
        {
            "status": "completed",
            "policies": list(policies),
            "strongest_validation_baseline": strongest,
            "pair_count": len(scenarios_by_pair),
            "primary_endpoint": primary_endpoint,
            "on_policy_retraining": False,
            "test_time_adaptation": False,
            "score_source_hash": hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest(),
        },
    )


def _requires_locked_access(
    *, locked_record_count: int, limit: int | None
) -> bool:
    return not (limit is not None and locked_record_count == 0)


def run_evaluation_workflow(
    config: AppConfig, *, resume: bool, limit: int | None
) -> None:
    training_dir = config.paths.artifact_root / "training"
    prediction_path = training_dir / "predictions_open.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("training predictions are required before evaluation")
    predictions = pd.read_csv(prediction_path)
    result_dir = config.paths.artifact_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_metric_rows(predictions)).to_csv(
        result_dir / "open_metrics.csv", index=False
    )
    _write_paired_statistics(
        predictions,
        config.paths.artifact_root / "simulation",
        result_dir,
        namespace="open",
    )
    preprocessing = json.loads(
        (training_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    temporal_enabled = (
        "selected_channels" in preprocessing or "tiers" in preprocessing
    )
    open_temporal_path = result_dir / "temporal_open_metrics.csv"
    if temporal_enabled and (not resume or not open_temporal_path.exists()):
        open_records = discover_training_records(
            config.paths.artifact_root / "simulation"
        )
        if limit is not None:
            open_records = open_records[:limit]
        _write_temporal_evaluation(
            config,
            open_records,
            training_dir,
            result_dir,
            namespace="open",
        )
    locked_predictions_path = result_dir / "predictions_locked.csv"
    if resume and locked_predictions_path.exists():
        locked_predictions = pd.read_csv(locked_predictions_path)
        validate_paired_results(locked_predictions)
        pd.DataFrame(_metric_rows(locked_predictions)).to_csv(
            result_dir / "locked_metrics.csv", index=False
        )
        _write_paired_statistics(
            locked_predictions,
            config.paths.artifact_root / "simulation",
            result_dir,
            namespace="locked",
        )
        locked_temporal_path = result_dir / "temporal_locked_metrics.csv"
        if temporal_enabled and not locked_temporal_path.exists():
            metadata = locked_predictions[
                ["scenario_id", "pair_id", "split", "label"]
            ].drop_duplicates()
            locked_records = [
                TraceRecord(
                    scenario_id=str(row.scenario_id),
                    pair_id=str(row.pair_id),
                    split=str(row.split),
                    label=int(row.label),
                    trace_path=(
                        config.paths.artifact_root
                        / "simulation"
                        / str(row.scenario_id)
                        / "trace.npy"
                    ),
                )
                for row in metadata.itertuples(index=False)
            ]
            _write_temporal_evaluation(
                config,
                locked_records,
                training_dir,
                result_dir,
                namespace="locked",
            )
        if temporal_enabled:
            _run_closed_loop(config, result_dir, resume=resume)
        _evaluate_robustness(
            config, training_dir, result_dir, resume=resume
        )
        return

    locked = config.raw.get("locked", {})
    manifest_value = locked.get("manifest")
    guard_value = locked.get("guard")
    if not manifest_value or not guard_value:
        raise PermissionError("locked manifest and guard must be configured")
    manifest_path = Path(manifest_value)
    guard_path = Path(guard_value)
    if not manifest_path.is_absolute():
        manifest_path = (config.source_path.parent / manifest_path).resolve()
    if not guard_path.is_absolute():
        guard_path = (config.source_path.parent / guard_path).resolve()
    hash_path = manifest_path.parent / "manifest_hash.json"
    if not hash_path.exists():
        raise FileNotFoundError("locked manifest hash receipt is missing")
    hash_receipt = json.loads(hash_path.read_text(encoding="utf-8"))
    if not _requires_locked_access(
        locked_record_count=int(hash_receipt["record_count"]),
        limit=limit,
    ):
        _write_json_atomic(
            result_dir / "evaluation_manifest.json",
            {
                "status": "smoke_open_only",
                "locked_record_count": 0,
                "limit": limit,
                "open_prediction_hash": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
            },
        )
        return
    _freeze_locked_protocol(
        config,
        training_dir,
        result_dir,
        manifest_hash=str(hash_receipt["manifest_hash"]),
        preprocessing=preprocessing,
    )
    token = os.environ.get("GRIDIMPAIR_UNLOCK_TOKEN")
    if not token:
        raise PermissionError(
            "set GRIDIMPAIR_UNLOCK_TOKEN for the one-time locked access"
        )
    config_hash = stable_hash(config.raw)
    split_hash = preprocessing["split_hash"]
    if not guard_path.exists():
        manifest_hash = hash_receipt["manifest_hash"]
        guard_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = guard_path.with_suffix(guard_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "token_hash": hash_unlock_token(token),
                    "config_hash": config_hash,
                    "split_hash": split_hash,
                    "manifest_hash": manifest_hash,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(guard_path)
    reader = LockedManifestReader(manifest_path, guard_path)
    reader.unlock(token, config_hash=config_hash, split_hash=split_hash)
    locked_manifest = reader.read()
    locked_records = _discover_locked_records(
        config.paths.artifact_root / "simulation", locked_manifest
    )
    if limit is not None:
        locked_records = locked_records[:limit]
    locked_predictions = _score_frozen_models(
        config,
        locked_records,
        training_dir,
        latency_output=result_dir / "inference_latency.csv",
    )
    validate_paired_results(locked_predictions)
    locked_predictions.to_csv(locked_predictions_path, index=False)
    pd.DataFrame(_metric_rows(locked_predictions)).to_csv(
        result_dir / "locked_metrics.csv", index=False
    )
    _write_paired_statistics(
        locked_predictions,
        config.paths.artifact_root / "simulation",
        result_dir,
        namespace="locked",
    )
    if temporal_enabled:
        _write_temporal_evaluation(
            config,
            locked_records,
            training_dir,
            result_dir,
            namespace="locked",
        )
        _run_closed_loop(config, result_dir, resume=resume)
    (result_dir / "locked_access_receipt.json").write_text(
        json.dumps(
            {
                "pair_count": len({record.pair_id for record in locked_records}),
                "manifest_hash": stable_hash(locked_manifest),
                "status": "unlocked_once",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _evaluate_robustness(
        config, training_dir, result_dir, resume=resume
    )
