"""Load frozen detectors once and score arbitrary (optionally truncated) traces.

This mirrors ``workflow._score_frozen_models`` but separates loading from
scoring so that the same frozen models can be re-scored on prefixes of a run
(online evaluation) or repeatedly (energy benchmark) without changing them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch

from gridimpair.config import AppConfig
from gridimpair.train.checkpoints import load_checkpoint
from gridimpair.workflow import (
    ProbabilityCalibrator,
    TraceRecord,
    _apply_frozen_feature_tiers,
    _load_equal_length,
    _model,
    _neural_sequence_scores,
    _trace_columns,
)


@dataclass
class FrozenInputs:
    records: list[TraceRecord]
    raw_values: np.ndarray  # (runs, all trace columns, time)
    labels: np.ndarray
    columns: list[str]


@dataclass
class PreparedInputs:
    values_by_tier: dict[str, np.ndarray]
    residuals_by_tier: dict[str, np.ndarray]
    age_values: np.ndarray


@dataclass
class FrozenDetector:
    name: str
    kind: str  # "rule", "classical", or "neural"
    feature_tier: str
    score: Callable[..., np.ndarray]
    calibrator: ProbabilityCalibrator
    parameters: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def locked_records(artifact_root: Path, predictions: pd.DataFrame) -> list[TraceRecord]:
    metadata = predictions[["scenario_id", "pair_id", "split", "label"]].drop_duplicates()
    return [
        TraceRecord(
            scenario_id=str(row.scenario_id),
            pair_id=str(row.pair_id),
            split=str(row.split),
            label=int(row.label),
            trace_path=artifact_root / "simulation" / str(row.scenario_id) / "trace.npy",
        )
        for row in metadata.itertuples(index=False)
    ]


def load_inputs(records: list[TraceRecord]) -> FrozenInputs:
    selected, raw_values, labels = _load_equal_length(records)
    return FrozenInputs(selected, raw_values, labels, _trace_columns(selected))


def prepare(inputs: FrozenInputs, training_dir: Path, prefix_steps: int | None = None) -> PreparedInputs:
    """Apply the frozen, causal feature transforms; optionally keep a prefix.

    Normalization is pointwise and the AR residual is causal, so truncating the
    raw trace before the transform equals truncating the transformed trace.
    """
    raw = inputs.raw_values if prefix_steps is None else inputs.raw_values[..., :prefix_steps]
    preprocessing = json.loads((training_dir / "preprocessing.json").read_text(encoding="utf-8"))
    values, residuals = _apply_frozen_feature_tiers(raw, inputs.columns, preprocessing, training_dir)
    channel = preprocessing.get("watchdog_channel", "message_age_max")
    return PreparedInputs(values, residuals, raw[:, inputs.columns.index(channel), :])


def load_detectors(
    training_dir: Path,
    *,
    include: set[str] | None = None,
) -> dict[str, FrozenDetector]:
    calibrators: dict[str, ProbabilityCalibrator] = joblib.load(training_dir / "calibrators.joblib")
    preprocessing = json.loads((training_dir / "preprocessing.json").read_text(encoding="utf-8"))
    detectors: dict[str, FrozenDetector] = {}
    for name, calibrator in calibrators.items():
        if include is not None and name not in include:
            continue
        artifact = training_dir / f"{name}.joblib"
        if not artifact.exists():
            continue
        detector = joblib.load(artifact)
        if name == "message_age_watchdog":
            def score(prep, _d=detector, **_):
                return np.asarray([float(np.max(c) > _d.max_age) for c in prep.age_values])
            detectors[name] = FrozenDetector(name, "rule", "cyber", score, calibrator)
        elif name.startswith(("xgboost_", "multirocket_hydra_")):
            tier = name.rsplit("_", 1)[-1]
            def score(prep, _d=detector, _t=tier, **_):
                return np.asarray([_d.score(c) for c in prep.values_by_tier[_t]], dtype=float)
            detectors[name] = FrozenDetector(name, "classical", tier, score, calibrator)
        else:
            def score(prep, _d=detector, **_):
                return np.asarray([_d.score(c.T) for c in prep.values_by_tier["physical"]], dtype=float)
            detectors[name] = FrozenDetector(name, "rule", "physical", score, calibrator)
    registry_path = training_dir / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else []
    for entry in registry:
        family = entry["family"]
        if include is not None and family not in include:
            continue
        tier = entry.get("feature_tier", "physical")
        trial_values = entry["selected_trial"]["values"]
        channels = len(next(spec for t, spec in preprocessing["tiers"].items() if t == tier)["channels"]) \
            if "tiers" in preprocessing else len(preprocessing.get("selected_channels", []))
        models = []
        for seed_entry in entry["seeds"]:
            model = _model(family, channels, trial_values)
            load_checkpoint(
                seed_entry["checkpoint"],
                model=model,
                expected_config_hash=seed_entry["config_hash"],
                expected_split_hash=preprocessing["split_hash"],
                restore_rng=False,
            )
            model.eval()
            models.append(model)
        parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)

        def score(prep, _models=models, _family=family, _tier=tier, device="auto", batch_size=64, instances=False):
            bags, steps = [], []
            for m in _models:
                bag, inst, _ = _neural_sequence_scores(
                    m, _family, prep.values_by_tier[_tier], prep.residuals_by_tier[_tier],
                    device_name=device, batch_size=batch_size,
                )
                bags.append(bag)
                steps.append(inst)
            if instances:
                return np.mean(bags, axis=0), np.mean(steps, axis=0)
            return np.mean(bags, axis=0)

        detectors[family] = FrozenDetector(
            family, "neural", tier, score, calibrators[family], parameters,
            {"seeds": len(models), "hidden": trial_values.get("hidden"), "models": models},
        )
    return detectors


def calibrated(detector: FrozenDetector, raw_scores: np.ndarray) -> np.ndarray:
    return np.clip(detector.calibrator.predict(raw_scores), 0.0, 1.0)


def device_cuda_available() -> bool:
    return torch.cuda.is_available()
