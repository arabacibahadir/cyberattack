"""Score frozen detectors under the prespecified metadata/measurement degradations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridimpair.evaluation.metrics import bag_metrics
from gridimpair.r1 import frozen, metadata
from gridimpair.r1.uncertainty import bag_metric_intervals

METADATA_KINDS = {"clock_offset", "clock_drift", "timestamp_dropout", "sequence_corruption", "metadata_loss", "metadata_delay"}
PHYSICAL_KINDS = {"physical_noise", "physical_missing"}


def _affected(detector: frozen.FrozenDetector, perturbation: metadata.Perturbation) -> bool:
    if perturbation.kind == "none":
        return True
    uses_cyber = detector.feature_tier in ("cyber", "hybrid") or detector.name == "message_age_watchdog"
    uses_physical = detector.feature_tier in ("physical", "hybrid") and detector.name != "message_age_watchdog"
    return uses_cyber if perturbation.kind in METADATA_KINDS else uses_physical


def run_degradation(
    inputs: frozen.FrozenInputs,
    detectors: dict[str, frozen.FrozenDetector],
    training_dir: Path,
    simulation_root: Path,
    *,
    grid: tuple[metadata.Perturbation, ...] = metadata.GRID,
    seed: int = 2026,
    ci_models: tuple[str, ...] = (),
    strata: dict[str, str] | None = None,
    n_resamples: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessing = json.loads((training_dir / "preprocessing.json").read_text(encoding="utf-8"))
    physical = preprocessing["tiers"]["physical"]
    physical_channels = list(physical["channels"])
    physical_std = np.asarray(physical["normalization"]["std"], dtype=np.float32)
    ledgers = {r.scenario_id: metadata.load_ledger(simulation_root, r.scenario_id) for r in inputs.records}
    baseline_scores: dict[str, np.ndarray] = {}
    metric_rows, prediction_rows = [], []
    for perturbation in grid:
        if perturbation.kind == "none":
            raw = inputs.raw_values
        else:
            raw = np.stack([
                metadata.perturb_trace(inputs.raw_values[i], inputs.columns, ledgers[r.scenario_id], perturbation,
                                       metadata._rng(r.scenario_id, perturbation, seed), physical_channels, physical_std)
                for i, r in enumerate(inputs.records)
            ])
        prepared = frozen.prepare(frozen.FrozenInputs(inputs.records, raw, inputs.labels, inputs.columns), training_dir)
        for name, detector in detectors.items():
            if perturbation.kind != "none" and not _affected(detector, perturbation):
                probability = baseline_scores[name]
            else:
                probability = frozen.calibrated(detector, detector.score(prepared))
                if perturbation.kind == "none":
                    baseline_scores[name] = probability
            metric_rows.append({"perturbation": perturbation.kind, "level": perturbation.level, "model": name,
                                "affected": perturbation.kind == "none" or _affected(detector, perturbation),
                                **bag_metrics(inputs.labels, probability)})
            if name in ci_models:
                for r, p in zip(inputs.records, probability):
                    prediction_rows.append({"perturbation": perturbation.kind, "level": perturbation.level,
                                            "model": name, "scenario_id": r.scenario_id, "pair_id": r.pair_id,
                                            "label": r.label, "status": "impaired" if r.label else "clean", "score": float(p)})
        print(f"degradation: {perturbation.label} done", flush=True)
    metrics = pd.DataFrame(metric_rows)
    ci = pd.DataFrame()
    if prediction_rows and strata is not None:
        preds = pd.DataFrame(prediction_rows)
        parts = []
        for (kind, level), g in preds.groupby(["perturbation", "level"], sort=False):
            cis, _ = bag_metric_intervals(g, strata, reference="__none__", n=n_resamples)
            parts.append(cis.assign(perturbation=kind, level=level))
        ci = pd.concat(parts, ignore_index=True)
    return metrics, ci
