"""Online (prefix-horizon) evaluation of the frozen run-level detectors.

For each fraction f of the 900-step horizon, every frozen detector scores
only x_1..x_t with t = ceil(f * 900), and its frozen calibration map and the
0.5 decision threshold are applied unchanged. Because every transform is
causal, this is the decision that would have been available at time t.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from gridimpair.evaluation.metrics import bag_metrics
from gridimpair.r1 import frozen

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)


def prefix_predictions(
    inputs: frozen.FrozenInputs,
    detectors: dict[str, frozen.FrozenDetector],
    training_dir: Path,
    fractions: tuple[float, ...] = FRACTIONS,
) -> pd.DataFrame:
    horizon = inputs.raw_values.shape[-1]
    rows = []
    for fraction in fractions:
        steps = max(1, math.ceil(fraction * horizon))
        prepared = frozen.prepare(inputs, training_dir, prefix_steps=steps)
        for name, detector in detectors.items():
            try:
                raw = detector.score(prepared)
            except Exception as error:  # noqa: BLE001 - e.g. transforms needing longer series
                rows.append({"model": name, "fraction": fraction, "steps": steps, "status": f"failed: {type(error).__name__}"})
                continue
            probability = frozen.calibrated(detector, raw)
            for record, p in zip(inputs.records, probability):
                rows.append({
                    "model": name, "fraction": fraction, "steps": steps, "status": "ok",
                    "scenario_id": record.scenario_id, "pair_id": record.pair_id,
                    "label": record.label, "run_status": "impaired" if record.label else "clean",
                    "score": float(p),
                })
    return pd.DataFrame(rows)


def prefix_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = frame[frame["status"] == "ok"]
    for (model, fraction), g in ok.groupby(["model", "fraction"], sort=True):
        y, p = g["label"].to_numpy(int), g["score"].to_numpy(float)
        flagged = p >= 0.5
        rows.append({
            "model": model, "fraction": fraction, "steps": int(g["steps"].iloc[0]),
            **bag_metrics(y, p),
            "true_positive_rate": float(flagged[y == 1].mean()),
            "false_positive_rate": float(flagged[y == 0].mean()),
            "precision_at_alarm": float(y[flagged].mean()) if flagged.any() else np.nan,
        })
    return pd.DataFrame(rows)


def time_to_detection(frame: pd.DataFrame) -> pd.DataFrame:
    """Earliest evaluated horizon at which each impaired run is flagged."""
    ok = frame[(frame["status"] == "ok") & (frame["label"] == 1)]
    rows = []
    for (model, scenario), g in ok.groupby(["model", "scenario_id"], sort=True):
        g = g.sort_values("fraction")
        hit = g[g["score"] >= 0.5]
        rows.append({"model": model, "scenario_id": scenario,
                     "first_fraction": float(hit["fraction"].iloc[0]) if len(hit) else np.nan,
                     "first_step": int(hit["steps"].iloc[0]) if len(hit) else np.nan})
    per_run = pd.DataFrame(rows)
    return per_run.groupby("model", as_index=False).agg(
        detected_by_end=("first_fraction", lambda s: float(s.notna().mean())),
        median_first_step=("first_step", "median"),
        mean_first_step=("first_step", "mean"),
    )
