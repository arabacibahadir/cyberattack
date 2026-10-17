"""Calibration under network shift: reliability tables, ECE, MCE, histograms."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gridimpair.r1.uncertainty import bootstrap_pair_draws


def reliability_table(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(labels, float)
    p = np.clip(np.asarray(probabilities, float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        rows.append({
            "bin": b,
            "lower": edges[b],
            "upper": edges[b + 1],
            "count": int(mask.sum()),
            "mean_confidence": float(p[mask].mean()) if mask.any() else np.nan,
            "observed_frequency": float(y[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_errors(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, float]:
    y = np.asarray(labels, float)
    p = np.clip(np.asarray(probabilities, float), 0.0, 1.0)
    index = np.clip(np.digitize(p, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    used = counts > 0
    confidence = np.bincount(index, weights=p, minlength=bins)[used] / counts[used]
    frequency = np.bincount(index, weights=y, minlength=bins)[used] / counts[used]
    gaps = np.abs(confidence - frequency)
    return {
        "ece": float(np.sum(counts[used] / counts.sum() * gaps)),
        "mce": float(gaps.max()) if gaps.size else np.nan,
    }


def calibration_summary(
    predictions: pd.DataFrame,
    *,
    context: str,
    strata: dict[str, str] | None = None,
    group_column: str | None = None,
    bins: int = 10,
    n: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """ECE/MCE per model (and optional subgroup) with pair-bootstrap CIs."""
    rows = []
    groups = [("all", predictions)] if group_column is None else list(predictions.groupby(group_column))
    for group_name, frame in groups:
        for model, g in frame.groupby("model", sort=True):
            y, p = g["label"].to_numpy(int), g["score"].to_numpy(float)
            errors = calibration_errors(y, p, bins)
            row = {"context": context, "group": str(group_name), "model": model, "runs": len(g), **errors}
            if strata is not None and "pair_id" in g:
                work = g.assign(pair_id=g["pair_id"].astype(str))
                py = work.pivot(index="pair_id", columns="status", values="label")
                pp = work.pivot(index="pair_id", columns="status", values="score")
                if {"clean", "impaired"} <= set(pp) and not pp.isna().any().any():
                    pairs = list(pp.index)
                    ya, pa = py[["clean", "impaired"]].to_numpy(int), pp[["clean", "impaired"]].to_numpy(float)
                    eces = [
                        calibration_errors(ya[d].ravel(), pa[d].ravel(), bins)["ece"]
                        for d in bootstrap_pair_draws(pairs, strata, n, seed)
                    ]
                    row["ece_lower"] = float(np.quantile(eces, 0.025))
                    row["ece_upper"] = float(np.quantile(eces, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def probability_histogram(predictions: pd.DataFrame, *, context: str, bins: int = 20) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for (model, status), g in predictions.groupby(["model", "status"], sort=True):
        counts, _ = np.histogram(np.clip(g["score"].to_numpy(float), 0, 1), bins=edges)
        for b, c in enumerate(counts):
            rows.append({"context": context, "model": model, "status": status, "lower": edges[b], "upper": edges[b + 1], "count": int(c)})
    return pd.DataFrame(rows)
