"""Scenario-pair bootstrap confidence intervals for the main R1 endpoints.

The scenario pair is the resampling unit. Pairs are resampled with
replacement inside strata defined by regime and the complete named impairment
condition (delay, drop, endpoint fraction, and impairment kind), so burst and
jitter settings are no longer pooled with each other.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from gridimpair.evaluation.statistics import holm_correction

BAG_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": lambda y, p: float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
    "auprc": lambda y, p: float(average_precision_score(y, p)),
    "macro_f1": lambda y, p: float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
}


def pair_strata(predictions: pd.DataFrame, simulation_root: Path) -> dict[str, str]:
    strata: dict[str, str] = {}
    for pair_id, group in predictions.groupby("pair_id", sort=False):
        scenario_id = str(group.iloc[0]["scenario_id"])
        path = simulation_root / scenario_id / "public_manifest.json"
        if not path.exists():
            strata[str(pair_id)] = "all"
            continue
        public = json.loads(path.read_text(encoding="utf-8"))
        condition = public.get("condition", {})
        strata[str(pair_id)] = "|".join(
            str(x) for x in (
                public.get("regime", "na"),
                condition.get("delay_steps", "na"),
                condition.get("drop_probability", "na"),
                condition.get("endpoint_fraction", "na"),
                condition.get("impairment_kind", "na"),
            )
        )
    return strata


def bootstrap_pair_draws(pairs: list[str], strata: dict[str, str], n: int, seed: int) -> Iterable[np.ndarray]:
    """Yield index arrays into ``pairs`` for each stratified bootstrap draw."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, pair in enumerate(pairs):
        groups.setdefault(strata.get(pair, "all"), []).append(i)
    members = [np.asarray(v) for _, v in sorted(groups.items())]
    for _ in range(n):
        yield np.concatenate([g[rng.integers(0, len(g), len(g))] for g in members])


def _interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return np.nan, np.nan
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def bag_metric_intervals(
    predictions: pd.DataFrame,
    strata: dict[str, str],
    *,
    reference: str = "candidate",
    n: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model metric CIs and paired reference-minus-model differences."""
    pairs = sorted(predictions["pair_id"].astype(str).unique())
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    predictions = predictions.assign(pair_id=predictions["pair_id"].astype(str))
    for model, group in predictions.groupby("model", sort=True):
        piv_y = group.pivot(index="pair_id", columns="status", values="label").reindex(pairs)
        piv_p = group.pivot(index="pair_id", columns="status", values="score").reindex(pairs)
        if piv_p.isna().any().any():
            continue
        arrays[str(model)] = (piv_y[["clean", "impaired"]].to_numpy(int), piv_p[["clean", "impaired"]].to_numpy(float))
    observed = {
        m: {k: f(y.ravel(), p.ravel()) for k, f in BAG_METRICS.items()} for m, (y, p) in arrays.items()
    }
    samples = {m: {k: [] for k in BAG_METRICS} for m in arrays}
    for draw in bootstrap_pair_draws(pairs, strata, n, seed):
        for m, (y, p) in arrays.items():
            yy, pp = y[draw].ravel(), p[draw].ravel()
            for k, f in BAG_METRICS.items():
                samples[m][k].append(f(yy, pp))
    rows, diff_rows = [], []
    for m in arrays:
        for k in BAG_METRICS:
            s = np.asarray(samples[m][k], dtype=float)
            lo, hi = _interval(s, confidence)
            rows.append({"model": m, "metric": k, "estimate": observed[m][k], "lower": lo, "upper": hi,
                         "confidence": confidence, "n_pairs": len(pairs), "n_resamples": n})
    if reference in arrays:
        p_values: dict[str, float] = {}
        for m in arrays:
            if m == reference:
                continue
            for k in BAG_METRICS:
                d = np.asarray(samples[reference][k], float) - np.asarray(samples[m][k], float)
                d = d[np.isfinite(d)]
                lo, hi = _interval(d, confidence)
                p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))) if d.size else np.nan
                key = f"{m}|{k}"
                p_values[key] = p
                diff_rows.append({"reference": reference, "model": m, "metric": k,
                                  "difference": observed[reference][k] - observed[m][k],
                                  "lower": lo, "upper": hi, "p_bootstrap": p})
        holm = holm_correction({k: v for k, v in p_values.items() if np.isfinite(v)})
        for row in diff_rows:
            row["p_holm"] = holm.get(f"{row['model']}|{row['metric']}", np.nan)
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def mean_intervals(
    frame: pd.DataFrame,
    value_columns: list[str],
    group_columns: list[str],
    strata: dict[str, str],
    *,
    n: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """CIs for group means of per-run or per-pair values, resampling pairs."""
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        per_pair = group.groupby("pair_id")[value_columns].mean()
        pairs = [str(p) for p in per_pair.index]
        values = per_pair.to_numpy(float)
        draws = np.asarray([np.nanmean(values[d], axis=0) for d in bootstrap_pair_draws(pairs, strata, n, seed)])
        for j, column in enumerate(value_columns):
            lo, hi = _interval(draws[:, j], confidence)
            rows.append({**dict(zip(group_columns, key)), "metric": column,
                         "estimate": float(np.nanmean(group[column].to_numpy(float))),
                         "lower": lo, "upper": hi, "n_pairs": len(pairs), "n_resamples": n})
    return pd.DataFrame(rows)


def temporal_intervals(per_run: pd.DataFrame, onsets: pd.DataFrame | None, strata: dict[str, str], **kw) -> pd.DataFrame:
    """Temporal endpoints with explicit denominators.

    * impaired-only: event recall, VUS-PR, temporal AUPRC, IoU, detection delay
    * clean-only: false alarms per run and per hour (900 one-second steps)
    * mixed: the originally reported mixed-status IoU and VUS-PR
    """
    work = per_run.copy()
    work["false_alarms_per_hour"] = work["false_alarms_per_run"] * 3600.0 / 900.0
    impaired = work[work["status"] == "impaired"]
    clean = work[work["status"] == "clean"]
    parts = [
        mean_intervals(impaired.assign(denominator="impaired"),
                       ["event_recall", "vus_pr", "temporal_auprc", "interval_iou", "detection_delay"],
                       ["model", "denominator"], strata, **kw),
        mean_intervals(clean.assign(denominator="clean"),
                       ["false_alarms_per_run", "false_alarms_per_hour"], ["model", "denominator"], strata, **kw),
        mean_intervals(work.assign(denominator="mixed"), ["interval_iou", "vus_pr", "false_alarms_per_run"],
                       ["model", "denominator"], strata, **kw),
    ]
    if onsets is not None and not onsets.empty:
        cols = [c for c in onsets.columns if c.startswith("delay_from_")]
        parts.append(mean_intervals(onsets.assign(denominator="impaired_onset"), cols,
                                    ["model", "denominator"], strata, **kw))
    return pd.concat(parts, ignore_index=True)


def closed_loop_intervals(per_pair: pd.DataFrame, strata: dict[str, str], **kw) -> pd.DataFrame:
    return mean_intervals(per_pair, ["delta"], ["policy", "endpoint"], strata, **kw)
