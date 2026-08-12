from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    lower: float
    upper: float
    confidence: float
    n_pairs: int


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    effective_pairs: int
    status: str


@dataclass(frozen=True)
class GapClosureResult:
    value: float | None
    status: str


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    pair_column: str = "pair_id",
    stratum_column: str = "stratum",
    n_resamples: int = 10_000,
    confidence: float = 0.975,
    seed: int = 2026,
) -> BootstrapResult:
    required = {candidate_column, baseline_column, pair_column, stratum_column}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing bootstrap columns: {sorted(missing)}")
    if frame[pair_column].duplicated().any():
        raise ValueError("paired bootstrap expects one row per pair")
    if n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    work = frame.copy()
    work["_difference"] = work[candidate_column] - work[baseline_column]
    observed = float(work["_difference"].mean())
    rng = np.random.default_rng(seed)
    strata = [
        group["_difference"].to_numpy(float)
        for _, group in work.groupby(stratum_column, sort=True)
    ]
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        resampled = [
            rng.choice(values, size=len(values), replace=True) for values in strata
        ]
        samples[index] = np.concatenate(resampled).mean()
    alpha = 1 - confidence
    lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        observed=observed,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_pairs=len(work),
    )


def sign_flip_test(
    differences: Sequence[float],
    *,
    n_permutations: int = 100_000,
    seed: int = 2026,
) -> PermutationResult:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if not values.size:
        return PermutationResult(0.0, 1.0, 0, "all_ties")
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(values)))
    null = np.mean(signs * values, axis=1)
    extreme = int(np.sum(np.abs(null) >= abs(observed)))
    return PermutationResult(
        observed=observed,
        p_value=(extreme + 1) / (n_permutations + 1),
        effective_pairs=len(values),
        status="ok",
    )


def holm_correction(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        if not 0 <= value <= 1:
            raise ValueError("p-values must be in [0, 1]")
        running = max(running, (total - index) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def gap_closure(
    *, baseline: float, candidate: float, oracle: float
) -> GapClosureResult:
    headroom = oracle - baseline
    if headroom <= 0:
        return GapClosureResult(None, "no_oracle_headroom")
    return GapClosureResult((candidate - baseline) / headroom, "ok")
