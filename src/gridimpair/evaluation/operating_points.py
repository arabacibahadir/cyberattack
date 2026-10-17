from __future__ import annotations

import numpy as np

try:  # optional acceleration; results are identical to the pure-Python path
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _alarm_kernel(values, threshold, min_on_steps, min_off_steps, alarms):
    active = False
    on_count = 0
    off_count = 0
    for index in range(values.shape[0]):
        value = values[index]
        if active:
            if value < threshold:
                off_count += 1
                if off_count >= min_off_steps:
                    active = False
                    off_count = 0
            else:
                off_count = 0
        else:
            if value >= threshold:
                on_count += 1
                if on_count >= min_on_steps:
                    active = True
                    on_count = 0
            else:
                on_count = 0
        alarms[index] = active


def _mean_episodes_kernel(runs, threshold, min_on_steps, min_off_steps):
    total = 0
    for r in range(runs.shape[0]):
        active = False
        previous = False
        on_count = 0
        off_count = 0
        for index in range(runs.shape[1]):
            value = runs[r, index]
            if active:
                if value < threshold:
                    off_count += 1
                    if off_count >= min_off_steps:
                        active = False
                        off_count = 0
                else:
                    off_count = 0
            else:
                if value >= threshold:
                    on_count += 1
                    if on_count >= min_on_steps:
                        active = True
                        on_count = 0
                else:
                    on_count = 0
            if active and not previous:
                total += 1
            previous = active
    return total / runs.shape[0]


def _first_passing_threshold_kernel(runs, candidates, target, min_on_steps, min_off_steps):
    for k in range(candidates.shape[0]):
        if _mean_episodes(runs, candidates[k], min_on_steps, min_off_steps) <= target:
            return k
    return -1


if njit is not None:
    _alarm = njit(cache=True)(_alarm_kernel)
    _mean_episodes = njit(cache=True)(_mean_episodes_kernel)
    _first_passing = njit(cache=True)(_first_passing_threshold_kernel)
else:  # pragma: no cover
    _alarm = _alarm_kernel
    _mean_episodes = _mean_episodes_kernel
    _first_passing = _first_passing_threshold_kernel


def apply_alarm_policy(
    scores: np.ndarray,
    *,
    threshold: float,
    min_on_steps: int,
    min_off_steps: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    if min_on_steps < 1 or min_off_steps < 1:
        raise ValueError("alarm dwell lengths must be positive")
    alarms = np.zeros(values.shape, dtype=np.bool_)
    _alarm(values, float(threshold), int(min_on_steps), int(min_off_steps), alarms)
    return alarms


def _alarm_episode_count(alarms: np.ndarray) -> int:
    binary = np.asarray(alarms, dtype=bool).reshape(-1)
    return int(np.logical_and(binary, ~np.r_[False, binary[:-1]]).sum())


def fit_clean_false_alarm_threshold(
    clean_scores: np.ndarray,
    *,
    target_false_alarms_per_run: float,
    min_on_steps: int,
    min_off_steps: int,
) -> float:
    values = np.asarray(clean_scores, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("clean_scores must be a finite (runs, time) matrix")
    if target_false_alarms_per_run < 0:
        raise ValueError("false-alarm budget must be non-negative")
    unique = np.unique(values)
    candidates = np.r_[
        np.nextafter(unique, np.inf),
        np.nextafter(unique[-1], np.inf),
    ]
    ordered = np.unique(candidates)
    index = _first_passing(
        np.ascontiguousarray(values, dtype=np.float64),
        ordered.astype(np.float64),
        float(target_false_alarms_per_run),
        int(min_on_steps),
        int(min_off_steps),
    )
    if index >= 0:
        return float(ordered[index])
    return float(np.nextafter(unique[-1], np.inf))


def fit_temporal_operating_points(
    scores_by_model: dict[str, np.ndarray],
    *,
    clean_validation_mask: np.ndarray,
    target_false_alarms_per_run: float,
    min_on_steps: int,
    min_off_steps: int,
) -> dict[str, dict[str, float | int]]:
    mask = np.asarray(clean_validation_mask, dtype=bool).reshape(-1)
    if not mask.any():
        raise ValueError("at least one clean validation run is required")
    points: dict[str, dict[str, float | int]] = {}
    for model, scores in sorted(scores_by_model.items()):
        values = np.asarray(scores, dtype=float)
        if values.ndim != 2 or values.shape[0] != mask.size:
            raise ValueError(
                f"temporal scores for {model!r} must have shape (runs, time)"
            )
        threshold = fit_clean_false_alarm_threshold(
            values[mask],
            target_false_alarms_per_run=target_false_alarms_per_run,
            min_on_steps=min_on_steps,
            min_off_steps=min_off_steps,
        )
        points[model] = {
            "threshold": threshold,
            "min_on_steps": int(min_on_steps),
            "min_off_steps": int(min_off_steps),
            "clean_validation_runs": int(mask.sum()),
            "target_false_alarms_per_run": float(
                target_false_alarms_per_run
            ),
        }
    return points
