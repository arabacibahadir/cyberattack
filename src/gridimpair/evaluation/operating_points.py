from __future__ import annotations

import numpy as np


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
    alarms = np.zeros(values.shape, dtype=bool)
    active = False
    on_count = 0
    off_count = 0
    for index, value in enumerate(values):
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
    for threshold in np.unique(candidates):
        counts = [
            _alarm_episode_count(
                apply_alarm_policy(
                    run,
                    threshold=float(threshold),
                    min_on_steps=min_on_steps,
                    min_off_steps=min_off_steps,
                )
            )
            for run in values
        ]
        if float(np.mean(counts)) <= target_false_alarms_per_run:
            return float(threshold)
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
