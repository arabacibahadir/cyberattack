from __future__ import annotations

import numpy as np


def first_difference_view(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    return np.diff(raw, axis=-1, prepend=raw[..., :1])


def summary_features(cases: np.ndarray) -> np.ndarray:
    values = np.asarray(cases, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("expected (cases, channels, time)")
    time = np.arange(values.shape[-1], dtype=float)
    centered_time = time - time.mean()
    denominator = float(np.sum(centered_time**2)) or 1.0
    slopes = np.sum(
        (values - values.mean(axis=-1, keepdims=True)) * centered_time,
        axis=-1,
    ) / denominator
    parts = [
        values.mean(axis=-1),
        values.std(axis=-1),
        values.min(axis=-1),
        values.max(axis=-1),
        slopes,
    ]
    return np.concatenate(parts, axis=1)


def dual_views(raw: np.ndarray, ridge_residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_array = np.asarray(raw, dtype=np.float32)
    residual_array = np.asarray(ridge_residual, dtype=np.float32)
    if raw_array.shape != residual_array.shape:
        raise ValueError("raw and residual views must have identical shapes")
    raw_and_difference = np.concatenate(
        [raw_array, first_difference_view(raw_array)], axis=-2
    )
    return raw_and_difference, residual_array
