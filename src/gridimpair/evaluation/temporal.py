from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def interval_labels(
    length: int, intervals: Iterable[tuple[int, int] | list[int]]
) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be positive")
    labels = np.zeros(length, dtype=np.int8)
    for raw_start, raw_stop in intervals:
        start, stop = int(raw_start), int(raw_stop)
        if not 0 <= start < stop <= length:
            raise ValueError("impairment interval is outside the trace")
        labels[start:stop] = 1
    return labels


def rolling_derivative_energy(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must have shape (time, channels)")
    result = np.zeros(len(values), dtype=float)
    if len(values) > 1:
        result[1:] = np.mean(np.diff(values, axis=0) ** 2, axis=1)
    return result


def rolling_flatline_score(
    series: np.ndarray, *, window: int = 32
) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must have shape (time, channels)")
    if window < 2:
        raise ValueError("window must be at least two")
    output = np.zeros(len(values), dtype=float)
    for stop in range(2, len(values) + 1):
        segment = values[max(0, stop - window) : stop]
        output[stop - 1] = float(
            np.max(1.0 / (np.var(segment, axis=0) + 1e-8))
        )
    return output
