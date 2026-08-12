from __future__ import annotations

import numpy as np

from gridimpair.evaluation.temporal import (
    interval_labels,
    rolling_derivative_energy,
    rolling_flatline_score,
)


def test_interval_labels_marks_only_half_open_impairment_intervals() -> None:
    labels = interval_labels(8, [(2, 4), (6, 8)])
    assert labels.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_temporal_heuristics_do_not_read_future_values() -> None:
    original = np.arange(24, dtype=float).reshape(8, 3)
    changed_future = original.copy()
    changed_future[5:] += 10_000
    for scorer in (rolling_derivative_energy, rolling_flatline_score):
        before = scorer(original)
        after = scorer(changed_future)
        assert np.array_equal(before[:5], after[:5])
