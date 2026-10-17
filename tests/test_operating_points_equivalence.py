"""The accelerated threshold search must return exactly what the original pure-Python search returned."""
import numpy as np
import pytest

from gridimpair.evaluation import operating_points as fast


def _reference_alarms(values, threshold, min_on, min_off):
    alarms = np.zeros(len(values), dtype=bool)
    active, on_count, off_count = False, 0, 0
    for index, value in enumerate(values):
        if active:
            if value < threshold:
                off_count += 1
                if off_count >= min_off:
                    active, off_count = False, 0
            else:
                off_count = 0
        else:
            if value >= threshold:
                on_count += 1
                if on_count >= min_on:
                    active, on_count = True, 0
            else:
                on_count = 0
        alarms[index] = active
    return alarms


def _reference_threshold(scores, target, min_on, min_off):
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique, np.inf), np.nextafter(unique[-1], np.inf)]
    for threshold in np.unique(candidates):
        counts = []
        for run in scores:
            alarms = _reference_alarms(run, float(threshold), min_on, min_off)
            counts.append(int(np.logical_and(alarms, ~np.r_[False, alarms[:-1]]).sum()))
        if float(np.mean(counts)) <= target:
            return float(threshold)
    return float(np.nextafter(unique[-1], np.inf))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_threshold_search_matches_reference(seed):
    rng = np.random.default_rng(seed)
    continuous = rng.random((6, 120))
    ties = rng.integers(0, 4, (6, 120)).astype(float)
    for scores in (continuous, ties):
        for target in (0.0, 0.1, 0.5):
            assert fast.fit_clean_false_alarm_threshold(
                scores, target_false_alarms_per_run=target, min_on_steps=3, min_off_steps=3
            ) == _reference_threshold(scores, target, 3, 3)
        for threshold in (0.2, 0.5, 2.0):
            assert np.array_equal(
                fast.apply_alarm_policy(scores[0], threshold=threshold, min_on_steps=2, min_off_steps=3),
                _reference_alarms(scores[0], threshold, 2, 3),
            )
