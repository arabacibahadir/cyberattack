from __future__ import annotations

import numpy as np

from gridimpair.evaluation.operating_points import (
    apply_alarm_policy,
    fit_clean_false_alarm_threshold,
    fit_temporal_operating_points,
)


def test_alarm_policy_removes_isolated_spikes_and_requires_release_dwell() -> None:
    scores = np.asarray([0.0, 0.9, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0])
    alarms = apply_alarm_policy(
        scores,
        threshold=0.5,
        min_on_steps=3,
        min_off_steps=3,
    )
    assert alarms.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]


def test_operating_point_is_fit_from_clean_runs_to_false_alarm_budget() -> None:
    clean_scores = np.asarray(
        [
            [0.1, 0.2, 0.1, 0.2, 0.1],
            [0.2, 0.3, 0.2, 0.3, 0.2],
        ]
    )
    threshold = fit_clean_false_alarm_threshold(
        clean_scores,
        target_false_alarms_per_run=0.0,
        min_on_steps=1,
        min_off_steps=1,
    )
    assert threshold > 0.3
    assert not any(
        apply_alarm_policy(
            run,
            threshold=threshold,
            min_on_steps=1,
            min_off_steps=1,
        ).any()
        for run in clean_scores
    )


def test_each_model_gets_its_own_clean_validation_operating_point() -> None:
    points = fit_temporal_operating_points(
        {
            "small_scale": np.asarray(
                [[0.1, 0.2], [0.2, 0.3], [9.0, 9.0]]
            ),
            "large_scale": np.asarray(
                [[10.0, 20.0], [20.0, 30.0], [90.0, 90.0]]
            ),
        },
        clean_validation_mask=np.asarray([True, True, False]),
        target_false_alarms_per_run=0.0,
        min_on_steps=1,
        min_off_steps=1,
    )
    assert points["small_scale"]["threshold"] > 0.3
    assert points["large_scale"]["threshold"] > 30.0
    assert points["small_scale"]["threshold"] < points["large_scale"]["threshold"]
