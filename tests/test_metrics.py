from __future__ import annotations

import math

import numpy as np
import pytest

from gridimpair.evaluation.metrics import (
    bag_metrics,
    derive_effective_onsets,
    risk_coverage_auc,
    temporal_metrics,
    three_onset_delays,
    vus_pr,
)


def test_bag_metrics_match_hand_checked_perfect_ranking() -> None:
    result = bag_metrics(
        y_true=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.1, 0.4, 0.6, 0.9]),
        threshold=0.5,
    )
    assert result["macro_f1"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["brier"] == pytest.approx(0.085)


def test_temporal_metrics_do_not_point_adjust_events() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0])
    scores = np.array([0.0, 0.1, 0.4, 0.8, 0.3, 0.0])
    result = temporal_metrics(labels, scores, threshold=0.5)
    assert result["event_recall"] == 1.0
    assert result["interval_iou"] == 0.5
    assert result["detection_delay"] == 1.0
    assert result["false_alarms"] == 0


def test_temporal_metrics_accept_raw_ranking_scores_and_explicit_alarm_policy() -> None:
    truth = np.asarray([0, 1, 1, 0])
    raw_scores = np.asarray([-4.0, 3.0, 2.0, -1.0])
    alarms = np.asarray([False, True, True, False])
    result = temporal_metrics(
        truth,
        raw_scores,
        threshold=None,
        alarms=alarms,
    )
    assert result["event_recall"] == 1.0
    assert result["false_alarms"] == 0
    assert result["temporal_auprc"] == 1.0


def test_vus_pr_and_risk_coverage_are_bounded() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.4, 0.8, 0.3, 0.1])
    assert 0.0 <= vus_pr(labels, scores, max_tolerance=2) <= 1.0
    value = risk_coverage_auc(labels, scores)
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_three_onsets_are_kept_distinct() -> None:
    clean_command = np.array([0, 0, 1, 2, 3, 4, 5], dtype=float)
    impaired_command = np.array([0, 0, 1, 1, 1, 4, 5], dtype=float)
    clean_endpoint = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float)
    impaired_endpoint = np.array([1, 1, 1, 1, 0.8, 0.8, 1], dtype=float)
    onsets = derive_effective_onsets(
        [(2, 6)],
        clean_command,
        impaired_command,
        clean_endpoint,
        impaired_endpoint,
        command_tolerance=0.1,
        electrical_tolerance=0.05,
    )
    assert onsets[0] == {
        "communication_onset": 2,
        "command_impact_onset": 3,
        "electrical_effect_onset": 4,
        "search_stop": 7,
    }
    delays = three_onset_delays(
        np.array([0, 0, 0, 0, 0, 1, 0], dtype=float),
        threshold=0.5,
        onset_records=onsets,
    )
    assert delays == {
        "delay_from_communication": 3.0,
        "recall_from_communication": 1.0,
        "delay_from_command_impact": 2.0,
        "recall_from_command_impact": 1.0,
        "delay_from_electrical_effect": 1.0,
        "recall_from_electrical_effect": 1.0,
    }


def test_onset_delay_does_not_credit_a_detection_from_a_later_episode() -> None:
    delays = three_onset_delays(
        np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float),
        threshold=0.5,
        onset_records=[
            {
                "communication_onset": 2,
                "command_impact_onset": 3,
                "electrical_effect_onset": 4,
                "search_stop": 6,
            },
            {
                "communication_onset": 6,
                "command_impact_onset": 6,
                "electrical_effect_onset": 7,
                "search_stop": 10,
            },
        ],
    )
    assert delays["delay_from_communication"] == 1.0
    assert delays["recall_from_communication"] == 0.5
