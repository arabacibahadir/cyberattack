from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def _vectors(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if truth.shape != scores.shape or not truth.size:
        raise ValueError("truth and scores must be non-empty vectors of equal length")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("truth must be binary")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return truth, scores


def _score_vectors(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=int).reshape(-1)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if truth.shape != values.shape or not truth.size:
        raise ValueError("truth and scores must be non-empty vectors of equal length")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("truth must be binary")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return truth, values


def risk_coverage_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    truth, scores = _vectors(y_true, probabilities)
    predicted = (scores >= 0.5).astype(int)
    errors = (predicted != truth).astype(float)
    confidence = np.maximum(scores, 1 - scores)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(np.r_[0.0, cumulative_risk], np.r_[0.0, coverage]))


def bag_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    truth, scores = _vectors(y_true, probabilities)
    predicted = (scores >= threshold).astype(int)
    two_classes = np.unique(truth).size == 2
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "auroc": float(roc_auc_score(truth, scores)) if two_classes else float("nan"),
        "auprc": float(average_precision_score(truth, scores)),
        "brier": float(brier_score_loss(truth, scores)),
        "nll": float(log_loss(truth, np.c_[1 - scores, scores], labels=[0, 1])),
        "aurc": risk_coverage_auc(truth, scores),
    }


def _intervals(binary: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[0, np.asarray(binary, dtype=int), 0]
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return binary.astype(int, copy=True)
    output = np.zeros_like(binary, dtype=int)
    for start, stop in _intervals(binary):
        output[max(0, start - radius) : min(len(binary), stop + radius)] = 1
    return output


def vus_pr(
    labels: np.ndarray, scores: np.ndarray, *, max_tolerance: int = 20
) -> float:
    truth, ranking_scores = _score_vectors(labels, scores)
    if max_tolerance < 0:
        raise ValueError("max_tolerance must be non-negative")
    if not truth.any():
        return 0.0
    values = [
        average_precision_score(_dilate(truth, radius), ranking_scores)
        for radius in range(max_tolerance + 1)
    ]
    return float(np.mean(values))


def temporal_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None,
    alarms: np.ndarray | None = None,
    run_count: int = 1,
    vus_tolerance: int = 20,
) -> dict[str, Any]:
    truth, ranking_scores = _score_vectors(labels, scores)
    if run_count < 1:
        raise ValueError("run_count must be positive")
    if alarms is None:
        if threshold is None:
            raise ValueError("threshold is required when alarms are not supplied")
        predicted = ranking_scores >= threshold
    else:
        predicted = np.asarray(alarms, dtype=bool).reshape(-1)
        if predicted.shape != truth.shape:
            raise ValueError("alarms must match labels")
    true_intervals = _intervals(truth)
    detected = 0
    delays: list[int] = []
    for start, stop in true_intervals:
        hits = np.flatnonzero(predicted[start:stop])
        if hits.size:
            detected += 1
            delays.append(int(hits[0]))
    false_alarm_count = sum(
        not truth[start:stop].any() for start, stop in _intervals(predicted)
    )
    intersection = int(np.logical_and(truth, predicted).sum())
    union = int(np.logical_or(truth, predicted).sum())
    return {
        "event_recall": detected / len(true_intervals) if true_intervals else float("nan"),
        "false_alarms": int(false_alarm_count),
        "false_alarms_per_run": false_alarm_count / run_count,
        "interval_iou": intersection / union if union else 1.0,
        "temporal_auprc": float(average_precision_score(truth, ranking_scores))
        if truth.any()
        else 0.0,
        "vus_pr": vus_pr(truth, ranking_scores, max_tolerance=vus_tolerance),
        "detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay": float(np.median(delays)) if delays else float("nan"),
        "p95_detection_delay": float(np.quantile(delays, 0.95)) if delays else float("nan"),
    }


def _first_difference(
    clean: np.ndarray,
    impaired: np.ndarray,
    *,
    start: int,
    stop: int,
    tolerance: float,
) -> int | None:
    clean_values = np.asarray(clean, dtype=float)
    impaired_values = np.asarray(impaired, dtype=float)
    if clean_values.shape != impaired_values.shape:
        raise ValueError("paired traces must have identical shapes")
    difference = np.abs(clean_values - impaired_values)
    if difference.ndim > 1:
        difference = difference.reshape(len(difference), -1).max(axis=1)
    hits = np.flatnonzero(difference[start:stop] > tolerance)
    return int(start + hits[0]) if hits.size else None


def derive_effective_onsets(
    communication_intervals: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    clean_command: np.ndarray,
    impaired_command: np.ndarray,
    clean_endpoint: np.ndarray,
    impaired_endpoint: np.ndarray,
    *,
    command_tolerance: float,
    electrical_tolerance: float,
) -> list[dict[str, int | None]]:
    if command_tolerance < 0 or electrical_tolerance < 0:
        raise ValueError("onset tolerances must be non-negative")
    horizon = len(np.asarray(clean_command))
    records: list[dict[str, int | None]] = []
    for index, (start, stop) in enumerate(communication_intervals):
        if not 0 <= start < stop <= horizon:
            raise ValueError("communication interval is outside the trace")
        search_stop = (
            communication_intervals[index + 1][0]
            if index + 1 < len(communication_intervals)
            else horizon
        )
        records.append(
            {
                "communication_onset": int(start),
                "command_impact_onset": _first_difference(
                    clean_command,
                    impaired_command,
                    start=start,
                    stop=search_stop,
                    tolerance=command_tolerance,
                ),
                "electrical_effect_onset": _first_difference(
                    clean_endpoint,
                    impaired_endpoint,
                    start=start,
                    stop=search_stop,
                    tolerance=electrical_tolerance,
                ),
                "search_stop": int(search_stop),
            }
        )
    return records


def three_onset_delays(
    detection_scores: np.ndarray,
    *,
    threshold: float,
    onset_records: list[dict[str, int | None]],
) -> dict[str, float]:
    scores = np.asarray(detection_scores, dtype=float).reshape(-1)
    mapping = {
        "communication_onset": "delay_from_communication",
        "command_impact_onset": "delay_from_command_impact",
        "electrical_effect_onset": "delay_from_electrical_effect",
    }
    result: dict[str, float] = {}
    for onset_name, output_name in mapping.items():
        delays = []
        eligible = 0
        for record in onset_records:
            onset = record.get(onset_name)
            if onset is None:
                continue
            eligible += 1
            stop = int(record.get("search_stop") or len(scores))
            hits = np.flatnonzero(
                scores[int(onset) : stop] >= threshold
            )
            if hits.size:
                delays.append(float(hits[0]))
        result[output_name] = float(np.mean(delays)) if delays else float("nan")
        result[output_name.replace("delay_", "recall_")] = (
            len(delays) / eligible if eligible else float("nan")
        )
    return result
