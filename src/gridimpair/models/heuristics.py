from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _development_only(split: str) -> None:
    if split not in {"development", "validation_a", "validation_b"}:
        raise PermissionError("detector fitting is restricted to development splits")


@dataclass
class _ThresholdDetector:
    quantile: float = 0.99
    threshold_: float | None = None

    def _fit_scores(self, scores: np.ndarray, split: str) -> "_ThresholdDetector":
        _development_only(split)
        finite = np.asarray(scores, dtype=float)
        finite = finite[np.isfinite(finite)]
        self.threshold_ = float(np.quantile(finite, self.quantile)) if finite.size else 0.0
        return self


class FlatlineDetector(_ThresholdDetector):
    def fit(self, train: np.ndarray, *, split: str) -> "FlatlineDetector":
        array = np.asarray(train, dtype=float)
        scores = 1.0 / (np.var(array, axis=0) + 1e-8)
        self._fit_scores(scores, split)
        return self

    def score(self, causal_window: np.ndarray) -> float:
        array = np.asarray(causal_window, dtype=float)
        if len(array) < 2:
            return 0.0
        return float(np.max(1.0 / (np.var(array, axis=0) + 1e-8)))


class DerivativeEnergyDetector(_ThresholdDetector):
    def fit(self, train: np.ndarray, *, split: str) -> "DerivativeEnergyDetector":
        array = np.asarray(train, dtype=float)
        scores = np.mean(np.diff(array, axis=0) ** 2, axis=1)
        self._fit_scores(scores, split)
        return self

    def score(self, causal_window: np.ndarray) -> float:
        differences = np.diff(np.asarray(causal_window, dtype=float), axis=0)
        return float(np.mean(differences[-1] ** 2)) if len(differences) else 0.0


class EwmaCusumDetector(_ThresholdDetector):
    def __init__(self, decay: float = 0.9, quantile: float = 0.99) -> None:
        super().__init__(quantile=quantile)
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay

    def _scores(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        center = np.zeros(array.shape[1], dtype=float)
        scores = []
        for row in array:
            innovation = row - center
            scores.append(np.abs(innovation).sum())
            center = self.decay * center + (1 - self.decay) * row
        return np.asarray(scores)

    def fit(self, train: np.ndarray, *, split: str) -> "EwmaCusumDetector":
        self._fit_scores(self._scores(train), split)
        return self

    def score(self, causal_window: np.ndarray) -> float:
        scores = self._scores(causal_window)
        return float(scores[-1]) if scores.size else 0.0


class AgeWatchdog:
    def __init__(self, max_age: float) -> None:
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        self.max_age = max_age

    def fit(self, train: np.ndarray, *, split: str = "development") -> "AgeWatchdog":
        _development_only(split)
        return self

    def score(self, causal_window: np.ndarray) -> float:
        values = np.asarray(causal_window, dtype=float)
        age = float(values.reshape(-1)[-1]) if values.size else 0.0
        return float(age > self.max_age)
