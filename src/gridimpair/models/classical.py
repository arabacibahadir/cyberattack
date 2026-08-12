from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from gridimpair.models.features import summary_features


def _check_train_split(split: str) -> None:
    if split not in {"development", "validation_a", "validation_b"}:
        raise PermissionError("fitting on locked_test is forbidden")


class CausalRidgeResidual:
    def __init__(self, lags: int = 4, alpha: float = 1.0) -> None:
        if lags < 1:
            raise ValueError("lags must be positive")
        self.lags = lags
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.channels_: int | None = None
        self.training_rows_: int = 0

    def _design(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(series, dtype=float)
        if values.ndim != 2:
            raise ValueError("series must have shape (time, channels)")
        rows = [
            values[index - self.lags : index].reshape(-1)
            for index in range(self.lags, len(values))
        ]
        return np.asarray(rows), values[self.lags :]

    def fit(self, train: np.ndarray, *, split: str) -> "CausalRidgeResidual":
        return self.fit_many([train], split=split)

    def fit_many(
        self,
        trains: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        split: str,
    ) -> "CausalRidgeResidual":
        _check_train_split(split)
        if not trains:
            raise ValueError("at least one training series is required")
        designs = [self._design(train) for train in trains]
        nonempty = [
            (features, targets)
            for features, targets in designs
            if len(features)
        ]
        if not nonempty:
            raise ValueError("training series is shorter than the lag count")
        channels = {np.asarray(train).shape[1] for train in trains}
        if len(channels) != 1:
            raise ValueError("all training series must share channel count")
        features = np.concatenate(
            [item[0] for item in nonempty], axis=0
        )
        targets = np.concatenate(
            [item[1] for item in nonempty], axis=0
        )
        self.model.fit(features, targets)
        self.channels_ = channels.pop()
        self.training_rows_ = len(features)
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        if self.channels_ is None:
            raise RuntimeError("model is not fitted")
        values = np.asarray(series, dtype=float)
        features, targets = self._design(values)
        residual = np.full(values.shape, np.nan, dtype=float)
        if len(features):
            residual[self.lags :] = targets - self.model.predict(features)
        return residual

    def score(self, causal_window: np.ndarray) -> float:
        residual = self.transform(causal_window)
        last = residual[-1]
        return float(np.nanmean(np.abs(last))) if np.isfinite(last).any() else 0.0


class XGBoostDetector:
    def __init__(self, **parameters: Any) -> None:
        self.parameters = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 6,
            "random_state": 2026,
            **parameters,
        }
        self.model: Any = None

    def fit(
        self, train: np.ndarray, labels: np.ndarray, *, split: str
    ) -> "XGBoostDetector":
        _check_train_split(split)
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("install the xgboost optional dependency") from exc
        self.model = XGBClassifier(**self.parameters)
        self.model.fit(summary_features(train), np.asarray(labels))
        return self

    def score(self, causal_window: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return float(self.model.predict_proba(summary_features(causal_window))[0, 1])


class MultiRocketHydraDetector:
    def __init__(self, **parameters: Any) -> None:
        self.parameters = {"random_state": 2026, "n_jobs": 6, **parameters}
        self.model: Any = None

    def fit(
        self, train: np.ndarray, labels: np.ndarray, *, split: str
    ) -> "MultiRocketHydraDetector":
        _check_train_split(split)
        values = np.asarray(train)
        if values.ndim != 3:
            raise ValueError("MultiRocket-Hydra input must be (cases, channels, time)")
        try:
            from aeon.classification.convolution_based import MultiRocketHydraClassifier
        except ImportError as exc:
            raise RuntimeError("install the aeon optional dependency") from exc
        self.model = MultiRocketHydraClassifier(**self.parameters)
        self.model.fit(values, np.asarray(labels))
        return self

    def score(self, causal_window: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        values = np.asarray(causal_window)
        if values.ndim == 2:
            values = values[None, ...]
        return float(self.model.predict_proba(values)[0, 1])
