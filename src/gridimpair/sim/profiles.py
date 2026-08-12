from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridimpair.contracts import stable_hash


@dataclass(frozen=True)
class ProfileBundle:
    load_scale: np.ndarray
    pv_scale: np.ndarray
    wind_scale: np.ndarray
    ambient: np.ndarray
    step_seconds: float
    profile_hash: str

    @property
    def steps(self) -> int:
        return int(self.load_scale.size)

    def at(self, logical_time: int) -> dict[str, float]:
        return {
            "load_scale": float(self.load_scale[logical_time]),
            "pv_scale": float(self.pv_scale[logical_time]),
            "wind_scale": float(self.wind_scale[logical_time]),
            "ambient": float(self.ambient[logical_time]),
        }


def _correlated_noise(
    rng: np.random.Generator,
    *,
    steps: int,
    innovation_scale: float,
    persistence: float = 0.98,
) -> np.ndarray:
    values = np.empty(steps, dtype=float)
    state = 0.0
    for index in range(steps):
        state = persistence * state + rng.normal(0.0, innovation_scale)
        values[index] = state
    return values


def make_profile(
    regime: str,
    seed: int,
    steps: int,
    *,
    step_seconds: float = 1.0,
) -> ProfileBundle:
    aliases = {
        "smooth": "balanced",
        "volatile": "high_load_low_der",
        "ramp": "low_load_high_der",
    }
    regime = aliases.get(regime, regime)
    if regime not in {"balanced", "high_load_low_der", "low_load_high_der"}:
        raise ValueError(
            "regime must be balanced, high_load_low_der, or low_load_high_der"
        )
    if steps < 1:
        raise ValueError("steps must be positive")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    rng = np.random.default_rng(seed)
    time_hours = np.arange(steps, dtype=float) * step_seconds / 3600.0
    start_hour = rng.uniform(8.0, 15.0)
    hour = start_hour + time_hours
    noise_scale = {
        "balanced": 0.012,
        "high_load_low_der": 0.025,
        "low_load_high_der": 0.018,
    }[regime]
    mean_load = {
        "balanced": 1.0,
        "high_load_low_der": 1.2,
        "low_load_high_der": 0.8,
    }[regime]
    mean_der = {
        "balanced": 0.75,
        "high_load_low_der": 0.25,
        "low_load_high_der": 1.0,
    }[regime]
    load = mean_load + 0.10 * np.sin(2 * np.pi * (hour - 7.0) / 24.0)
    pv_shape = np.maximum(0.05, np.sin(np.pi * (hour - 6.0) / 12.0))
    pv = mean_der * pv_shape / max(float(np.mean(pv_shape)), 1e-8)
    wind = mean_der + 0.10 * np.sin(2 * np.pi * hour / 6.0 + 0.8)
    innovation = noise_scale * 0.08
    load = np.clip(
        load
        + _correlated_noise(
            rng, steps=steps, innovation_scale=innovation
        ),
        0.65,
        1.25,
    )
    pv = np.clip(
        pv
        + _correlated_noise(
            rng, steps=steps, innovation_scale=innovation
        ),
        0.0,
        1.05,
    )
    wind = np.clip(
        wind
        + _correlated_noise(
            rng, steps=steps, innovation_scale=innovation
        ),
        0.20,
        1.05,
    )
    ambient = np.clip(
        20
        + 7 * np.sin(2 * np.pi * (hour - 6.0) / 24.0)
        + _correlated_noise(
            rng, steps=steps, innovation_scale=0.02
        ),
        5,
        40,
    )
    profile_hash = stable_hash(
        {
            "load": load.tolist(),
            "pv": pv.tolist(),
            "wind": wind.tolist(),
            "ambient": ambient.tolist(),
            "step_seconds": step_seconds,
        }
    )
    return ProfileBundle(
        load_scale=load,
        pv_scale=pv,
        wind_scale=wind,
        ambient=ambient,
        step_seconds=float(step_seconds),
        profile_hash=profile_hash,
    )
