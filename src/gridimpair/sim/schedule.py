from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gridimpair.contracts import stable_hash
from gridimpair.sim.profiles import ProfileBundle


@dataclass(frozen=True)
class ScheduleBundle:
    p_setpoint: np.ndarray
    q_setpoint: np.ndarray
    endpoint_ids: tuple[str, ...]
    schedule_hash: str

    def at(self, logical_time: int) -> tuple[float, ...]:
        return tuple(self.p_setpoint[logical_time].tolist())


def freeze_remote_schedule(
    profile: ProfileBundle, network: Any = None, *, seed: int = 0
) -> ScheduleBundle:
    rng = np.random.default_rng(seed)
    if network is not None and hasattr(network, "sgen"):
        sgen_count = len(network.sgen)
        storage_count = (
            len(network.storage) if hasattr(network, "storage") else 0
        )
        configured_capacity = (
            network.sgen.sn_mva.fillna(0.0).to_numpy(dtype=float)
            if "sn_mva" in network.sgen
            else np.zeros(sgen_count, dtype=float)
        )
        sgen_capacity = np.maximum.reduce(
            (
                configured_capacity,
                network.sgen.p_mw.to_numpy(dtype=float),
                np.full(sgen_count, 0.05),
            )
        )
    else:
        sgen_count = 3
        storage_count = 0
        sgen_capacity = rng.uniform(0.1, 0.5, sgen_count)
    sgen_p = np.empty((profile.steps, sgen_count), dtype=float)
    split = max(1, sgen_count // 2)
    sgen_p[:, :split] = np.outer(profile.pv_scale, sgen_capacity[:split])
    if split < sgen_count:
        sgen_p[:, split:] = np.outer(
            profile.wind_scale, sgen_capacity[split:]
        )
    seconds = np.arange(profile.steps, dtype=float) * profile.step_seconds
    for index in range(sgen_count):
        period = rng.uniform(90.0, 240.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        modulation = 0.86 + 0.10 * np.sin(
            2 * np.pi * seconds / period + phase
        )
        dispatch_noise = np.empty(profile.steps, dtype=float)
        state = 0.0
        for time_index in range(profile.steps):
            state = 0.95 * state + rng.normal(0.0, 0.002)
            dispatch_noise[time_index] = state
        sgen_p[:, index] *= np.clip(
            modulation + dispatch_noise, 0.65, 1.0
        )
        sgen_p[:, index] = np.clip(
            sgen_p[:, index], 0.0, sgen_capacity[index]
        )
    if storage_count:
        time = np.linspace(0, 2 * np.pi, profile.steps, endpoint=False)
        if hasattr(network.storage, "max_e_mwh"):
            energy = np.maximum(
                network.storage.max_e_mwh.to_numpy(dtype=float), 0.1
            )
            storage_power = np.minimum(0.25, energy / 2)
        else:
            storage_power = np.full(storage_count, 0.1)
        storage_p = np.outer(np.sin(time), storage_power)
        p = np.concatenate([sgen_p, storage_p], axis=1)
    else:
        p = sgen_p
    q = np.zeros_like(p)
    endpoint_ids = tuple(f"sgen-{index}" for index in range(sgen_count)) + tuple(
        f"storage-{index}" for index in range(storage_count)
    )
    schedule_hash = stable_hash(
        {"p": p.tolist(), "q": q.tolist(), "endpoint_ids": endpoint_ids}
    )
    return ScheduleBundle(p, q, endpoint_ids, schedule_hash)
