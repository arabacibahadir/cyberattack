"""Energy and resource accounting for training and inference.

GPU energy is read from the NVML board energy counter (millijoules since
driver load), so it is a measurement of the whole board, including idle draw.
CPU energy is an estimate: process CPU seconds multiplied by a per-core power
budget (package TDP divided by physical cores). AMD processors on Windows do
not expose RAPL counters, so no direct CPU measurement is available.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

CPU_TDP_WATTS = float(os.environ.get("GRIDIMPAIR_CPU_TDP_WATTS", "65"))
CPU_PHYSICAL_CORES = int(os.environ.get("GRIDIMPAIR_CPU_CORES", "6"))

try:  # optional dependency: nvidia-ml-py
    import pynvml

    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    pynvml.nvmlDeviceGetTotalEnergyConsumption(_HANDLE)
except Exception:  # noqa: BLE001 - any NVML failure disables GPU metering
    _HANDLE = None


def gpu_metering_available() -> bool:
    return _HANDLE is not None


def gpu_power_watts() -> float | None:
    if _HANDLE is None:
        return None
    return pynvml.nvmlDeviceGetPowerUsage(_HANDLE) / 1000.0


@dataclass(frozen=True)
class EnergySnapshot:
    wall: float
    cpu: float
    gpu_mj: int | None


def snapshot() -> EnergySnapshot:
    gpu = None
    if _HANDLE is not None:
        try:
            gpu = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(_HANDLE))
        except Exception:  # noqa: BLE001
            gpu = None
    return EnergySnapshot(time.perf_counter(), time.process_time(), gpu)


def delta(start: EnergySnapshot, end: EnergySnapshot | None = None) -> dict[str, float]:
    end = end or snapshot()
    cpu_seconds = end.cpu - start.cpu
    gpu_joules = (
        (end.gpu_mj - start.gpu_mj) / 1000.0
        if start.gpu_mj is not None and end.gpu_mj is not None
        else float("nan")
    )
    return {
        "wall_seconds": end.wall - start.wall,
        "cpu_seconds": cpu_seconds,
        "cpu_energy_j_est": cpu_seconds * CPU_TDP_WATTS / CPU_PHYSICAL_CORES,
        "gpu_energy_j": gpu_joules,
    }
