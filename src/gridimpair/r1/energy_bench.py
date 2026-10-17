"""Matched-hardware inference energy benchmark and training-energy summary.

Inference: every frozen detector re-scores the same locked runs repeatedly
until at least ``min_seconds`` have elapsed. GPU energy is the NVML board
counter minus an idle baseline measured immediately before (dynamic energy);
CPU energy is the per-core TDP estimate from :mod:`gridimpair.energy`.
Neural models are measured as deployed (three-seed ensemble) in two settings:
CUDA with batch 64 and CPU with batch 1 on one thread. Rule-based and
classical detectors run on CPU with batch 1, as in the frozen protocol.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gridimpair import energy
from gridimpair.r1 import frozen


def idle_gpu_watts(seconds: float = 10.0) -> float:
    if not energy.gpu_metering_available():
        return float("nan")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = energy.snapshot()
    time.sleep(seconds)
    cost = energy.delta(start)
    return cost["gpu_energy_j"] / cost["wall_seconds"]


def _measure(callable_, runs: int, min_seconds: float, min_repeats: int, idle_watts: float, cuda: bool) -> dict:
    callable_()  # warm-up
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    repeats = 0
    start = energy.snapshot()
    while True:
        callable_()
        if cuda:
            torch.cuda.synchronize()
        repeats += 1
        elapsed = time.perf_counter() - start.wall
        if repeats >= min_repeats and elapsed >= min_seconds:
            break
    cost = energy.delta(start)
    scored = runs * repeats
    dynamic = cost["gpu_energy_j"] - idle_watts * cost["wall_seconds"] if np.isfinite(idle_watts) else np.nan
    return {
        "repeats": repeats,
        "runs_scored": scored,
        "latency_ms_per_run": cost["wall_seconds"] * 1000 / scored,
        "throughput_runs_per_s": scored / cost["wall_seconds"],
        "gpu_dynamic_j_per_run": (dynamic / scored) if cuda else 0.0,
        "gpu_board_j_per_run": cost["gpu_energy_j"] / scored if cuda else np.nan,
        "cpu_j_per_run_est": cost["cpu_energy_j_est"] / scored,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0,
    }


def inference_benchmark(
    inputs: frozen.FrozenInputs,
    detectors: dict[str, frozen.FrozenDetector],
    training_dir: Path,
    *,
    min_seconds: float = 5.0,
    min_repeats: int = 2,
    idle_seconds: float = 10.0,
) -> pd.DataFrame:
    prepared = frozen.prepare(inputs, training_dir)
    runs = inputs.raw_values.shape[0]
    idle = idle_gpu_watts(idle_seconds)
    rows = []
    threads = torch.get_num_threads()
    for name, detector in detectors.items():
        base = {"model": name, "kind": detector.kind, "feature_tier": detector.feature_tier,
                "parameters_per_model": detector.parameters, "seeds": detector.extra.get("seeds", 1),
                "idle_gpu_watts": idle}
        if detector.kind == "neural":
            if torch.cuda.is_available():
                rows.append({**base, "device": "cuda", "batch_size": 64,
                             **_measure(lambda: detector.score(prepared, device="cuda", batch_size=64),
                                        runs, min_seconds, min_repeats, idle, cuda=True)})
            torch.set_num_threads(1)
            try:
                rows.append({**base, "device": "cpu", "batch_size": 1,
                             **_measure(lambda: detector.score(prepared, device="cpu", batch_size=1),
                                        runs, min_seconds, min_repeats, idle, cuda=False)})
            finally:
                torch.set_num_threads(threads)
        else:
            rows.append({**base, "device": "cpu", "batch_size": 1,
                         **_measure(lambda: detector.score(prepared), runs, min_seconds, min_repeats, idle, cuda=False)})
    return pd.DataFrame(rows)


_TRIAL = re.compile(r"trial[-_]?(\d+)", re.IGNORECASE)
_SEED = re.compile(r"seed[-_]?(\d+)", re.IGNORECASE)


def training_energy_summary(training_dir: Path) -> pd.DataFrame:
    """Aggregate per-epoch training cost from every metrics.csv under training/."""
    rows = []
    for path in sorted(training_dir.rglob("metrics.csv")):
        frame = pd.read_csv(path)
        if "gpu_energy_j" not in frame:
            continue
        relative = path.relative_to(training_dir).as_posix()
        trial, seed = _TRIAL.search(relative), _SEED.search(relative)
        rows.append({
            "run_path": relative,
            "phase": "hpo" if "hpo" in relative.lower() else "final",
            "trial": int(trial.group(1)) if trial else np.nan,
            "seed": int(seed.group(1)) if seed else np.nan,
            "epochs": len(frame),
            "train_seconds": float(frame["epoch_seconds"].sum()),
            "gpu_energy_j": float(frame["gpu_energy_j"].sum()),
            "cpu_energy_j_est": float(frame["cpu_energy_j_est"].sum()),
            "peak_vram_mb": float(frame["gpu_peak_memory_mb"].max()),
        })
    return pd.DataFrame(rows)


def baseline_training_cost(training_dir: Path) -> pd.DataFrame:
    paths = sorted(training_dir.rglob("baseline_cost.csv"))
    return pd.concat([pd.read_csv(p).assign(source=p.relative_to(training_dir).as_posix()) for p in paths], ignore_index=True) if paths else pd.DataFrame()
