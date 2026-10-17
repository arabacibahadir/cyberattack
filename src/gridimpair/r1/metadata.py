"""Receiver-side metadata replay and trust-degradation perturbations.

Every run stores its full message ledger (production index, actual delivery
index, sequence number, endpoint). ``replay`` re-executes the receiver
bookkeeping of ``sim.scenario`` exactly (deliver at the actual delivery step in
sequence order, reject non-newer sequences, age = t - production index, or
t + 1 without an accepted message), so the five cyber channels can be
recomputed under corrupted metadata without re-solving any power flow. The
physical channels are left unchanged except for the explicit physical
perturbations (measurement noise, missing measurements).
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CYBER = ("message_age_mean", "message_age_max", "delivery_fraction", "stale_rejection_count", "telemetry_age")
TELEMETRY = "telemetry-grid"


@dataclass(frozen=True)
class Perturbation:
    kind: str  # none, clock_offset, clock_drift, timestamp_dropout, sequence_corruption,
    #            metadata_loss, metadata_delay, physical_noise, physical_missing
    level: float = 0.0

    @property
    def label(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}={self.level:g}"


# Prespecified severity grid (frozen before any perturbed scoring).
GRID: tuple[Perturbation, ...] = (
    Perturbation("none"),
    *(Perturbation("clock_offset", s) for s in (1, 2, 5)),
    *(Perturbation("clock_drift", s) for s in (1, 2, 5)),
    *(Perturbation("timestamp_dropout", q) for q in (0.05, 0.10, 0.25, 0.50)),
    *(Perturbation("sequence_corruption", q) for q in (0.01, 0.05, 0.10, 0.25)),
    *(Perturbation("metadata_loss", q) for q in (0.05, 0.10, 0.25, 0.50)),
    *(Perturbation("metadata_delay", k) for k in (1, 2, 5, 10)),
    *(Perturbation("physical_noise", m) for m in (0.10, 0.25, 0.50, 1.00)),
    *(Perturbation("physical_missing", q) for q in (0.05, 0.10, 0.25, 0.50)),
)


def ledger_path(simulation_root: Path, scenario_id: str) -> Path:
    public = simulation_root / scenario_id / "event_ledger.json"
    return public if public.exists() else simulation_root / "restricted" / f"{scenario_id}.events.json"


def load_ledger(simulation_root: Path, scenario_id: str) -> list[dict]:
    return json.loads(ledger_path(simulation_root, scenario_id).read_text(encoding="utf-8"))


def _rng(scenario_id: str, perturbation: Perturbation, seed: int) -> np.random.Generator:
    key = zlib.crc32(f"{scenario_id}|{perturbation.label}".encode())
    return np.random.default_rng([seed, key])


def replay(ledger: list[dict], steps: int, perturbation: Perturbation = Perturbation("none"),
           rng: np.random.Generator | None = None, *, indicators: dict | None = None) -> np.ndarray:
    """Return the five cyber channels, shape (5, steps), under ``perturbation``.

    If ``indicators`` is a dict, receiver-observable trust counters are written
    into it: accepted messages without a timestamp, accepted messages whose
    timestamp lies in the receiver's future, and stale (non-newer) rejections.
    """
    rng = rng or np.random.default_rng(0)
    events = [e for e in ledger if not e["dropped"]]
    endpoints = sorted({e["endpoint_id"] for e in ledger if e["endpoint_id"] != TELEMETRY})
    all_ids = endpoints + [TELEMETRY]
    max_seq = max((e["sequence"] for e in ledger), default=0)

    observed_production = {id(e): int(e["production_time"]) for e in events}
    has_timestamp = {id(e): True for e in events}
    observed_sequence = {id(e): int(e["sequence"]) for e in events}
    kind, level = perturbation.kind, perturbation.level
    if kind == "clock_offset":
        offsets = {ep: int(rng.integers(-int(level), int(level) + 1)) for ep in all_ids}
        for e in events:
            observed_production[id(e)] += offsets[e["endpoint_id"]]
    elif kind == "clock_drift":
        rates = {ep: float(rng.uniform(-level, level)) / steps for ep in all_ids}
        for e in events:
            observed_production[id(e)] += int(round(rates[e["endpoint_id"]] * e["production_time"]))
    elif kind == "timestamp_dropout":
        for e in events:
            has_timestamp[id(e)] = bool(rng.random() >= level)
    elif kind == "sequence_corruption":
        for e in events:
            if rng.random() < level:
                observed_sequence[id(e)] = int(rng.integers(0, max_seq + 1))

    by_step: dict[int, list[dict]] = {}
    for e in events:
        by_step.setdefault(int(e["actual_delivery_time"]), []).append(e)
    latest_seq: dict[str, int] = {}
    latest_stamp: dict[str, int] = {}
    accepted: set[str] = set()
    rejected = 0
    missing_stamps = 0
    future_stamps = 0
    out = np.empty((5, steps), dtype=np.float32)
    for t in range(steps):
        for e in sorted(by_step.get(t, ()), key=lambda ev: ev["sequence"]):
            ep = e["endpoint_id"]
            seq = observed_sequence[id(e)]
            if ep in latest_seq and seq <= latest_seq[ep]:
                rejected += 1
                continue
            latest_seq[ep] = seq
            accepted.add(ep)
            if has_timestamp[id(e)]:
                latest_stamp[ep] = observed_production[id(e)]
                if observed_production[id(e)] > t:
                    future_stamps += 1
            else:
                missing_stamps += 1

        def age(ep: str) -> float:
            if ep in latest_stamp:
                return float(max(0, t - latest_stamp[ep])) if kind in ("clock_offset", "clock_drift") else float(t - latest_stamp[ep])
            return float(t + 1)

        ages = [age(ep) for ep in endpoints]
        out[0, t] = float(np.mean(ages)) if ages else 0.0
        out[1, t] = max(ages, default=0.0)
        out[2, t] = sum(ep in accepted for ep in endpoints) / max(1, len(endpoints))
        out[3, t] = rejected
        out[4, t] = age(TELEMETRY)
    if indicators is not None:
        indicators.update(missing_timestamps=missing_stamps, future_timestamps=future_stamps, stale_rejections=rejected)
    return out


def _hold_missing(series: np.ndarray, missing: np.ndarray) -> np.ndarray:
    result = series.copy()
    for t in range(1, series.shape[-1]):
        result[..., t] = np.where(missing[..., t], result[..., t - 1], series[..., t])
    return result


def perturb_trace(raw: np.ndarray, columns: list[str], ledger: list[dict], perturbation: Perturbation,
                  rng: np.random.Generator, physical_channels: list[str], physical_std: np.ndarray,
                  indicators: dict | None = None) -> np.ndarray:
    """Return a perturbed copy of one run's raw trace (columns x steps).

    ``indicators`` (optional dict) receives the receiver's trust counters.
    """
    trace = np.array(raw, dtype=np.float32, copy=True)
    steps = trace.shape[-1]
    cyber_index = [columns.index(c) for c in CYBER]
    phys_index = [columns.index(c) for c in physical_channels]
    kind, level = perturbation.kind, perturbation.level
    replayed = kind in ("clock_offset", "clock_drift", "timestamp_dropout", "sequence_corruption")
    if not replayed and indicators is not None:
        indicators.update(missing_timestamps=0, future_timestamps=0,
                          stale_rejections=int(trace[columns.index("stale_rejection_count"), -1]))
    if replayed:
        trace[cyber_index] = replay(ledger, steps, perturbation, rng, indicators=indicators)
    elif kind == "metadata_loss":
        missing = rng.random((len(cyber_index), steps)) < level
        missing[:, 0] = False
        trace[cyber_index] = _hold_missing(trace[cyber_index], missing)
    elif kind == "metadata_delay":
        k = int(level)
        delayed = trace[cyber_index].copy()
        delayed[:, k:] = trace[cyber_index][:, : steps - k]
        delayed[:, :k] = trace[cyber_index][:, :1]
        trace[cyber_index] = delayed
    elif kind == "physical_noise":
        noise = rng.normal(0.0, 1.0, (len(phys_index), steps)) * (level * physical_std[:, None])
        trace[phys_index] = trace[phys_index] + noise.astype(np.float32)
    elif kind == "physical_missing":
        missing = rng.random((len(phys_index), steps)) < level
        missing[:, 0] = False
        trace[phys_index] = _hold_missing(trace[phys_index], missing)
    return trace
