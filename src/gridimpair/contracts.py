from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonical(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values cannot be hashed")
        return 0.0 if value == 0.0 else value
    if hasattr(value, "item") and callable(value.item):
        return _canonical(value.item())
    return value


def stable_hash(value: Mapping[str, Any] | Any) -> str:
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MessageEvent:
    production_time: int
    intended_delivery_time: int
    actual_delivery_time: int | None
    sequence: int
    endpoint_id: str
    payload: tuple[float, ...]
    dropped: bool
    payload_hash: str

    def __post_init__(self) -> None:
        if self.production_time < 0 or self.intended_delivery_time < self.production_time:
            raise ValueError("message times must be causal and non-negative")
        if self.dropped != (self.actual_delivery_time is None):
            raise ValueError("dropped messages must have no actual delivery time")
        if self.actual_delivery_time is not None and self.actual_delivery_time < self.production_time:
            raise ValueError("actual delivery cannot predate production")


def make_message_event(
    *,
    production_time: int,
    intended_delivery_time: int,
    actual_delivery_time: int | None,
    sequence: int,
    endpoint_id: str,
    payload: tuple[float, ...],
    dropped: bool,
) -> MessageEvent:
    payload_tuple = tuple(float(value) for value in payload)
    return MessageEvent(
        production_time=production_time,
        intended_delivery_time=intended_delivery_time,
        actual_delivery_time=actual_delivery_time,
        sequence=sequence,
        endpoint_id=endpoint_id,
        payload=payload_tuple,
        dropped=dropped,
        payload_hash=stable_hash({"endpoint_id": endpoint_id, "payload": payload_tuple}),
    )


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    pair_id: str
    split: str
    regime: str
    delay_steps: int
    drop_probability: float
    endpoint_fraction: float
    status: str
    profile_seed: int
    schedule_seed: int
    impairment_seed: int
    network: str = "cigre_mv"
    impairment_kind: str = "bernoulli"
    burst_enter_probability: float = 0.08
    burst_exit_probability: float = 0.25
    jitter_min_steps: int = 0
    jitter_max_steps: int = 5

    def __post_init__(self) -> None:
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        for name in ("drop_probability", "endpoint_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.status not in {"clean", "impaired"}:
            raise ValueError("status must be clean or impaired")
        if self.network not in {"cigre_mv", "ieee33_der"}:
            raise ValueError("network must be cigre_mv or ieee33_der")
        if self.impairment_kind not in {"bernoulli", "burst", "jitter"}:
            raise ValueError(
                "impairment_kind must be bernoulli, burst, or jitter"
            )
        if not 0 <= self.burst_enter_probability <= 1:
            raise ValueError("burst_enter_probability must be in [0, 1]")
        if not 0 <= self.burst_exit_probability <= 1:
            raise ValueError("burst_exit_probability must be in [0, 1]")
        if self.jitter_min_steps > self.jitter_max_steps:
            raise ValueError(
                "jitter_min_steps cannot exceed jitter_max_steps"
            )


@dataclass(frozen=True)
class RunSummary:
    scenario_id: str
    pair_id: str
    status: str
    steps: int
    converged_steps: int
    event_count: int
    trace_path: str
    artifact_hash: str


@dataclass(frozen=True)
class DetectorDecision:
    logical_time: int
    score: float
    alarm: bool


@dataclass(frozen=True)
class ControlAction:
    p_mw: tuple[float, ...] = ()
    q_mvar: tuple[float, ...] = ()
    shed_fraction: float = 0.0
    safe_mode: bool = False


@dataclass(frozen=True)
class PowerState:
    logical_time: int
    vm_pu: tuple[float, ...]
    loading_percent: tuple[float, ...]
    soc: tuple[float, ...] = ()
    p_mw: tuple[float, ...] = ()
    q_mvar: tuple[float, ...] = ()
    i_ka: tuple[float, ...] = ()
    load_p_mw: float = 0.0
    generation_p_mw: float = 0.0
    converged: bool = True


@dataclass(frozen=True)
class ModelOutput:
    bag_logit: "torch.Tensor"
    instance_logits: "torch.Tensor"
    attention: "torch.Tensor | None" = None
