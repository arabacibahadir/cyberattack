from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    artifact_root: Path


@dataclass(frozen=True)
class RuntimeConfig:
    simulation_workers: int = 4
    precision: str = "auto"
    seed: int = 2026
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 1 <= self.simulation_workers <= 4:
            raise ValueError("simulation_workers must be between 1 and 4")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision must be auto, bf16, fp16, or fp32")


@dataclass(frozen=True)
class SimulationConfig:
    steps: int = 900
    network: str = "cigre_mv"
    recycle: bool = False
    storage_duration_hours: float = 2.0
    step_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("simulation.steps must be positive")
        if self.network not in {"cigre_mv", "ieee33_der"}:
            raise ValueError(
                "simulation.network must be cigre_mv or ieee33_der"
            )
        if self.storage_duration_hours <= 0:
            raise ValueError("simulation.storage_duration_hours must be positive")
        if self.step_seconds <= 0:
            raise ValueError("simulation.step_seconds must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 80
    min_epochs: int = 12
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_every: int = 10
    final_refit: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("training epochs must satisfy 1 <= min_epochs <= max_epochs")
        if self.patience < 1:
            raise ValueError("training.patience must be positive")


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    paths: PathConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            return tomllib.load(handle)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported config format: {path.suffix}")


def _known_fields(dataclass_type: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {field.name for field in dataclasses.fields(dataclass_type)}
    return {name: value for name, value in values.items() if name in names}


def load_config(path: Path | str) -> AppConfig:
    source = Path(path).resolve()
    raw = _load_mapping(source)
    base = source.parent
    paths = raw.get("paths", {})
    if "data_root" not in paths or "artifact_root" not in paths:
        raise ValueError("paths.data_root and paths.artifact_root are required")
    return AppConfig(
        source_path=source,
        paths=PathConfig(
            data_root=_resolve(base, paths["data_root"]),
            artifact_root=_resolve(base, paths["artifact_root"]),
        ),
        runtime=RuntimeConfig(
            **_known_fields(RuntimeConfig, raw.get("runtime", {}))
        ),
        simulation=SimulationConfig(
            **_known_fields(SimulationConfig, raw.get("simulation", {}))
        ),
        training=TrainingConfig(
            **_known_fields(TrainingConfig, raw.get("training", {}))
        ),
        raw=raw,
    )
