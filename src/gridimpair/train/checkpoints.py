from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    config_hash: str
    split_hash: str
    best_metric: float | None
    training_state: dict[str, Any]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    config_hash: str,
    split_hash: str,
    best_metric: float | None,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "config_hash": config_hash,
        "split_hash": split_hash,
        "best_metric": best_metric,
        "training_state": dict(training_state or {}),
        "rng": _rng_state(),
    }


def save_checkpoint(payload: dict[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    expected_config_hash: str | None = None,
    expected_split_hash: str | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> CheckpointMetadata:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if (
        expected_config_hash is not None
        and payload["config_hash"] != expected_config_hash
    ):
        raise ValueError("checkpoint config hash mismatch")
    if expected_split_hash is not None and payload["split_hash"] != expected_split_hash:
        raise ValueError("checkpoint split hash mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload["scaler"] is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if torch.cuda.is_available() and "torch_cuda" in rng:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in rng["torch_cuda"]]
            )
    return CheckpointMetadata(
        epoch=int(payload["epoch"]),
        config_hash=str(payload["config_hash"]),
        split_hash=str(payload["split_hash"]),
        best_metric=payload.get("best_metric"),
        training_state=dict(payload.get("training_state", {})),
    )
