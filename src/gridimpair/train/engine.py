from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from gridimpair.config import TrainingConfig
from gridimpair.contracts import ModelOutput, stable_hash
from gridimpair.train.checkpoints import (
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class LoaderBundle:
    train: Iterable
    validation: Iterable
    split_hash: str


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_metric: float
    completed_epochs: int
    checkpoint_path: str


def _choose_runtime(precision: str, device_name: str) -> tuple[torch.device, torch.dtype | None]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    if precision == "fp32" or device.type == "cpu":
        return device, None
    if precision == "bf16" or (
        precision == "auto"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return device, torch.bfloat16
    return device, torch.float16


def _forward_batch(model: torch.nn.Module, batch: Any, device: torch.device) -> tuple[ModelOutput, torch.Tensor]:
    if len(batch) == 2:
        inputs, labels = batch
        output = model(inputs.to(device))
    elif len(batch) == 3:
        raw, residual, labels = batch
        output = model(raw.to(device), residual.to(device))
    else:
        raise ValueError("batch must be (inputs, labels) or (raw, residual, labels)")
    return output, labels.to(device=device, dtype=torch.float32)


def _run_epoch(
    model: torch.nn.Module,
    batches: Iterable,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    autocast_dtype: torch.dtype | None,
    scaler: Any,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch in batches:
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.set_grad_enabled(training), context:
            output, labels = _forward_batch(model, batch, device)
            loss = F.binary_cross_entropy_with_logits(output.bag_logit, labels)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def _append_metrics(run_dir: Path, row: dict[str, Any]) -> None:
    csv_path = run_dir / "metrics.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def fit_model(
    model: torch.nn.Module,
    loaders: LoaderBundle,
    train_config: TrainingConfig,
    run_dir: Path,
    *,
    precision: str = "auto",
    device_name: str = "auto",
    resume: bool = False,
) -> FitResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device, autocast_dtype = _choose_runtime(precision, device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.max_epochs
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda" and autocast_dtype == torch.float16
        else None
    )
    config_hash = stable_hash(asdict(train_config))
    start_epoch = 1
    best_metric = float("inf")
    best_epoch = 0
    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        metadata = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_hash=config_hash,
            expected_split_hash=loaders.split_hash,
            map_location=device,
        )
        start_epoch = metadata.epoch + 1
        best_metric = metadata.best_metric if metadata.best_metric is not None else best_metric
        best_epoch = int(metadata.training_state.get("best_epoch", 0))
    stale_epochs = (
        int(metadata.training_state.get("stale_epochs", 0))
        if resume and last_path.exists()
        else 0
    )
    completed = start_epoch - 1
    for epoch in range(start_epoch, train_config.max_epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = _run_epoch(
            model,
            loaders.train,
            device=device,
            optimizer=optimizer,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
        )
        validation_loss = _run_epoch(
            model,
            loaders.validation,
            device=device,
            optimizer=None,
            autocast_dtype=autocast_dtype,
            scaler=None,
        )
        scheduler.step()
        improved = validation_loss < best_metric
        if improved:
            best_metric, best_epoch, stale_epochs = validation_loss, epoch, 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "is_best": int(improved),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0
            ),
        }
        _append_metrics(run_dir, row)
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            config_hash=config_hash,
            split_hash=loaders.split_hash,
            best_metric=best_metric,
            training_state={
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            },
        )
        save_checkpoint(payload, last_path)
        if improved:
            save_checkpoint(payload, run_dir / "best.pt")
        if train_config.final_refit and epoch % train_config.checkpoint_every == 0:
            save_checkpoint(payload, run_dir / f"epoch-{epoch:03d}.pt")
        completed = epoch
        if epoch >= train_config.min_epochs and stale_epochs >= train_config.patience:
            break
    return FitResult(
        best_epoch=best_epoch,
        best_metric=best_metric,
        completed_epochs=completed,
        checkpoint_path=str(last_path.resolve()),
    )
