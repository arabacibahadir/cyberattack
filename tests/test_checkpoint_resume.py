from __future__ import annotations

from pathlib import Path

import torch

from gridimpair.train.checkpoints import (
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_restores_model_optimizer_and_rng_after_one_step(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    saved_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=1,
        config_hash="cfg",
        split_hash="split",
        best_metric=0.75,
        training_state={"best_epoch": 1, "stale_epochs": 0},
    )
    path = save_checkpoint(payload, tmp_path / "last.pt")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    metadata = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_config_hash="cfg",
        expected_split_hash="split",
        map_location=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    assert metadata.epoch == 1
    assert metadata.best_metric == 0.75
    assert metadata.training_state == {"best_epoch": 1, "stale_epochs": 0}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_weights[name])


def test_checkpoint_rejects_hash_mismatch(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = save_checkpoint(
        checkpoint_payload(
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=0,
            config_hash="cfg",
            split_hash="split",
            best_metric=None,
        ),
        tmp_path / "last.pt",
    )
    try:
        load_checkpoint(path, model=model, expected_config_hash="other")
    except ValueError as exc:
        assert "config hash" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")
