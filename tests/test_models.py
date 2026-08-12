from __future__ import annotations

import numpy as np
import pytest
import torch

from gridimpair.models.candidate import PhysicsResidualMIL
from gridimpair.models.classical import CausalRidgeResidual
from gridimpair.models.heuristics import (
    AgeWatchdog,
    DerivativeEnergyDetector,
    EwmaCusumDetector,
    FlatlineDetector,
)
from gridimpair.models.tail_mil import TailMIL
from gridimpair.models.tcn import StandardCausalTCN


def test_heuristics_score_only_the_supplied_causal_prefix() -> None:
    prefix = np.array([[1.0, 0.0], [1.1, 0.1], [1.2, 0.2]])
    extended = np.vstack([prefix, [[999.0, -999.0]]])
    detectors = [
        FlatlineDetector(),
        DerivativeEnergyDetector(),
        EwmaCusumDetector(),
    ]
    for detector in detectors:
        detector.fit(prefix, split="development")
        assert detector.score(prefix) == detector.score(extended[: len(prefix)])
    watchdog = AgeWatchdog(max_age=2)
    assert watchdog.score(np.array([0, 1, 3])) == 1.0


def test_ridge_residual_uses_strictly_past_lags() -> None:
    series = np.arange(24, dtype=float).reshape(12, 2)
    model = CausalRidgeResidual(lags=2, alpha=1.0).fit(
        series, split="development"
    )
    baseline = model.transform(series)
    changed = series.copy()
    changed[8:] += 1000
    modified = model.transform(changed)
    np.testing.assert_allclose(baseline[:8], modified[:8])


def test_ridge_fit_many_drops_lag_rows_at_every_scenario_boundary() -> None:
    first = np.arange(12, dtype=float).reshape(6, 2)
    second = 1000 + np.arange(12, dtype=float).reshape(6, 2)
    model = CausalRidgeResidual(lags=2, alpha=1.0).fit_many(
        [first, second],
        split="development",
    )
    assert model.training_rows_ == 8


def test_neural_models_preserve_instance_length() -> None:
    raw = torch.randn(2, 4, 16)
    residual = torch.randn(2, 4, 16)
    models_and_inputs = [
        (StandardCausalTCN(channels=4, hidden=8), (raw,)),
        (TailMIL(channels=4, hidden=8), (raw,)),
        (PhysicsResidualMIL(channels=4, hidden=8), (raw, residual)),
    ]
    with torch.no_grad():
        for model, args in models_and_inputs:
            output = model(*args)
            assert output.bag_logit.shape == (2,)
            assert output.instance_logits.shape == (2, 16)


def test_causal_tcn_future_suffix_does_not_change_prefix_logits() -> None:
    torch.manual_seed(4)
    model = StandardCausalTCN(channels=3, hidden=8).eval()
    sample = torch.randn(1, 3, 12)
    changed = sample.clone()
    changed[:, :, 8:] = 500
    with torch.no_grad():
        original = model(sample).instance_logits[:, :8]
        modified = model(changed).instance_logits[:, :8]
    torch.testing.assert_close(original, modified)


def test_candidate_is_small_and_rejects_forbidden_metadata() -> None:
    model = PhysicsResidualMIL(channels=18, hidden=24)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 1_000_000
    raw = torch.randn(1, 18, 8)
    residual = torch.randn(1, 18, 8)
    with pytest.raises(TypeError):
        model(raw=raw, residual=residual, message_age=torch.ones(1, 8))


def test_single_candidate_forward_backward_has_finite_loss() -> None:
    torch.manual_seed(7)
    model = PhysicsResidualMIL(channels=3, hidden=8)
    output = model(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.bag_logit, torch.tensor([0.0, 1.0])
    )
    loss.backward()
    assert torch.isfinite(loss)
